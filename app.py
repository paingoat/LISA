import argparse
import os
import re
import sys

from utils.hf_env import load_runtime_env

load_runtime_env()

import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.vis_save import save_segmentation_run


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument(
        "--server_name",
        default="0.0.0.0",
        type=str,
        help="bind address; 0.0.0.0 so RunPod/HTTP proxy can reach Gradio",
    )
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        default=True,
        help="create a public Gradio URL (*.gradio.live)",
    )
    parser.add_argument(
        "--no-share",
        dest="share",
        action="store_false",
        help="disable the public Gradio tunnel",
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
tokenizer = AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half

kwargs = {"torch_dtype": torch_dtype}
if args.load_in_4bit or args.load_in_8bit:
    from transformers import BitsAndBytesConfig

    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    else:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

model = LISAForCausalLM.from_pretrained(
    args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif (
    args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
):
    # Same eager cast as bf16. Original DeepSpeed kernel inject is not
    # required for chat-equivalent quality and needs an extra package.
    model = model.half().cuda()
elif args.precision == "fp32":
    model = model.float().cuda()

vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device=args.local_rank)

clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
transform = ResizeLongestSide(args.image_size)

model.eval()


# Gradio
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_IMGS_DIR = os.path.join(_REPO_ROOT, "imgs")


def _placeholder_rgb(message="No output"):
    img = np.full((256, 256, 3), 40, dtype=np.uint8)
    cv2.putText(
        img,
        message,
        (16, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return img


def _load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


examples = [
    pair
    for pair in [
        [
            "Can you segment the character in this image? Please output segmentation mask.",
            os.path.join(_IMGS_DIR, "anime.png"),
        ],
        [
            "Can you segment the knight in this image? Please output segmentation mask.",
            os.path.join(_IMGS_DIR, "knight.png"),
        ],
        [
            "Can you segment the unusual part in this image? Please output segmentation mask and explain why.",
            os.path.join(_IMGS_DIR, "hollow.png"),
        ],
        [
            "Can you segment the person in this image? Please output segmentation mask.",
            os.path.join(_IMGS_DIR, "viking.png"),
        ],
    ]
    if os.path.isfile(pair[1])
]
output_labels = ["Segmentation Output"]

title = "LISA: Reasoning Segmentation via Large Language Model"

description = """
<font size=4>
This is the online demo of LISA. \n
If multiple users are using it at the same time, they will enter a queue, which may delay some time. \n
**Note**: **Different prompts can lead to significantly varied results**. \n
**Note**: Please try to **standardize** your input text prompts to **avoid ambiguity**, and also pay attention to whether the **punctuations** of the input are correct. \n
**Note**: Default checkpoint is **LISA-13B-llama2-v1** in bf16 (same as `chat.py`). \n
**Usage**: <br>
&ensp;(1) To let LISA **segment something**, input prompt like: "Can you segment xxx in this image?", "What is xxx in this image? Please output segmentation mask."; <br>
&ensp;(2) To let LISA **output an explanation**, input prompt like: "What is xxx in this image? Please output segmentation mask and explain why."; <br>
&ensp;(3) To obtain **solely language output**, you can input like what you should do in current multi-modal LLM (e.g., LLaVA). <br>
Hope you can enjoy our work!
</font>
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2308.00692' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='https://github.com/dvlab-research/LISA' target='_blank'>   Github Repo </a></p>
"""


## to be implemented
def inference(input_str, input_image):
    ## filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    ## input valid check
    if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
        output_str = "[Error] Invalid input: ", input_str
        # output_image = np.zeros((128, 128, 3))
        ## error happened
        output_image = _load_rgb(os.path.join(_REPO_ROOT, "resources", "error_happened.png"))
        if output_image is None:
            output_image = _placeholder_rgb("Invalid input")
        return output_image, output_str

    # Model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = input_str
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()


    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    print("text_output: ", text_output)
    _, save_img = save_segmentation_run(
        args.vis_save_path,
        input_image,
        image_np,
        pred_masks,
        prompt=input_str,
        text_output=text_output,
    )

    output_str = "ASSITANT: " + text_output  # input_str
    if save_img is not None:
        output_image = save_img  # input_image
    else:
        output_image = _load_rgb(os.path.join(_REPO_ROOT, "resources", "no_seg_out.png"))
        if output_image is None:
            output_image = _placeholder_rgb("No segmentation")
    return output_image, output_str


demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Image(type="pil", label="Segmentation Output"),
        gr.Textbox(lines=1, placeholder=None, label="Text Output"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples or None,
    allow_flagging="auto",
)

demo.queue()
demo.launch(
    server_name=args.server_name,
    server_port=args.port,
    share=args.share,
    inbrowser=False,
)
