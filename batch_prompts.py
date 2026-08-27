"""Batch inference with the same load/evaluate path as chat.py, then exit.

Stop app.py first (A40 cannot hold two 13B bf16 processes). Then:

    CUDA_VISIBLE_DEVICES=0 python batch_prompts.py --precision=bf16
"""

from __future__ import annotations

import argparse
import os
import sys

from utils.hf_env import load_runtime_env

load_runtime_env()

import cv2
import torch
from transformers import AutoTokenizer, CLIPImageProcessor

from chat import preprocess
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.vis_save import save_segmentation_run

PROMPTS = [
    "Please segment the character's head",
    "Please segment the character's torso",
    "Please segment the character's left arm",
    "Please segment the character's right arm",
    "Please segment the character's left leg",
    "Please segment the character's right leg",
]

IMAGES = [
    "anime.png",
    "knight.png",
    "viking.png",
    "fei.png",
    "hollow.png",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Batch LISA prompts (chat.py path)")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
    )
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--imgs_dir", default="./imgs", type=str)
    parser.add_argument(
        "--add_mask_phrase",
        action="store_true",
        help='append " Please output segmentation mask." to each prompt',
    )
    return parser.parse_args(argv)


def load_model(args):
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

    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        torch_dtype=torch_dtype,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    model.eval()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    return tokenizer, model, clip_image_processor, transform, torch_dtype


def run_one(args, tokenizer, model, clip_image_processor, transform, image_path, user_prompt):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
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

    save_segmentation_run(
        args.vis_save_path,
        image_path,
        image_np,
        pred_masks,
        prompt=user_prompt,
        text_output=text_output,
    )
    return text_output


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.vis_save_path, exist_ok=True)

    imgs_dir = os.path.abspath(args.imgs_dir)
    image_paths = []
    missing = []
    for name in IMAGES:
        path = os.path.join(imgs_dir, name)
        if os.path.isfile(path):
            image_paths.append(path)
        else:
            missing.append(path)
    if missing:
        print("Missing images:")
        for path in missing:
            print("  " + path)
        sys.exit(1)

    prompts = list(PROMPTS)
    if args.add_mask_phrase:
        suffix = " Please output segmentation mask."
        prompts = [p.rstrip() + suffix for p in prompts]

    tokenizer, model, clip_image_processor, transform, _ = load_model(args)

    total = len(image_paths) * len(prompts)
    n = 0
    failed = 0
    for image_path in image_paths:
        for user_prompt in prompts:
            n += 1
            print(
                "[{}/{}] {} | {}".format(
                    n, total, os.path.basename(image_path), user_prompt
                )
            )
            try:
                text_output = run_one(
                    args,
                    tokenizer,
                    model,
                    clip_image_processor,
                    transform,
                    image_path,
                    user_prompt,
                )
                print("text_output: ", text_output)
            except Exception as exc:
                failed += 1
                print("FAILED:", exc)

    print("Done. {}/{} failed. Outputs in {}".format(failed, total, args.vis_save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
