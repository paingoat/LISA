import json
import os
from datetime import datetime

import cv2
import numpy as np


def image_stem(image_path):
    name = os.path.splitext(os.path.basename(os.path.normpath(str(image_path))))[0]
    return name or "image"


def save_segmentation_run(
    vis_save_path, image_path, image_np, pred_masks, prompt, text_output=None
):
    """Save one run under vis_output/{stem}_{timestamp}/.

    Files match chat.py: {stem}_mask_{i}.jpg and {stem}_masked_img_{i}.jpg,
    plus a JSON with the user prompt sent to the LLM.
    Returns (run_dir, last overlay RGB or None).
    """
    stem = image_stem(image_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(vis_save_path, "{}_{}".format(stem, stamp))
    if os.path.exists(run_dir):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(vis_save_path, "{}_{}".format(stem, stamp))
    os.makedirs(run_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "{}.json".format(stem))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": prompt,
                "image": os.path.basename(os.path.normpath(str(image_path))),
                "text_output": text_output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("{} has been saved.".format(meta_path))

    last_overlay_rgb = None
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        mask_path = os.path.join(run_dir, "{}_mask_{}.jpg".format(stem, i))
        cv2.imwrite(mask_path, pred_mask * 100)
        print("{} has been saved.".format(mask_path))

        overlay = image_np.copy()
        overlay[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        last_overlay_rgb = overlay
        img_path = os.path.join(run_dir, "{}_masked_img_{}.jpg".format(stem, i))
        cv2.imwrite(img_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("{} has been saved.".format(img_path))

    return run_dir, last_overlay_rgb
