import os
from datetime import datetime

import cv2
import numpy as np


def image_stem(image_path):
    name = os.path.splitext(os.path.basename(os.path.normpath(str(image_path))))[0]
    return name or "image"


def save_segmentation_run(vis_save_path, image_path, image_np, pred_masks):
    """Save one run under vis_output/{stem}_{timestamp}/ with the original two files.

    Files match chat.py: {stem}_mask_{i}.jpg and {stem}_masked_img_{i}.jpg.
    Returns (run_dir or None, last overlay RGB or None).
    """
    stem = image_stem(image_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(vis_save_path, "{}_{}".format(stem, stamp))
    if os.path.exists(run_dir):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(vis_save_path, "{}_{}".format(stem, stamp))

    last_overlay_rgb = None
    saved = False
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        if not saved:
            os.makedirs(run_dir, exist_ok=True)
            saved = True

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

    if not saved:
        return None, None
    return run_dir, last_overlay_rgb
