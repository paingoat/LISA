"""Collect masked_img files into {out}/{character}/{part}.jpg

Part name comes from the saved prompt JSON, e.g.
"Please segment the character's head" -> head
"Please segment the character's left arm" -> left_arm

    python collect_masked.py
    python collect_masked.py --src vis_output --out vis_output_masked
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

RUN_DIR_RE = re.compile(r"^(.+)_(\d{8}_\d{6}(?:_\d+)?)$")
MASKED_RE = re.compile(r".+_masked_img_.+\.(jpg|jpeg|png)$", re.IGNORECASE)
PART_RE = re.compile(
    r"character's\s+(.+?)(?:\s+please output segmentation mask\.?)?$",
    re.IGNORECASE,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect masked_img files by character and body part"
    )
    parser.add_argument("--src", default="./vis_output", help="batch output root")
    parser.add_argument(
        "--out",
        default="./vis_output_masked",
        help="destination root, split by character",
    )
    return parser.parse_args(argv)


def character_from_run_dir(name):
    match = RUN_DIR_RE.match(name)
    if match:
        return match.group(1)
    return name


def load_prompt(run_dir):
    for filename in os.listdir(run_dir):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(run_dir, filename)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        prompt = data.get("prompt")
        if prompt:
            return prompt
    return None


def part_from_prompt(prompt):
    text = " ".join(prompt.strip().split())
    match = PART_RE.search(text)
    if match:
        part = match.group(1)
    else:
        part = text
    part = re.sub(r"[^\w]+", "_", part.strip().lower()).strip("_")
    return part or "unknown"


def main(argv):
    args = parse_args(argv)
    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    if not os.path.isdir(src):
        print("Source not found: {}".format(src))
        sys.exit(1)

    copied = 0
    skipped = 0
    os.makedirs(out, exist_ok=True)

    for run_name in sorted(os.listdir(src)):
        run_dir = os.path.join(src, run_name)
        if not os.path.isdir(run_dir):
            continue

        prompt = load_prompt(run_dir)
        if not prompt:
            print("Skip (no prompt json): {}".format(run_dir))
            skipped += 1
            continue

        character = character_from_run_dir(run_name)
        part = part_from_prompt(prompt)
        masked = [
            name
            for name in sorted(os.listdir(run_dir))
            if MASKED_RE.match(name)
        ]
        if not masked:
            skipped += 1
            continue

        dest_dir = os.path.join(out, character)
        os.makedirs(dest_dir, exist_ok=True)
        for i, filename in enumerate(masked):
            ext = os.path.splitext(filename)[1]
            dest_name = "{}{}".format(part, ext) if len(masked) == 1 else "{}_{}{}".format(part, i, ext)
            dest_file = os.path.join(dest_dir, dest_name)
            shutil.copy2(os.path.join(run_dir, filename), dest_file)
            copied += 1
            print("{}  <-  {}".format(dest_file, prompt))

    print("Copied {} masked_img files to {}".format(copied, out))
    if skipped:
        print("Skipped {} run folders".format(skipped))


if __name__ == "__main__":
    main(sys.argv[1:])
