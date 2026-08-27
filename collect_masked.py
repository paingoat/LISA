"""Copy every *_masked_img_*.jpg from vis_output run folders into one tree.

    python collect_masked.py
    python collect_masked.py --src vis_output --out vis_output_masked

Layout:

    vis_output_masked/anime/anime_20260827_210530/anime_masked_img_0.jpg
    vis_output_masked/knight/knight_20260827_210612/knight_masked_img_0.jpg
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

RUN_DIR_RE = re.compile(r"^(.+)_(\d{8}_\d{6}(?:_\d+)?)$")
MASKED_RE = re.compile(r".+_masked_img_.+\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Collect masked_img files by character")
    parser.add_argument("--src", default="./vis_output", help="batch output root")
    parser.add_argument(
        "--out",
        default="./vis_output_masked",
        help="destination root, split by character",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="put files directly in {out}/{character}/ (overwrites on name clash)",
    )
    return parser.parse_args(argv)


def character_from_run_dir(name):
    match = RUN_DIR_RE.match(name)
    if match:
        return match.group(1)
    return name


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
        character = character_from_run_dir(run_name)
        for filename in sorted(os.listdir(run_dir)):
            if not MASKED_RE.match(filename):
                continue
            src_file = os.path.join(run_dir, filename)
            if args.flat:
                dest_dir = os.path.join(out, character)
                dest_file = os.path.join(dest_dir, filename)
            else:
                dest_dir = os.path.join(out, character, run_name)
                dest_file = os.path.join(dest_dir, filename)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            copied += 1
            print(dest_file)

    print("Copied {} masked_img files to {}".format(copied, out))
    if skipped:
        print("Skipped {}".format(skipped))


if __name__ == "__main__":
    main(sys.argv[1:])
