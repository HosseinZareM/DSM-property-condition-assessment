#!/usr/bin/env python3
"""
Utility to randomly and evenly sample images from class folders into a single folder.

Default behavior:
- Source root: /home/exouser/DSM-property-condition-assessment/Data/extractedimages
- Classes: NHTyp1..NHTyp5
- Copy 4 images per class (random without replacement)
- Destination: /home/exouser/DSM-property-condition-assessment/Data/randomly-image-test

Filenames in destination are prefixed with the class, e.g. NHTyp3__original.jpg
"""

import argparse
import os
import random
import shutil
from typing import Iterable, List


def list_jpg_files(directory_path: str) -> List[str]:
    return [
        filename for filename in os.listdir(directory_path)
        if filename.lower().endswith(".jpg")
    ]


def sample_even_images(
    source_root: str,
    class_folders: Iterable[str],
    images_per_class: int,
    dest_folder: str,
    seed: int = 42,
) -> int:
    random.seed(seed)
    os.makedirs(dest_folder, exist_ok=True)

    total_copied = 0
    for class_folder in class_folders:
        class_path = os.path.join(source_root, class_folder)
        if not os.path.isdir(class_path):
            print(f"Skipping missing folder: {class_path}")
            continue

        jpg_files = list_jpg_files(class_path)

        if not jpg_files:
            print(f"No .jpg files found in {class_path}")
            continue

        if len(jpg_files) < images_per_class:
            print(
                f"Only {len(jpg_files)} images in {class_folder}; copying all available."
            )
            chosen = jpg_files
        else:
            chosen = random.sample(jpg_files, images_per_class)

        for filename in chosen:
            src_path = os.path.join(class_path, filename)
            dest_name = f"{class_folder}__{filename}"
            dest_path = os.path.join(dest_folder, dest_name)
            shutil.copy2(src_path, dest_path)
            total_copied += 1

    return total_copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly and evenly sample images from class folders into one folder",
    )
    parser.add_argument(
        "--source-root",
        default="/home/exouser/DSM-property-condition-assessment/Data/extractedimages",
        help="Root folder containing class subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["NHTyp1", "NHTyp2", "NHTyp3", "NHTyp4", "NHTyp5"],
        help="Class subfolder names (default: %(default)s)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=4,
        help="Number of images per class to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--dest",
        default="/home/exouser/DSM-property-condition-assessment/Data/randomly-image-test",
        help="Destination folder for sampled images (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s)",
    )

    args = parser.parse_args()

    print("Sampling images with configuration:")
    print(f"  source_root: {args.source_root}")
    print(f"  classes: {args.classes}")
    print(f"  images_per_class: {args.per_class}")
    print(f"  dest_folder: {args.dest}")
    print(f"  seed: {args.seed}")

    copied = sample_even_images(
        source_root=args.source_root,
        class_folders=args.classes,
        images_per_class=args.per_class,
        dest_folder=args.dest,
        seed=args.seed,
    )

    print(f"Done. Copied {copied} images to {args.dest}")


if __name__ == "__main__":
    main()


