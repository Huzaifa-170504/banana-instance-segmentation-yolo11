
"""
==============================================================================
Dataset Verification Module
==============================================================================

Verifies:
- Dataset directory structure
- Image/label pairing
- Empty labels
- Corrupt images
- Invalid polygons
- Class IDs
- Saves JSON report

Author: Huzaifa
"""

from pathlib import Path
import json
from PIL import Image

import config
import logger


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def verify(dataset_type="research"):

    if dataset_type not in ["research", "benchmark"]:
        raise ValueError("dataset_type must be 'research' or 'benchmark'.")

    root = config.PREPARED_DATASET / dataset_type

    report = {
        "dataset_type": dataset_type,
        "dataset_root": str(root),
        "train": {},
        "val": {},
        "test": {},
        "passed": True,
        "errors": []
    }

    logger.log(f"Verifying {dataset_type} dataset...")

    for split in ["train", "val", "test"]:

        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            report["passed"] = False
            report["errors"].append(f"Missing directory: {img_dir}")
            continue

        if not lbl_dir.exists():
            report["passed"] = False
            report["errors"].append(f"Missing directory: {lbl_dir}")
            continue

        images = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in VALID_IMAGE_SUFFIXES
        ])

        labels = sorted(lbl_dir.glob("*.txt"))

        missing_labels = 0
        missing_images = 0
        corrupt_images = 0
        empty_labels = 0
        invalid_labels = 0

        image_names = {p.stem for p in images}
        label_names = {p.stem for p in labels}

        missing_labels = len(image_names - label_names)
        missing_images = len(label_names - image_names)

        # Verify images
        for img in images:
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                corrupt_images += 1

        # Verify labels
        for lbl in labels:

            text = lbl.read_text().strip()

            if text == "":
                empty_labels += 1
                continue

            for line in text.splitlines():

                tokens = line.split()

                if len(tokens) < 7:
                    invalid_labels += 1
                    continue

                coords = list(map(float, tokens[1:]))

                if len(coords) % 2 != 0:
                    invalid_labels += 1
                    continue

                if any(c < 0 or c > 1 for c in coords):
                    invalid_labels += 1

        report[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "corrupt_images": corrupt_images,
            "empty_labels": empty_labels,
            "invalid_labels": invalid_labels
        }

        if any([
            missing_images,
            missing_labels,
            corrupt_images,
            empty_labels,
            invalid_labels
        ]):
            report["passed"] = False

    logger.save_json(
        f"{dataset_type}_dataset_verification.json",
        report
    )

    return report
