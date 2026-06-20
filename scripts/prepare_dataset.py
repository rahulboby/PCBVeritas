"""
Dataset Preparation Script
============================
PURPOSE:
    Converts the raw PCB dataset (Pascal VOC XML format) into YOLO format
    and creates train/val/test splits. Handles the crucial requirement
    that ROTATED images never appear in validation or test sets.

INPUT:
    data/raw/PCB_DATASET/ (Pascal VOC XML annotations)

OUTPUT:
    data/splits/
    ├── train/images/  (80% of originals + ALL rotated)
    ├── train/labels/  (YOLO format .txt files)
    ├── val/images/    (10% originals ONLY)
    ├── val/labels/
    ├── test/images/   (10% originals ONLY)
    ├── test/labels/
    └── dataset.yaml   (YOLO config file)

HOW IT WORKS:
    1. Finds all original and rotated images
    2. Converts Pascal VOC XML -> YOLO txt format
       YOLO format: class_id cx cy w h (all normalized 0-1)
    3. Splits ONLY original images into train/val/test (80/10/10)
    4. Adds ALL rotated images to TRAINING ONLY
    5. Creates dataset.yaml for Ultralytics

YOLO FORMAT EXPLAINED:
    Each .txt file corresponds to an image.
    Each line: class_id center_x center_y width height
    All values normalized to [0,1] by image dimensions.
    Example: "0 0.5 0.3 0.2 0.15" means class 0, centered at
    50% width, 30% height, occupying 20% width, 15% height.

USAGE:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --data-dir data/raw/PCB_DATASET --seed 42
"""

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import yaml

from loguru import logger
from rich.console import Console
from rich.progress import track
from rich import print as rprint

console = Console()

# Class mapping: name -> YOLO integer ID
CLASS_TO_ID = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5,
}

# Reverse mapping for reference
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

# Normalize alternate class names
CLASS_ALIASES = {
    "missing_hole": "missing_hole",
    "Missing_hole": "missing_hole",
    "mouse_bite": "mouse_bite",
    "Mouse_bite": "mouse_bite",
    "open_circuit": "open_circuit",
    "Open_circuit": "open_circuit",
    "short": "short",
    "Short": "short",
    "spur": "spur",
    "Spur": "spur",
    "spurious_copper": "spurious_copper",
    "Spurious_copper": "spurious_copper",
}


def parse_xml_annotation(xml_path: Path) -> tuple[int, int, list[dict]]:
    """
    Parse a Pascal VOC XML annotation file.

    Pascal VOC bboxes are in absolute pixel coordinates:
    xmin, ymin, xmax, ymax

    We need to convert to YOLO normalized format:
    center_x, center_y, width, height (all divided by image dims)

    Args:
        xml_path: Path to XML annotation file.

    Returns:
        Tuple of (image_width, image_height, list_of_annotations)
        where each annotation is {'class': str, 'bbox': [xmin,ymin,xmax,ymax]}
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse {xml_path}: {e}")
        return 0, 0, []

    # Get image size
    size = root.find("size")
    if size is None:
        logger.warning(f"No size element in {xml_path}")
        return 0, 0, []

    try:
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
    except (AttributeError, ValueError, TypeError):
        logger.warning(f"Invalid size in {xml_path}")
        return 0, 0, []

    annotations = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue

        raw_name = name_elem.text.strip()
        class_name = CLASS_ALIASES.get(raw_name, raw_name.lower().replace(" ", "_"))

        if class_name not in CLASS_TO_ID:
            logger.warning(f"Unknown class '{raw_name}' in {xml_path}")
            continue

        bbox_elem = obj.find("bndbox")
        if bbox_elem is None:
            continue

        try:
            xmin = float(bbox_elem.find("xmin").text)
            ymin = float(bbox_elem.find("ymin").text)
            xmax = float(bbox_elem.find("xmax").text)
            ymax = float(bbox_elem.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            logger.warning(f"Invalid bbox in {xml_path}")
            continue

        # Clamp values to image boundaries
        xmin = max(0, min(xmin, img_w))
        ymin = max(0, min(ymin, img_h))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))

        if xmin >= xmax or ymin >= ymax:
            logger.warning(f"Degenerate bbox in {xml_path}: [{xmin},{ymin},{xmax},{ymax}]")
            continue

        annotations.append({
            "class": class_name,
            "class_id": CLASS_TO_ID[class_name],
            "bbox": [xmin, ymin, xmax, ymax],
        })

    return img_w, img_h, annotations


def convert_to_yolo_format(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """
    Convert Pascal VOC absolute bbox to YOLO normalized format.

    Pascal VOC: [xmin, ymin, xmax, ymax] in pixels
    YOLO:       [cx, cy, w, h] normalized to [0,1]

    Args:
        xmin, ymin, xmax, ymax: Absolute pixel coordinates.
        img_w, img_h: Image dimensions.

    Returns:
        (center_x, center_y, width, height) all normalized.
    """
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    # Clamp to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def write_yolo_label(
    label_path: Path,
    annotations: list[dict],
    img_w: int,
    img_h: int,
) -> None:
    """
    Write YOLO format label file.

    Args:
        label_path: Where to save the .txt file.
        annotations: List of annotation dicts.
        img_w, img_h: Image dimensions for normalization.
    """
    lines = []
    for ann in annotations:
        xmin, ymin, xmax, ymax = ann["bbox"]
        cx, cy, w, h = convert_to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def find_annotation(image_path: Path, data_dir: Path) -> Optional[Path]:
    """
    Find the XML annotation file corresponding to an image.

    The PCB dataset typically stores annotations alongside images
    or in a parallel 'Annotations' directory.

    Args:
        image_path: Path to the image.
        data_dir: Dataset root directory.

    Returns:
        Path to XML annotation file, or None if not found.
    """
    # Try same directory
    xml_same_dir = image_path.with_suffix(".xml")
    if xml_same_dir.exists():
        return xml_same_dir

    # Try 'Annotations' subdirectory at same level
    ann_dir = image_path.parent.parent / "Annotations" / image_path.stem
    xml_ann_dir = ann_dir.with_suffix(".xml")
    if xml_ann_dir.exists():
        return xml_ann_dir

    # Try sibling directory named 'Annotations'
    for parent in image_path.parents:
        ann_candidate = parent / "Annotations" / (image_path.stem + ".xml")
        if ann_candidate.exists():
            return ann_candidate

    # Recursive search in data_dir
    candidates = list(data_dir.rglob(f"{image_path.stem}.xml"))
    if candidates:
        return candidates[0]

    return None


def prepare_dataset(
    data_dir: str = "data/raw/PCB_DATASET",
    output_dir: str = "data/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Main dataset preparation function.

    Args:
        data_dir: Raw dataset directory.
        output_dir: Where to save processed splits.
        train_ratio: Fraction of originals for training.
        val_ratio: Fraction of originals for validation.
        test_ratio: Fraction of originals for testing.
        seed: Random seed for reproducibility.

    Returns:
        Statistics dictionary.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(seed)

    data_path = Path(data_dir)
    out_path = Path(output_dir)

    console.rule("[bold blue]PCB Dataset Preparation")
    logger.info(f"Preparing dataset from: {data_path}")
    logger.info(f"Output directory: {out_path}")

    # --- Find all images ---
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(data_path.rglob(f"*{ext}"))
        all_images.extend(data_path.rglob(f"*{ext.upper()}"))

    if not all_images:
        logger.error(f"No images found in {data_path}")
        sys.exit(1)

    # --- Separate originals from rotated ---
    # Rotated images have 'rotation' in their filename
    original_images = sorted([p for p in all_images if "rotation" not in p.stem.lower()])
    rotated_images = sorted([p for p in all_images if "rotation" in p.stem.lower()])

    console.print(f"\nFound {len(original_images)} original images")
    console.print(f"Found {len(rotated_images)} rotated images")

    # --- Split ONLY original images ---
    random.shuffle(original_images)
    n_total = len(original_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_originals = original_images[:n_train]
    val_images = original_images[n_train:n_train + n_val]
    test_images = original_images[n_train + n_val:]

    # Training set = original train + ALL rotated
    train_images = train_originals + rotated_images

    console.print(f"\nDataset split:")
    console.print(f"  Train: {len(train_images)} ({len(train_originals)} orig + {len(rotated_images)} rotated)")
    console.print(f"  Val:   {len(val_images)} (originals only)")
    console.print(f"  Test:  {len(test_images)} (originals only)")

    # --- Create output directories ---
    for split in ["train", "val", "test"]:
        (out_path / split / "images").mkdir(parents=True, exist_ok=True)
        (out_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- Process each split ---
    stats = {
        "train": {"images": 0, "annotations": 0, "class_counts": {c: 0 for c in CLASS_TO_ID}},
        "val": {"images": 0, "annotations": 0, "class_counts": {c: 0 for c in CLASS_TO_ID}},
        "test": {"images": 0, "annotations": 0, "class_counts": {c: 0 for c in CLASS_TO_ID}},
    }

    splits_data = [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]

    for split_name, images in splits_data:
        console.print(f"\n[yellow]Processing {split_name} split...[/yellow]")
        split_dir = out_path / split_name
        missing_annotations = 0

        for img_path in track(images, description=f"  {split_name}..."):
            # Find annotation
            ann_path = find_annotation(img_path, data_path)

            if ann_path is None:
                missing_annotations += 1
                logger.warning(f"No annotation found for: {img_path.name}")
                # Skip images without annotations
                continue

            # Parse annotation
            img_w, img_h, annotations = parse_xml_annotation(ann_path)

            if img_w == 0 or not annotations:
                missing_annotations += 1
                continue

            # Copy image
            dest_image = split_dir / "images" / img_path.name
            shutil.copy2(img_path, dest_image)

            # Write YOLO label
            label_name = img_path.stem + ".txt"
            dest_label = split_dir / "labels" / label_name
            write_yolo_label(dest_label, annotations, img_w, img_h)

            # Update stats
            stats[split_name]["images"] += 1
            stats[split_name]["annotations"] += len(annotations)
            for ann in annotations:
                stats[split_name]["class_counts"][ann["class"]] += 1

        if missing_annotations > 0:
            logger.warning(f"{split_name}: {missing_annotations} images had no/invalid annotations")

    # --- Create dataset.yaml ---
    dataset_yaml = {
        "path": str(out_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_TO_ID),
        "names": list(CLASS_TO_ID.keys()),
    }

    yaml_path = out_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    # --- Save split metadata ---
    metadata = {
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "class_mapping": CLASS_TO_ID,
        "stats": stats,
    }
    with open(out_path / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Print summary ---
    console.rule("[bold green]Dataset Preparation Complete")
    console.print(f"\nDataset YAML: {yaml_path}")
    console.print(f"\nSplit Summary:")
    for split_name, split_stats in stats.items():
        console.print(f"  {split_name}: {split_stats['images']} images, "
                     f"{split_stats['annotations']} defects")

    logger.info("Dataset preparation complete!")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PCB Dataset for YOLO Training")
    parser.add_argument("--data-dir", default="data/raw/PCB_DATASET")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
