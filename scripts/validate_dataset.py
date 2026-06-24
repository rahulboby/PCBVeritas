"""
Dataset Validation Script
==========================
PURPOSE:
    Validates the downloaded PCB Defect Dataset structure, file integrity,
    and annotation quality before any training begins.

INPUT:
    PCB Defect Dataset at data/raw/PCB_DATASET/

OUTPUT:
    Validation report printed to console and saved to logs/dataset_validation.json

HOW IT WORKS:
    1. Checks directory structure exists
    2. Counts images and annotation files
    3. Validates XML annotation format (Pascal VOC style)
    4. Checks for class balance
    5. Detects corrupted images
    6. Reports any issues found

USAGE:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data-dir path/to/dataset
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import argparse

import cv2
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

console = Console()

# PCB Dataset class names (from dataset description)
VALID_CLASSES = {
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
}

# Also accept these alternate formats
CLASS_ALIASES = {
    "Missing_hole": "missing_hole",
    "Mouse_bite": "mouse_bite",
    "Open_circuit": "open_circuit",
    "Short": "short",
    "Spur": "spur",
    "Spurious_copper": "spurious_copper",
}


def validate_xml_annotation(xml_path: Path) -> dict[str, Any]:
    """
    Parse and validate a Pascal VOC XML annotation file.

    Pascal VOC XML format looks like:
    <annotation>
        <filename>image.jpg</filename>
        <size>
            <width>640</width>
            <height>640</height>
        </size>
        <object>
            <name>missing_hole</name>
            <bndbox>
                <xmin>10</xmin>
                <ymin>20</ymin>
                <xmax>50</xmax>
                <ymax>80</ymax>
            </bndbox>
        </object>
    </annotation>

    Args:
        xml_path: Path to XML file.

    Returns:
        Dict with 'valid' bool, 'objects' list, 'errors' list.
    """
    result: dict[str, Any] = {
        "valid": True,
        "filename": str(xml_path.name),
        "objects": [],
        "errors": [],
        "width": 0,
        "height": 0,
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size_elem = root.find("size")
        if size_elem is not None:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            if w_elem is not None:
                result["width"] = int(w_elem.text or 0)
            if h_elem is not None:
                result["height"] = int(h_elem.text or 0)

        # Validate each object annotation
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                result["errors"].append("Object missing 'name' element")
                result["valid"] = False
                continue

            class_name = name_elem.text.strip()
            # Normalize class name
            class_name = CLASS_ALIASES.get(class_name, class_name.lower().replace(" ", "_"))

            if class_name not in VALID_CLASSES:
                result["errors"].append(f"Unknown class: '{class_name}'")
                # Don't mark as invalid - just warn

            bbox = obj.find("bndbox")
            if bbox is None:
                result["errors"].append(f"Object '{class_name}' missing bndbox")
                result["valid"] = False
                continue

            try:
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
            except (AttributeError, TypeError, ValueError) as e:
                result["errors"].append(f"Invalid bbox coordinates: {e}")
                result["valid"] = False
                continue

            # Validate bbox sanity
            if xmin >= xmax or ymin >= ymax:
                result["errors"].append(
                    f"Invalid bbox for '{class_name}': "
                    f"[{xmin},{ymin},{xmax},{ymax}] - min >= max"
                )
                result["valid"] = False

            if xmin < 0 or ymin < 0:
                result["errors"].append(
                    f"Negative bbox coordinates for '{class_name}'"
                )

            result["objects"].append({
                "class": class_name,
                "bbox": [xmin, ymin, xmax, ymax],
            })

    except ET.ParseError as e:
        result["valid"] = False
        result["errors"].append(f"XML parse error: {e}")

    return result


def validate_image(image_path: Path) -> dict[str, Any]:
    """
    Validate that an image can be loaded and has expected properties.

    Args:
        image_path: Path to image file.

    Returns:
        Dict with 'valid' bool, 'width', 'height', 'channels', 'errors'.
    """
    result: dict[str, Any] = {
        "valid": True,
        "path": str(image_path),
        "width": 0,
        "height": 0,
        "channels": 0,
        "errors": [],
    }

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            result["valid"] = False
            result["errors"].append("Could not read image (corrupted or unsupported format)")
            return result

        result["height"], result["width"] = img.shape[:2]
        result["channels"] = img.shape[2] if len(img.shape) == 3 else 1

        if result["width"] == 0 or result["height"] == 0:
            result["valid"] = False
            result["errors"].append("Image has zero dimensions")

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Error loading image: {e}")

    return result


def run_validation(data_dir: str) -> dict[str, Any]:
    """
    Run complete dataset validation.

    Args:
        data_dir: Root directory of PCB dataset.

    Returns:
        Comprehensive validation report dictionary.
    """
    data_path = Path(data_dir)
    
    console.rule("[bold blue]PCB Dataset Validation")
    console.print(f"Dataset directory: {data_path.resolve()}")

    report: dict[str, Any] = {
        "data_dir": str(data_path),
        "status": "unknown",
        "total_images": 0,
        "total_annotations": 0,
        "valid_images": 0,
        "invalid_images": [],
        "valid_annotations": 0,
        "invalid_annotations": [],
        "class_distribution": {cls: 0 for cls in VALID_CLASSES},
        "unknown_classes": [],
        "total_defects": 0,
        "errors": [],
        "warnings": [],
    }

    # --- Check directory exists ---
    if not data_path.exists():
        report["status"] = "FAILED"
        report["errors"].append(f"Dataset directory not found: {data_path}")
        console.print(f"[red]ERROR: Dataset directory not found: {data_path}[/red]")
        console.print("\n[yellow]Please download the PCB Defect Dataset and place it at:[/yellow]")
        console.print(f"  {data_path.resolve()}")
        console.print("\n[yellow]Dataset URL: https://robotics.pkusz.edu.cn/resources/dataset/[/yellow]")
        return report

    # --- Find all images ---
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []
    for ext in image_extensions:
        all_images.extend(data_path.rglob(f"*{ext}"))
        all_images.extend(data_path.rglob(f"*{ext.upper()}"))

    # Filter out augmented/rotated images for initial check
    original_images = [p for p in all_images if "rotation" not in p.stem.lower()]
    rotated_images = [p for p in all_images if "rotation" in p.stem.lower()]

    report["total_images"] = len(all_images)
    report["original_images"] = len(original_images)
    report["rotated_images"] = len(rotated_images)

    console.print(f"\n[green]Found {len(all_images)} total images:[/green]")
    console.print(f"  - Original: {len(original_images)}")
    console.print(f"  - Rotated:  {len(rotated_images)}")

    if len(all_images) == 0:
        report["status"] = "FAILED"
        report["errors"].append("No images found in dataset directory")
        return report

    # --- Validate images ---
    console.print("\n[yellow]Validating images...[/yellow]")
    image_errors = []
    for img_path in track(all_images[:100], description="Checking images..."):
        img_result = validate_image(img_path)
        if not img_result["valid"]:
            image_errors.append({
                "file": str(img_path),
                "errors": img_result["errors"],
            })
        else:
            report["valid_images"] += 1

    report["invalid_images"] = image_errors
    if image_errors:
        console.print(f"[red]Found {len(image_errors)} invalid images[/red]")
        for err in image_errors[:5]:
            console.print(f"  - {err['file']}: {err['errors']}")

    # --- Find and validate annotations ---
    console.print("\n[yellow]Validating annotations...[/yellow]")
    all_annotations = list(data_path.rglob("*.xml"))
    report["total_annotations"] = len(all_annotations)

    console.print(f"Found {len(all_annotations)} annotation files")

    annotation_errors = []
    class_counts: dict[str, int] = {cls: 0 for cls in VALID_CLASSES}
    unknown_classes: list[str] = []

    for ann_path in track(all_annotations, description="Checking annotations..."):
        ann_result = validate_xml_annotation(ann_path)
        
        if ann_result["valid"]:
            report["valid_annotations"] += 1
        else:
            annotation_errors.append({
                "file": str(ann_path),
                "errors": ann_result["errors"],
            })

        # Count classes
        for obj in ann_result["objects"]:
            cls = obj["class"]
            if cls in class_counts:
                class_counts[cls] += 1
                report["total_defects"] += 1
            else:
                unknown_classes.append(cls)

    report["invalid_annotations"] = annotation_errors
    report["class_distribution"] = class_counts
    report["unknown_classes"] = list(set(unknown_classes))

    # --- Print class distribution table ---
    table = Table(title="Defect Class Distribution")
    table.add_column("Class", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    total_defects = sum(class_counts.values())
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_defects * 100) if total_defects > 0 else 0
        table.add_row(cls, str(count), f"{pct:.1f}%")
    
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_defects}[/bold]", "100%")
    console.print(table)

    # --- Warnings for class imbalance ---
    if total_defects > 0:
        max_count = max(class_counts.values()) if class_counts else 0
        min_count = min(v for v in class_counts.values() if v > 0) if class_counts else 0
        imbalance_ratio = max_count / max(min_count, 1)
        
        if imbalance_ratio > 10:
            report["warnings"].append(
                f"High class imbalance detected: ratio = {imbalance_ratio:.1f}x. "
                "Consider oversampling minority classes."
            )
            console.print(f"[yellow]WARNING: Class imbalance ratio = {imbalance_ratio:.1f}x[/yellow]")

    # --- Determine overall status ---
    critical_errors = len(report["errors"]) + len(image_errors) + len(annotation_errors)
    if critical_errors == 0 and total_defects > 0:
        report["status"] = "PASSED"
        console.print("\n[bold green]✓ Dataset validation PASSED[/bold green]")
    elif total_defects > 0:
        report["status"] = "PASSED_WITH_WARNINGS"
        console.print(f"\n[bold yellow]⚠ Dataset validation passed with {critical_errors} issues[/bold yellow]")
    else:
        report["status"] = "FAILED"
        console.print("\n[bold red]✗ Dataset validation FAILED[/bold red]")

    # --- Save report ---
    report_path = Path("logs/dataset_validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    console.print(f"\nValidation report saved to: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate PCB Defect Dataset")
    parser.add_argument(
        "--data-dir",
        default="data/raw/PCB_DATASET",
        help="Path to PCB dataset root directory",
    )
    args = parser.parse_args()

    report = run_validation(args.data_dir)
    
    if report["status"] == "FAILED":
        sys.exit(1)
