"""
Detector Inference Script
==========================
PURPOSE:
    Standalone inference script for running the trained PCB defect
    detector on individual images or directories of images.

USAGE:
    python detector/inference.py --image path/to/pcb.jpg
    python detector/inference.py --dir path/to/images/ --save
    python detector/inference.py --image pcb.jpg --conf 0.4 --show
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

from detector.detector import PCBDefectDetector, Detection

console = Console()


def run_inference(
    source: Union[str, Path],
    weights: str = "models/detector/best.pt",
    config: str = "configs/inference.yaml",
    conf: Optional[float] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> list[list[Detection]]:
    """
    Run inference on an image or directory of images.

    Args:
        source: Image path or directory path.
        weights: Model weights path.
        config: Inference config path.
        conf: Override confidence threshold.
        save_dir: Directory to save annotated images.
        show: Display results using OpenCV window.

    Returns:
        List of detection lists (one per image).
    """
    source_path = Path(source)
    
    # --- Collect input images ---
    if source_path.is_file():
        images = [source_path]
    elif source_path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [p for p in source_path.iterdir() if p.suffix.lower() in image_exts]
        images.sort()
    else:
        raise ValueError(f"Source not found: {source_path}")

    if not images:
        logger.warning(f"No images found at: {source_path}")
        return []

    logger.info(f"Running inference on {len(images)} image(s)")

    # --- Initialize detector ---
    detector = PCBDefectDetector(weights_path=weights, config_path=config)
    
    # Override confidence if provided
    if conf is not None:
        detector.conf_threshold = conf
        logger.info(f"Confidence threshold overridden to: {conf}")

    # --- Setup output directory ---
    if save_dir:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run inference ---
    all_detections = []
    all_results_json = []

    for img_path in images:
        logger.info(f"Processing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Could not read image: {img_path}")
            all_detections.append([])
            continue

        # Detect defects
        detections = detector.detect(image, return_crops=True)
        all_detections.append(detections)

        # --- Print results ---
        _print_detection_results(img_path.name, detections)

        # --- Save annotated image ---
        if save_dir or show:
            annotated = detector.draw_detections(image, detections)
            
            if save_dir:
                save_path = output_dir / f"detected_{img_path.name}"
                cv2.imwrite(str(save_path), annotated)
                logger.info(f"Saved: {save_path}")
            
            if show:
                cv2.imshow(f"PCB Defect Detection - {img_path.name}", annotated)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
        
        # Collect JSON results
        all_results_json.append({
            "image": img_path.name,
            "detections": [d.to_dict() for d in detections],
        })

    if show:
        cv2.destroyAllWindows()

    # --- Save JSON results ---
    if save_dir:
        json_path = output_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(all_results_json, f, indent=2)
        logger.info(f"JSON results saved to: {json_path}")

    return all_detections


def _print_detection_results(image_name: str, detections: list[Detection]) -> None:
    """Print detection results in a formatted table."""
    if not detections:
        console.print(f"[yellow]{image_name}: No defects detected[/yellow]")
        return

    table = Table(title=f"Detections: {image_name}")
    table.add_column("Class", style="cyan")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Bounding Box", style="dim")
    table.add_column("Area (px²)", justify="right")

    for det in detections:
        conf_str = f"{det.confidence:.1%}"
        bbox_str = f"[{int(det.bbox[0])},{int(det.bbox[1])},{int(det.bbox[2])},{int(det.bbox[3])}]"
        area_str = f"{det.area:,.0f}"
        
        # Color code by confidence
        conf_style = "green" if det.confidence >= 0.7 else "yellow" if det.confidence >= 0.5 else "red"
        table.add_row(
            det.class_name.replace("_", " ").title(),
            f"[{conf_style}]{conf_str}[/{conf_style}]",
            bbox_str,
            area_str,
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Defect Detector Inference")
    parser.add_argument("--image", help="Path to single PCB image")
    parser.add_argument("--dir", help="Directory of PCB images")
    parser.add_argument("--weights", default="models/detector/best.pt")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--save", default="outputs/detections", help="Save directory")
    parser.add_argument("--show", action="store_true", help="Show results in window")
    args = parser.parse_args()

    source = args.image or args.dir
    if not source:
        parser.error("Provide --image or --dir")

    run_inference(
        source=source,
        weights=args.weights,
        config=args.config,
        conf=args.conf,
        save_dir=args.save,
        show=args.show,
    )
