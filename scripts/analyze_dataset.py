"""
Dataset Analysis and Visualization Script
==========================================
PURPOSE:
    Generates comprehensive statistics and visualizations about the
    PCB defect dataset. Useful for understanding class imbalance,
    bounding box distributions, and dataset quality.

INPUT:
    data/splits/ (after running prepare_dataset.py)

OUTPUT:
    outputs/dataset_analysis/
    ├── class_distribution.png
    ├── bbox_size_distribution.png
    ├── aspect_ratio_distribution.png
    └── analysis_report.json

USAGE:
    python scripts/analyze_dataset.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper"
]

CLASS_COLORS = [
    "#E74C3C", "#E67E22", "#F1C40F",
    "#2ECC71", "#3498DB", "#9B59B6"
]


def load_yolo_labels(split_dir: Path) -> dict[str, Any]:
    """
    Load all YOLO label files from a split directory.

    Args:
        split_dir: Path to split directory (e.g., data/splits/train)

    Returns:
        Dict with class_counts, bbox_data, and per_image_defects.
    """
    labels_dir = split_dir / "labels"

    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        return {"class_counts": {}, "bboxes": [], "defects_per_image": []}

    class_counts = defaultdict(int)
    bboxes = []  # List of (class_id, cx, cy, w, h)
    defects_per_image = []

    for label_file in labels_dir.glob("*.txt"):
        defect_count = 0
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    continue

                if 0 <= class_id < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[class_id]] += 1
                    bboxes.append((class_id, cx, cy, w, h))
                    defect_count += 1

        defects_per_image.append(defect_count)

    return {
        "class_counts": dict(class_counts),
        "bboxes": bboxes,
        "defects_per_image": defects_per_image,
    }


def plot_class_distribution(stats: dict[str, dict], output_dir: Path) -> None:
    """
    Create a grouped bar chart showing class distribution across splits.

    Args:
        stats: Dict of {split_name: {class_counts: {...}}}
        output_dir: Where to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PCB Defect Class Distribution", fontsize=16, fontweight="bold")

    # --- Plot 1: Stacked bar chart per split ---
    splits = list(stats.keys())
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    ax1 = axes[0]
    for i, split in enumerate(splits):
        counts = [stats[split]["class_counts"].get(cls, 0) for cls in CLASS_NAMES]
        ax1.bar(x + i * width, counts, width, label=split, alpha=0.8)

    ax1.set_xlabel("Defect Class")
    ax1.set_ylabel("Count")
    ax1.set_title("Defect Count by Split")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(
        [c.replace("_", "\n") for c in CLASS_NAMES],
        fontsize=8
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- Plot 2: Pie chart for overall distribution ---
    ax2 = axes[1]
    all_counts = defaultdict(int)
    for split_data in stats.values():
        for cls, count in split_data["class_counts"].items():
            all_counts[cls] += count

    counts = [all_counts.get(cls, 0) for cls in CLASS_NAMES]
    non_zero = [(cls, cnt, col) for cls, cnt, col in zip(CLASS_NAMES, counts, CLASS_COLORS) if cnt > 0]

    if non_zero:
        labels, pie_counts, colors = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            pie_counts,
            labels=[l.replace("_", " ").title() for l in labels],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax2.set_title("Overall Class Distribution")

    plt.tight_layout()
    output_path = output_dir / "class_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_bbox_distributions(all_bboxes: list, output_dir: Path) -> None:
    """
    Plot bounding box size and aspect ratio distributions.

    Args:
        all_bboxes: List of (class_id, cx, cy, w, h) tuples.
        output_dir: Where to save plots.
    """
    if not all_bboxes:
        logger.warning("No bboxes to plot")
        return

    widths = [b[3] for b in all_bboxes]
    heights = [b[4] for b in all_bboxes]
    areas = [w * h for w, h in zip(widths, heights)]
    aspect_ratios = [w / max(h, 1e-6) for w, h in zip(widths, heights)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bounding Box Statistics", fontsize=14, fontweight="bold")

    # Width distribution
    axes[0, 0].hist(widths, bins=50, color="#3498DB", alpha=0.7, edgecolor="white")
    axes[0, 0].set_title("Normalized Width Distribution")
    axes[0, 0].set_xlabel("Width (0-1)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.3f}")
    axes[0, 0].legend()

    # Height distribution
    axes[0, 1].hist(heights, bins=50, color="#E74C3C", alpha=0.7, edgecolor="white")
    axes[0, 1].set_title("Normalized Height Distribution")
    axes[0, 1].set_xlabel("Height (0-1)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(np.mean(heights), color="blue", linestyle="--", label=f"Mean: {np.mean(heights):.3f}")
    axes[0, 1].legend()

    # Area distribution
    axes[1, 0].hist(areas, bins=50, color="#2ECC71", alpha=0.7, edgecolor="white")
    axes[1, 0].set_title("Bounding Box Area Distribution")
    axes[1, 0].set_xlabel("Normalized Area (w×h)")
    axes[1, 0].set_ylabel("Frequency")

    # Scatter: Width vs Height colored by class
    scatter_colors = [CLASS_COLORS[b[0]] for b in all_bboxes]
    axes[1, 1].scatter(widths, heights, c=scatter_colors, alpha=0.3, s=5)
    axes[1, 1].set_title("Width vs Height by Class")
    axes[1, 1].set_xlabel("Normalized Width")
    axes[1, 1].set_ylabel("Normalized Height")
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    axes[1, 1].legend(handles=patches, fontsize=7, loc="upper right")

    plt.tight_layout()
    output_path = output_dir / "bbox_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_defects_per_image(stats: dict, output_dir: Path) -> None:
    """
    Plot histogram of defects per image across splits.

    Args:
        stats: Dict with split data including defects_per_image.
        output_dir: Output directory.
    """
    fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4))
    if len(stats) == 1:
        axes = [axes]

    fig.suptitle("Defects Per Image Distribution", fontsize=14, fontweight="bold")

    for ax, (split_name, split_data) in zip(axes, stats.items()):
        dpi = split_data.get("defects_per_image", [])
        if dpi:
            ax.hist(dpi, bins=range(0, max(dpi) + 2), color="#9B59B6", alpha=0.7, edgecolor="white")
            ax.set_title(f"{split_name.title()} Split")
            ax.set_xlabel("Defects per Image")
            ax.set_ylabel("Image Count")
            ax.axvline(np.mean(dpi), color="red", linestyle="--",
                      label=f"Mean: {np.mean(dpi):.1f}")
            ax.legend()

    plt.tight_layout()
    output_path = output_dir / "defects_per_image.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def run_analysis(splits_dir: str = "data/splits") -> None:
    """
    Run complete dataset analysis.

    Args:
        splits_dir: Path to processed dataset splits.
    """
    splits_path = Path(splits_dir)
    output_dir = Path("outputs/dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]PCB Dataset Analysis")

    if not splits_path.exists():
        console.print(f"[red]Splits directory not found: {splits_path}[/red]")
        console.print("Run: python scripts/prepare_dataset.py first")
        sys.exit(1)

    # Load stats for all splits
    all_stats = {}
    all_bboxes = []

    for split_name in ["train", "val", "test"]:
        split_dir = splits_path / split_name
        if not split_dir.exists():
            continue

        split_data = load_yolo_labels(split_dir)
        all_stats[split_name] = split_data
        all_bboxes.extend(split_data["bboxes"])

    if not all_stats:
        console.print("[red]No split data found. Run prepare_dataset.py first.[/red]")
        sys.exit(1)

    # Print summary table
    table = Table(title="Dataset Split Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Images", justify="right", style="green")
    table.add_column("Defects", justify="right", style="yellow")
    table.add_column("Avg/Image", justify="right")

    for split_name, split_data in all_stats.items():
        n_images = len(split_data["defects_per_image"])
        n_defects = sum(split_data["class_counts"].values())
        avg = n_defects / max(n_images, 1)
        table.add_row(split_name, str(n_images), str(n_defects), f"{avg:.1f}")

    console.print(table)

    # Generate visualizations
    console.print("\n[yellow]Generating visualizations...[/yellow]")
    plot_class_distribution(all_stats, output_dir)
    plot_bbox_distributions(all_bboxes, output_dir)
    plot_defects_per_image(all_stats, output_dir)

    # Save analysis report
    report = {
        "splits": {
            split_name: {
                "n_images": len(data["defects_per_image"]),
                "n_defects": sum(data["class_counts"].values()),
                "class_counts": data["class_counts"],
            }
            for split_name, data in all_stats.items()
        }
    }
    with open(output_dir / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[bold green]Analysis complete! Outputs in: {output_dir}[/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()
    run_analysis(args.splits_dir)
