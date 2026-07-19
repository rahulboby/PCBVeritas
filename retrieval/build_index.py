"""
FAISS Index Builder
====================
PURPOSE:
    Processes all training dataset defect crops, generates SigLIP embeddings
    for each, and builds a FAISS vector index for fast similarity search.

    This is a ONE-TIME setup step run after dataset preparation and before
    inference. The resulting index is loaded at runtime for retrieval.

INPUT:
    data/splits/train/ (processed training images with YOLO labels)

OUTPUT:
    data/embeddings/faiss_index.bin  (FAISS index file)
    data/embeddings/metadata.json    (labels and paths for each vector)
    data/processed/crops/            (extracted defect crop images)

HOW IT WORKS:
    1. Load all training images and their YOLO annotations
    2. For each bounding box annotation, extract a crop from the image
    3. Save the crop to data/processed/crops/
    4. Generate SigLIP embedding for each crop
    5. Add all embeddings to a FAISS index
    6. Save the index and metadata to disk

USAGE:
    python retrieval/build_index.py
    python retrieval/build_index.py --splits-dir data/splits --force-rebuild
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import track

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.embedder import SigLIPEmbedder

console = Console()

# Defect class names (must match training order)
CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]


def extract_crops_from_split(
    split_dir: Path,
    crops_dir: Path,
    padding: int = 20,
    min_size: int = 32,
) -> list[dict]:
    """
    Extract defect crop images from a dataset split.

    Reads YOLO label files (.txt), parses bounding boxes, and crops
    the corresponding regions from the original images.

    Args:
        split_dir: Path to split directory (e.g., data/splits/train).
        crops_dir: Directory to save extracted crops.
        padding: Pixels to pad around each bbox.
        min_size: Minimum crop dimension (smaller crops are skipped).

    Returns:
        List of metadata dicts for each extracted crop.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logger.warning(f"Missing images or labels directory in: {split_dir}")
        return []

    crops_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for img_path in track(image_paths, description=f"Extracting crops from {split_dir.name}..."):
        # Find corresponding label file
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        # Parse YOLO annotations
        with open(label_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for ann_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            if class_id >= len(CLASS_NAMES):
                continue

            class_name = CLASS_NAMES[class_id]

            # Convert YOLO normalized coords to pixel coords
            # YOLO: center_x, center_y, width, height (normalized)
            x1 = int((cx - bw / 2) * img_w) - padding
            y1 = int((cy - bh / 2) * img_h) - padding
            x2 = int((cx + bw / 2) * img_w) + padding
            y2 = int((cy + bh / 2) * img_h) + padding

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Skip tiny crops
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < min_size or crop_h < min_size:
                logger.debug(f"Skipping tiny crop ({crop_w}x{crop_h}) in {img_path.name}")
                continue

            # Extract crop
            crop = image[y1:y2, x1:x2]

            # Save crop
            crop_name = f"{class_name}_{img_path.stem}_{ann_idx:03d}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            metadata.append({
                "label": class_name,
                "class_id": class_id,
                "crop_path": str(crop_path),
                "source_image": str(img_path),
                "bbox": [x1, y1, x2, y2],
                "bbox_normalized": [cx, cy, bw, bh],
            })

    logger.info(f"Extracted {len(metadata)} crops from {split_dir.name}")
    return metadata


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "IndexFlatIP",
) -> "faiss.Index":
    """
    Build a FAISS index from embedding vectors.

    IndexFlatIP:
        - Exact brute-force inner product search
        - Best accuracy, moderate speed
        - Suitable for datasets up to ~100k vectors
        - "IP" = Inner Product = cosine similarity (for L2-normalized vectors)

    IndexIVFFlat (for larger datasets):
        - Inverted file index with flat quantizer
        - Approximate search (faster but slight accuracy loss)
        - Requires training step before adding vectors

    Args:
        embeddings: Embedding matrix, shape (N, D).
        index_type: FAISS index type string.

    Returns:
        Populated FAISS index ready for search.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("Install FAISS: pip install faiss-cpu")

    n, d = embeddings.shape
    logger.info(f"Building FAISS {index_type} index | vectors={n} | dim={d}")

    embeddings_f32 = embeddings.astype(np.float32)

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(d)

    elif index_type == "IndexIVFFlat":
        # IVF requires a coarse quantizer and training
        n_clusters = min(int(np.sqrt(n)), 256)  # Rule of thumb: sqrt(N) clusters
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
        logger.info(f"Training IVF index with {n_clusters} clusters...")
        index.train(embeddings_f32)

    else:
        logger.warning(f"Unknown index type '{index_type}'. Using IndexFlatIP.")
        index = faiss.IndexFlatIP(d)

    # Add all vectors to the index
    index.add(embeddings_f32)

    logger.info(f"FAISS index built | total_vectors={index.ntotal}")
    return index


def run_build_index(
    splits_dir: str = "data/splits",
    crops_dir: str = "data/processed/crops",
    output_dir: str = "data/embeddings",
    config: Optional[dict] = None,
    splits: list[str] = None,
    force_rebuild: bool = False,
) -> None:
    """
    Main function to build the FAISS index.

    Args:
        splits_dir: Base directory for dataset splits.
        crops_dir: Where to save/find extracted crops.
        output_dir: Where to save FAISS index and metadata.
        config: Retrieval configuration dictionary.
        splits: Which splits to include (default: ['train']).
        force_rebuild: Rebuild even if crops already exist.
    """
    if splits is None:
        splits = ["train"]

    splits_path = Path(splits_dir)
    crops_path = Path(crops_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    if config is None:
        from configs.settings import RETRIEVAL_CONFIG
        config = RETRIEVAL_CONFIG

    padding = config.get("crops", {}).get("padding", 20)
    min_size = config.get("crops", {}).get("min_size", 32)
    index_path = output_path / Path(config["faiss"]["index_path"]).name
    metadata_path = output_path / Path(config["faiss"]["metadata_path"]).name

    console.rule("[bold blue]Building FAISS Retrieval Index")

    # --- Step 1: Extract crops ---
    all_metadata = []

    if crops_path.exists() and any(crops_path.iterdir()) and not force_rebuild:
        console.print(f"[green]Crops directory exists: {crops_path}[/green]")
        console.print("Loading existing crops. Use --force-rebuild to re-extract.")

        # Load existing metadata if available
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                all_metadata = json.load(f)
            console.print(f"Loaded {len(all_metadata)} existing metadata records")
        else:
            # Rebuild metadata from crop filenames
            for crop_file in sorted(crops_path.glob("*.jpg")):
                parts = crop_file.stem.rsplit("_", 2)
                label = "_".join(parts[:-2]) if len(parts) >= 3 else "unknown"
                all_metadata.append({
                    "label": label,
                    "crop_path": str(crop_file),
                    "source_image": "",
                })
    else:
        # Extract crops from splits
        for split_name in splits:
            split_dir = splits_path / split_name
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            split_metadata = extract_crops_from_split(
                split_dir=split_dir,
                crops_dir=crops_path,
                padding=padding,
                min_size=min_size,
            )
            all_metadata.extend(split_metadata)

    if not all_metadata:
        logger.error("No crop metadata found. Cannot build index.")
        return

    console.print(f"\n[yellow]Total crops to embed: {len(all_metadata)}[/yellow]")

    # --- Step 2: Generate SigLIP embeddings ---
    console.print("\n[yellow]Loading SigLIP embedder...[/yellow]")
    embedder = SigLIPEmbedder(config=config)

    console.print("[yellow]Generating embeddings...[/yellow]")

    # Load crop images in batches
    batch_size = config.get("embedding", {}).get("batch_size", 32)
    all_embeddings = []
    valid_metadata = []

    crop_paths = [meta["crop_path"] for meta in all_metadata]

    for i in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[i:i + batch_size]
        batch_images = []
        batch_meta = []

        for path, meta in zip(batch_paths, all_metadata[i:i + batch_size]):
            img = cv2.imread(path)
            if img is not None:
                batch_images.append(img)
                batch_meta.append(meta)
            else:
                logger.warning(f"Could not read crop: {path}")

        if not batch_images:
            continue

        embeddings_batch = embedder.embed_batch(batch_images)
        all_embeddings.append(embeddings_batch)
        valid_metadata.extend(batch_meta)

        progress = min(i + batch_size, len(crop_paths))
        console.print(f"  Embedded {progress}/{len(crop_paths)} crops...", end="\r")

    console.print()

    if not all_embeddings:
        logger.error("No embeddings generated. Check crop images.")
        return

    embeddings_matrix = np.vstack(all_embeddings)
    console.print(f"\n[green]Embeddings shape: {embeddings_matrix.shape}[/green]")

    # --- Step 3: Build FAISS index ---
    index_type = config.get("faiss", {}).get("index_type", "IndexFlatIP")
    faiss_index = build_faiss_index(embeddings_matrix, index_type=index_type)

    # --- Step 4: Save index and metadata ---
    faiss.write_index(faiss_index, str(index_path))
    console.print(f"[green]FAISS index saved: {index_path}[/green]")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(valid_metadata, f, indent=2)
    console.print(f"[green]Metadata saved: {metadata_path}[/green]")

    # --- Summary ---
    console.rule("[bold green]Index Build Complete")
    console.print(f"  Total vectors: {faiss_index.ntotal}")
    console.print(f"  Embedding dim: {faiss_index.d}")
    console.print(f"  Index path:    {index_path}")
    console.print(f"  Metadata path: {metadata_path}")

    # Class distribution
    class_counts: dict[str, int] = {}
    for meta in valid_metadata:
        lbl = meta.get("label", "unknown")
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    console.print("\nClass distribution in index:")
    for cls, cnt in sorted(class_counts.items()):
        console.print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--crops-dir", default="data/processed/crops")
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    from configs.settings import RETRIEVAL_CONFIG
    
    run_build_index(
        splits_dir=args.splits_dir,
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        config=RETRIEVAL_CONFIG,
        splits=args.splits,
        force_rebuild=args.force_rebuild,
    )
