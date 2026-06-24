"""
SigLIP Visual Embedder for PCB Defect Crops
=============================================
PURPOSE:
    Generates dense vector embeddings for PCB defect crop images using
    SigLIP (Sigmoid Loss for Language Image Pre-Training). These embeddings
    capture the visual semantics of each defect and are stored in FAISS
    for fast similarity search.

WHAT IS SIGLIP?
    SigLIP is a vision-language model from Google that learns aligned
    representations of images and text. We use only its image encoder here.
    Unlike CLIP, SigLIP uses sigmoid loss (not softmax), which makes it
    better for fine-grained visual similarity.

    SigLIP was trained on billions of image-text pairs, so it understands:
    - Visual textures and patterns
    - Object shapes and edges
    - Color distributions
    - Spatial arrangements

WHY USE SIGLIP FOR PCB DEFECTS?
    PCB defects are subtle visual patterns. SigLIP's rich visual
    representations capture these nuances better than raw pixels.
    Similar defects will have high cosine similarity in embedding space,
    enabling accurate retrieval.

INPUT:
    - Defect crop image (numpy array, BGR)

OUTPUT:
    - Embedding vector (float32, shape [768])
    - Normalized to unit length (enables cosine similarity via dot product)

HOW IT WORKS:
    1. Resize crop to 224x224 (SigLIP input size)
    2. Normalize with SigLIP's mean/std values
    3. Pass through SigLIP vision encoder (ViT-B/16 backbone)
    4. Extract [CLS] token representation
    5. L2-normalize the embedding vector

CONNECTS TO:
    - retrieval/build_index.py: Builds FAISS index from all training crops
    - retrieval/faiss_search.py: Queries index with new defect embeddings
    - pipeline/orchestrator.py: Called for each detected defect
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import yaml
from loguru import logger


def _load_runtime_deps():
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    return cv2, torch, Image, AutoModel, AutoProcessor


class SigLIPEmbedder:
    """
    Generates visual embeddings for PCB defect crops using SigLIP.

    Uses the HuggingFace transformers library to load SigLIP locally —
    no internet connection required after initial download.

    Example:
        embedder = SigLIPEmbedder()
        crop = cv2.imread("defect_crop.jpg")
        embedding = embedder.embed(crop)   # shape: (768,)
        print(f"Embedding shape: {embedding.shape}")
    """

    SIGLIP_MODEL = "google/siglip-base-patch16-224"
    EMBEDDING_DIM = 768

    def __init__(
        self,
        config_path: str = "configs/retrieval.yaml",
        model_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize SigLIP embedder.

        Args:
            config_path: Path to retrieval configuration.
            model_dir: Local directory to cache the model. If None, uses config.
        """
        self.config = self._load_config(config_path)
        (
            self.cv2,
            self.torch,
            self.Image,
            self.AutoModel,
            self.AutoProcessor,
        ) = _load_runtime_deps()

        # Model directory (for offline use)
        self.model_dir = model_dir or self.config.get("embedding", {}).get(
            "model_dir", "models/embeddings/siglip"
        )
        self.model_name = self.config.get("embedding", {}).get(
            "model_name", self.SIGLIP_MODEL
        )
        self.image_size = self.config.get("embedding", {}).get("image_size", 224)
        self.batch_size = self.config.get("embedding", {}).get("batch_size", 32)
        self.normalize = self.config.get("embedding", {}).get("normalize", True)

        # Device selection
        device_str = self.config.get("embedding", {}).get("device", "cuda")
        if device_str == "cuda" and self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
            logger.info(f"SigLIP using GPU: {self.torch.cuda.get_device_name(0)}")
        else:
            self.device = self.torch.device("cpu")
            logger.info("SigLIP using CPU")

        # Load model
        self.model, self.processor = self._load_model()

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}. Using defaults.")
            return {}

    def _load_model(self) -> tuple:
        """
        Load SigLIP model and processor.

        Tries local directory first (for offline use), then HuggingFace hub.

        Returns:
            Tuple of (model, processor).
        """
        local_path = Path(self.model_dir)

        # Check if model is cached locally
        if local_path.exists() and any(local_path.iterdir()):
            load_from = str(local_path)
            logger.info(f"Loading SigLIP from local cache: {local_path}")
        else:
            load_from = self.model_name
            logger.info(f"Downloading SigLIP from HuggingFace: {self.model_name}")
            logger.info("This will be cached locally for future offline use.")

        try:
            processor = self.AutoProcessor.from_pretrained(
                load_from,
                cache_dir=str(local_path),
            )
            model = self.AutoModel.from_pretrained(
                load_from,
                cache_dir=str(local_path),
                torch_dtype=(
                    self.torch.float16
                    if self.device.type == "cuda"
                    else self.torch.float32
                ),
            )
            model = model.to(self.device)
            model.eval()

            # Save locally for offline use
            if load_from != str(local_path):
                local_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(local_path))
                processor.save_pretrained(str(local_path))
                logger.info(f"SigLIP saved locally to: {local_path}")

            logger.info("SigLIP model loaded successfully")
            return model, processor

        except Exception as e:
            logger.error(f"Failed to load SigLIP: {e}")
            raise

    def embed(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single defect crop image.

        Args:
            image: BGR numpy array (from cv2) or PIL Image.

        Returns:
            Embedding vector, shape (768,), float32, L2-normalized if configured.
        """
        # Convert numpy BGR -> PIL RGB
        if isinstance(image, np.ndarray):
            image_rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            pil_image = self.Image.fromarray(image_rgb)
        else:
            pil_image = image

        return self.embed_batch([pil_image])[0]

    def embed_batch(
        self, images: list[Union[np.ndarray, Image.Image]]
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Batching is more efficient than calling embed() in a loop
        because the GPU processes multiple images simultaneously.

        Args:
            images: List of BGR numpy arrays or PIL Images.

        Returns:
            Embedding matrix, shape (N, 768), float32.
        """
        if not images:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)

        # Convert all to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
                pil_images.append(self.Image.fromarray(rgb))
            else:
                pil_images.append(img)

        # Process in mini-batches to avoid OOM
        all_embeddings = []

        for i in range(0, len(pil_images), self.batch_size):
            batch = pil_images[i:i + self.batch_size]

            # SigLIP processor handles resizing and normalization
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                # SigLIP returns image_embeds from the vision encoder
                outputs = self.model.get_image_features(**inputs)

            # outputs shape: [batch_size, embedding_dim]
            embeddings = outputs.cpu().float().numpy()

            # L2 normalize for cosine similarity via dot product
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
                embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def embed_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk and generate its embedding.

        Args:
            image_path: Path to image file.

        Returns:
            Embedding vector, shape (768,).
        """
        img = self.cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.embed(img)

    def embed_dataset_crops(
        self,
        crops_dir: str,
        metadata: Optional[list[dict]] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Generate embeddings for all crop images in a directory.

        Used during index building to embed the entire training dataset.

        Args:
            crops_dir: Directory containing defect crop images.
            metadata: Optional pre-built metadata list. If None, builds from filenames.

        Returns:
            Tuple of:
            - embeddings: Array of shape (N, 768)
            - metadata: List of dicts with 'path', 'label', 'image_path'
        """
        crops_path = Path(crops_dir)
        if not crops_path.exists():
            raise FileNotFoundError(f"Crops directory not found: {crops_path}")

        # Collect all crop images
        image_paths = sorted(crops_path.rglob("*.jpg")) + \
                     sorted(crops_path.rglob("*.png"))

        if not image_paths:
            raise ValueError(f"No images found in: {crops_path}")

        logger.info(f"Embedding {len(image_paths)} crops...")

        # Build metadata from filenames if not provided
        if metadata is None:
            metadata = []
            for p in image_paths:
                # Filename format: class_name_imagename_idx.jpg
                metadata.append({
                    "path": str(p),
                    "label": _extract_class_from_filename(p.name),
                    "filename": p.name,
                })

        # Load and embed in batches
        images = []
        valid_metadata = []

        for path_item, meta in zip(image_paths, metadata):
            img = self.cv2.imread(str(path_item))
            if img is not None:
                images.append(img)
                valid_metadata.append(meta)
            else:
                logger.warning(f"Could not read crop: {path_item}")

        if not images:
            raise ValueError("No valid images could be loaded")

        embeddings = self.embed_batch(images)
        logger.info(f"Generated {len(embeddings)} embeddings, shape: {embeddings.shape}")

        return embeddings, valid_metadata


def _extract_class_from_filename(filename: str) -> str:
    """
    Extract PCB defect class name from crop filename.

    Crop filenames follow the pattern:
    missing_hole_PCB_00001_1.jpg  -> missing_hole
    short_PCB_00042_2.jpg         -> short

    Args:
        filename: Crop image filename.

    Returns:
        Class name string.
    """
    known_classes = [
        "spurious_copper", "missing_hole", "mouse_bite",
        "open_circuit", "short", "spur",
    ]
    filename_lower = filename.lower()
    for cls in known_classes:
        if filename_lower.startswith(cls):
            return cls
    # Fallback: use first part before digit
    stem = Path(filename).stem
    for cls in known_classes:
        if cls in stem:
            return cls
    return "unknown"
