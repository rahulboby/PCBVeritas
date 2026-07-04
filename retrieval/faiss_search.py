"""
FAISS Vector Search for Similar Defect Retrieval
=================================================
PURPOSE:
    Uses FAISS (Facebook AI Similarity Search) to find the most visually
    similar historical defects for any newly detected defect.

    When a new defect is detected, we embed it with SigLIP and search
    the FAISS index to find the top-K most similar historical cases.
    This helps engineers understand: "Has this pattern been seen before?"

WHAT IS FAISS?
    FAISS is a library for efficient similarity search in high-dimensional
    vector spaces. It can search millions of 768-dimensional vectors in
    milliseconds by organizing them into specialized data structures.

    We use IndexFlatIP (Inner Product):
    - "Flat" = brute-force (exact search, no approximation)
    - "IP" = Inner Product (= cosine similarity when vectors are normalized)
    
    For larger datasets (>100k samples), IndexIVFFlat would be faster
    but requires approximate search (slight accuracy trade-off).

COSINE SIMILARITY EXPLAINED:
    Two vectors A and B have cosine similarity: cos(θ) = A·B / (|A| × |B|)
    
    Since we L2-normalize all embeddings: |A| = |B| = 1
    Therefore: cosine_similarity = A·B (just the dot product!)
    
    Similarity ranges:
    - 1.0 = identical (same defect type and appearance)
    - 0.8+ = very similar
    - 0.5-0.8 = somewhat similar
    - <0.5 = different

INPUT:
    - Query embedding (numpy float32 array, shape [768])

OUTPUT:
    - List of top-K results, each containing:
      * label: defect class name
      * similarity: cosine similarity score [0,1]
      * image_path: path to the similar historical image
      * metadata: additional info about the historical case

CONNECTS TO:
    - retrieval/build_index.py: Creates the index this module searches
    - pipeline/orchestrator.py: Called for each detected defect
    - app/pages/retrieval_page.py: Results displayed as gallery
"""

import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import yaml
from loguru import logger

if os.name == "nt":
    # Windows pip wheels for Torch and FAISS can load different OpenMP runtimes.
    # Without this, combining detector + retrieval may abort the interpreter.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error(
        "FAISS not installed. Install with:\n"
        "  pip install faiss-gpu  (with GPU support)\n"
        "  pip install faiss-cpu  (CPU only)"
    )


class FAISSSearchEngine:
    """
    FAISS-based approximate nearest neighbor search for defect retrieval.

    Maintains an in-memory FAISS index and associated metadata.
    The index maps embedding vectors to historical defect records.

    Example:
        engine = FAISSSearchEngine()
        engine.load()

        query_embedding = embedder.embed(new_defect_crop)
        results = engine.search(query_embedding, top_k=3)

        for r in results:
            print(f"Similar: {r['label']} | Similarity: {r['similarity']:.2f}")
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Initialize the search engine.

        Args:
            config: Retrieval configuration dictionary.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install faiss-gpu or faiss-cpu.")

        self.config = config or {}
        faiss_cfg = self.config.get("faiss", {})

        self.index_path = Path(
            faiss_cfg.get("index_path", "data/embeddings/faiss_index.bin")
        )
        self.metadata_path = Path(
            faiss_cfg.get("metadata_path", "data/embeddings/metadata.json")
        )
        self.top_k = self.config.get("retrieval", {}).get("top_k", 3)
        self.min_similarity = self.config.get("retrieval", {}).get("min_similarity", 0.5)
        self.embedding_dim = self.config.get("embedding", {}).get("embedding_dim", 768)

        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []

        logger.info(
            f"FAISSSearchEngine initialized | "
            f"index={self.index_path.name} | top_k={self.top_k}"
        )

    def load(self) -> None:
        """
        Load FAISS index and metadata from disk.

        The index file contains the compressed embedding vectors.
        The metadata JSON contains class labels and image paths.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}\n"
                "Build the index first: python retrieval/build_index.py"
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {self.metadata_path}\n"
                "Build the index first: python retrieval/build_index.py"
            )

        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        logger.info(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(
            f"FAISS index loaded | "
            f"vectors={self.index.ntotal} | "
            f"metadata={len(self.metadata)} records | "
            f"dim={self.index.d}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        filter_class: Optional[str] = None,
    ) -> list[dict]:
        """
        Find the most similar historical defects.

        Args:
            query_embedding: L2-normalized embedding of the query defect crop.
                            Shape: (768,) or (1, 768).
            top_k: Number of results to return. Uses config default if None.
            filter_class: Optional: only return results of this defect class.

        Returns:
            List of result dicts sorted by similarity (highest first):
            [{
                'label': 'missing_hole',
                'similarity': 0.92,
                'image_path': 'data/processed/crops/...',
                'rank': 1,
                ...metadata fields
            }]
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        k = top_k or self.top_k

        # Ensure query is 2D: (1, embedding_dim)
        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Validate embedding dimension
        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension {query.shape[1]} != index dimension {self.index.d}"
            )

        # FAISS search returns:
        # - distances: similarity scores (inner product = cosine for normalized vectors)
        # - indices: indices into the metadata list
        #
        # We search for more than k to allow for class filtering
        search_k = k * 3 if filter_class else k
        search_k = min(search_k, self.index.ntotal)  # Can't return more than we have

        distances, indices = self.index.search(query, search_k)

        # Parse results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                # FAISS returns -1 for invalid indices
                continue

            similarity = float(dist)  # Already cosine similarity (L2-normalized IP)

            # Filter by minimum similarity
            if similarity < self.min_similarity:
                continue

            meta = self.metadata[idx].copy()
            meta["similarity"] = round(similarity, 4)
            meta["rank"] = len(results) + 1

            # Apply class filter if specified
            if filter_class and meta.get("label") != filter_class:
                continue

            results.append(meta)

            if len(results) >= k:
                break

        top_similarity = results[0]["similarity"] if results else 0.0
        logger.debug(
            f"FAISS search complete | "
            f"found={len(results)} | "
            f"top_similarity={top_similarity:.3f}"
        )

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None,
    ) -> list[list[dict]]:
        """
        Search for multiple queries at once (batch mode).

        More efficient than calling search() in a loop because FAISS
        can parallelize batch queries.

        Args:
            query_embeddings: Array of shape (N, 768).
            top_k: Results per query.

        Returns:
            List of result lists, one per query.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        k = top_k or self.top_k
        query = np.array(query_embeddings, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query, k)

        all_results = []
        for q_idx in range(len(query)):
            results = []
            for dist, idx in zip(distances[q_idx], indices[q_idx]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                if float(dist) < self.min_similarity:
                    continue
                meta = self.metadata[idx].copy()
                meta["similarity"] = round(float(dist), 4)
                meta["rank"] = len(results) + 1
                results.append(meta)
            all_results.append(results)

        return all_results

    def get_index_stats(self) -> dict:
        """
        Return statistics about the loaded index.

        Returns:
            Dict with index metadata and statistics.
        """
        if self.index is None:
            return {"status": "not_loaded"}

        # Count per-class distribution in metadata
        class_counts: dict[str, int] = {}
        for meta in self.metadata:
            label = meta.get("label", "unknown")
            class_counts[label] = class_counts.get(label, 0) + 1

        return {
            "status": "loaded",
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.index.d,
            "index_type": type(self.index).__name__,
            "metadata_count": len(self.metadata),
            "class_distribution": class_counts,
            "index_path": str(self.index_path),
        }

    def is_loaded(self) -> bool:
        """Check if index is loaded and ready for search."""
        return self.index is not None and self.index.ntotal > 0
