"""Retrieval module: SigLIP embeddings + FAISS vector search."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.embedder import SigLIPEmbedder
    from retrieval.faiss_search import FAISSSearchEngine

__all__ = ["SigLIPEmbedder", "FAISSSearchEngine"]


def __getattr__(name: str):
    if name == "SigLIPEmbedder":
        from retrieval.embedder import SigLIPEmbedder
        return SigLIPEmbedder
    if name == "FAISSSearchEngine":
        from retrieval.faiss_search import FAISSSearchEngine
        return FAISSSearchEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
