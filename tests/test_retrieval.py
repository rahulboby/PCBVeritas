"""
Tests for Retrieval Module (FAISS + Embedder)
"""
import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestFAISSSearchEngine:
    """Tests for FAISSSearchEngine."""

    @pytest.fixture
    def mock_index_files(self, tmp_path):
        """Create mock FAISS index and metadata files."""
        import faiss

        dim = 768
        n = 20
        embeddings = np.random.rand(n, dim).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        index_path = tmp_path / "faiss_index.bin"
        faiss.write_index(index, str(index_path))

        class_names = [
            "missing_hole", "mouse_bite", "open_circuit",
            "short", "spur", "spurious_copper",
        ]
        metadata = [
            {"label": class_names[i % 6], "crop_path": f"/fake/crop_{i}.jpg"}
            for i in range(n)
        ]
        meta_path = tmp_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        return str(index_path), str(meta_path), embeddings

    @pytest.fixture
    def engine_with_mock_config(self, tmp_path, mock_index_files):
        """Create engine with mock config pointing to temp files."""
        index_path, meta_path, embeddings = mock_index_files
        config = {
            "embedding": {"embedding_dim": 768, "normalize": True, "device": "cpu"},
            "faiss": {
                "index_type": "IndexFlatIP",
                "index_path": index_path,
                "metadata_path": meta_path,
            },
            "retrieval": {"top_k": 3, "min_similarity": 0.0},
        }
        config_path = tmp_path / "retrieval.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from retrieval.faiss_search import FAISSSearchEngine
        engine = FAISSSearchEngine(config_path=str(config_path))
        engine.load()
        return engine, embeddings

    def test_load_index(self, engine_with_mock_config):
        engine, _ = engine_with_mock_config
        assert engine.is_loaded()
        assert engine.index.ntotal == 20

    def test_search_returns_results(self, engine_with_mock_config):
        engine, embeddings = engine_with_mock_config
        query = embeddings[0:1]
        results = engine.search(query, top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_search_top_result_is_self(self, engine_with_mock_config):
        """The most similar embedding to itself should have similarity ~1.0."""
        engine, embeddings = engine_with_mock_config
        query = embeddings[5]
        results = engine.search(query, top_k=1)
        assert len(results) >= 1
        assert results[0]["similarity"] > 0.99

    def test_search_result_fields(self, engine_with_mock_config):
        engine, embeddings = engine_with_mock_config
        results = engine.search(embeddings[0], top_k=2)
        for r in results:
            assert "label" in r
            assert "similarity" in r
            assert "rank" in r
            assert isinstance(r["similarity"], float)

    def test_search_batch(self, engine_with_mock_config):
        engine, embeddings = engine_with_mock_config
        queries = embeddings[:3]
        all_results = engine.search_batch(queries, top_k=2)
        assert len(all_results) == 3
        for res in all_results:
            assert isinstance(res, list)

    def test_get_index_stats(self, engine_with_mock_config):
        engine, _ = engine_with_mock_config
        stats = engine.get_index_stats()
        assert stats["status"] == "loaded"
        assert stats["total_vectors"] == 20
        assert stats["embedding_dim"] == 768

    def test_search_without_load_raises(self, tmp_path):
        config = {
            "embedding": {"embedding_dim": 768},
            "faiss": {
                "index_path": str(tmp_path / "idx.bin"),
                "metadata_path": str(tmp_path / "meta.json"),
            },
            "retrieval": {"top_k": 3, "min_similarity": 0.0},
        }
        config_path = tmp_path / "retrieval.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from retrieval.faiss_search import FAISSSearchEngine
        engine = FAISSSearchEngine(config_path=str(config_path))
        query = np.random.rand(768).astype(np.float32)
        with pytest.raises(RuntimeError):
            engine.search(query)

    def test_is_loaded_false_before_load(self, tmp_path):
        config = {
            "embedding": {"embedding_dim": 768},
            "faiss": {
                "index_path": str(tmp_path / "idx.bin"),
                "metadata_path": str(tmp_path / "meta.json"),
            },
            "retrieval": {"top_k": 3, "min_similarity": 0.0},
        }
        config_path = tmp_path / "retrieval.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from retrieval.faiss_search import FAISSSearchEngine
        engine = FAISSSearchEngine(config_path=str(config_path))
        assert not engine.is_loaded()


class TestSigLIPEmbedder:
    """Tests for SigLIPEmbedder (unit tests only, no model loading)."""

    def test_extract_class_from_filename(self):
        from retrieval.embedder import _extract_class_from_filename
        assert _extract_class_from_filename("missing_hole_img001_0.jpg") == "missing_hole"
        assert _extract_class_from_filename("short_PCB_00042_2.jpg") == "short"
        assert _extract_class_from_filename("spurious_copper_board_3.jpg") == "spurious_copper"
        assert _extract_class_from_filename("spur_test_0.png") == "spur"

    def test_extract_class_from_filename_unknown(self):
        from retrieval.embedder import _extract_class_from_filename
        result = _extract_class_from_filename("unknown_img_0.jpg")
        assert result == "unknown"
