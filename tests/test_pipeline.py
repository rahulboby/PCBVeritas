"""Tests for Pipeline Orchestrator, FAISS, and Embedder utilities."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestPipelineResult:
    def test_defaults(self):
        from pipeline.orchestrator import PipelineResult
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = PipelineResult(image=image)
        assert result.num_defects == 0
        assert result.detections == []
        assert result.success is True
        assert result.report == ""

    def test_get_summary(self):
        from pipeline.orchestrator import PipelineResult
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = PipelineResult(image=image, num_defects=2,
                                processing_time=1.5,
                                stage_times={"detection": 0.05},
                                success=True)
        summary = result.get_summary()
        assert summary["num_defects"] == 2
        assert summary["success"] is True

    def test_error_state(self):
        from pipeline.orchestrator import PipelineResult
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = PipelineResult(image=image, success=False, error_message="Model not found")
        assert not result.success
        assert "not found" in result.error_message


class TestPCBInspectionPipeline:
    def test_initialization(self):
        from pipeline.orchestrator import PCBInspectionPipeline
        pipeline = PCBInspectionPipeline(enable_xai=False, enable_retrieval=False, enable_llm=False)
        assert pipeline.enable_xai is False
        assert pipeline.enable_retrieval is False
        assert pipeline.enable_llm is False

    def test_fallback_report_no_defects(self):
        from pipeline.orchestrator import PCBInspectionPipeline
        pipeline = PCBInspectionPipeline()
        report = pipeline._fallback_report([], {})
        assert "No defects" in report

    def test_fallback_report_with_defects(self):
        from pipeline.orchestrator import PCBInspectionPipeline
        from detector.detector import Detection
        pipeline = PCBInspectionPipeline()
        detections = [Detection(
            class_name="missing_hole", class_id=0, confidence=0.9,
            bbox=[10.0, 10.0, 50.0, 50.0], bbox_normalized=[0.3, 0.3, 0.2, 0.2],
        )]
        report = pipeline._fallback_report(detections, {"missing_hole": "High severity."})
        assert "missing_hole" in report.lower() or "Missing Hole" in report


class TestFAISSSearch:
    def test_not_loaded(self):
        from retrieval.faiss_search import FAISSSearchEngine
        with patch('retrieval.faiss_search.FAISS_AVAILABLE', True):
            engine = FAISSSearchEngine.__new__(FAISSSearchEngine)
            engine.index = None
            engine.metadata = []
            assert not engine.is_loaded()

    def test_search_raises_when_not_loaded(self):
        from retrieval.faiss_search import FAISSSearchEngine
        with patch('retrieval.faiss_search.FAISS_AVAILABLE', True):
            engine = FAISSSearchEngine.__new__(FAISSSearchEngine)
            engine.index = None
            with pytest.raises(RuntimeError, match="Index not loaded"):
                engine.search(np.random.randn(768).astype(np.float32))

    def test_stats_not_loaded(self):
        from retrieval.faiss_search import FAISSSearchEngine
        with patch('retrieval.faiss_search.FAISS_AVAILABLE', True):
            engine = FAISSSearchEngine.__new__(FAISSSearchEngine)
            engine.index = None
            engine.metadata = []
            stats = engine.get_index_stats()
            assert stats["status"] == "not_loaded"


class TestEmbedder:
    def test_extract_class_from_filename(self):
        from retrieval.embedder import _extract_class_from_filename
        assert _extract_class_from_filename("missing_hole_PCB_001_0.jpg") == "missing_hole"
        assert _extract_class_from_filename("short_image_002_1.jpg") == "short"
        assert _extract_class_from_filename("spurious_copper_test_003_2.png") == "spurious_copper"
        assert _extract_class_from_filename("unknown_file.jpg") == "unknown"
        assert _extract_class_from_filename("mouse_bite_PCB_100_3.jpg") == "mouse_bite"
        assert _extract_class_from_filename("open_circuit_img_005_0.jpg") == "open_circuit"
