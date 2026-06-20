"""
Tests for Knowledge Engine
"""
import pytest
import json
import tempfile
from pathlib import Path


@pytest.fixture
def sample_knowledge_json(tmp_path):
    """Create a minimal test knowledge base JSON."""
    data = {
        "version": "test",
        "defects": {
            "missing_hole": {
                "name": "Missing Hole",
                "code": "MH",
                "severity": "high",
                "category": "mechanical",
                "description": "A missing drilled hole in the PCB.",
                "visual_characteristics": ["No hole visible"],
                "causes": ["Drill bit breakage", "Misalignment"],
                "manufacturing_process": {
                    "stage": "Drilling",
                    "process_description": "CNC drilling failure.",
                    "contributing_factors": ["Tool wear"],
                },
                "potential_risks": ["Open circuit"],
                "severity_rationale": "High because circuit fails.",
                "inspection_procedure": ["Visual inspection"],
                "repair_recommendations": ["Re-drill", "Check drill bits"],
                "ipc_standard": "IPC-6012D",
                "common_in_layer_types": ["outer"],
                "detection_difficulty": "medium",
            },
            "short": {
                "name": "Short Circuit",
                "code": "SC",
                "severity": "critical",
                "category": "electrical",
                "description": "Unintended electrical connection.",
                "visual_characteristics": ["Copper bridge"],
                "causes": ["Over-etching"],
                "manufacturing_process": {
                    "stage": "Etching",
                    "process_description": "Etchant failure.",
                    "contributing_factors": ["Chemistry imbalance"],
                },
                "potential_risks": ["Component damage"],
                "severity_rationale": "Critical — immediate failure.",
                "inspection_procedure": ["AOI"],
                "repair_recommendations": ["Reject board"],
                "ipc_standard": "IPC-6012D",
                "common_in_layer_types": ["outer"],
                "detection_difficulty": "low",
            },
        },
        "severity_levels": {
            "critical": {"description": "Immediate failure", "color_code": "#FF0000"},
            "high": {"description": "High risk", "color_code": "#FF8800"},
            "medium": {"description": "Moderate risk", "color_code": "#FFCC00"},
            "low": {"description": "Minor concern", "color_code": "#44FF44"},
        },
    }
    knowledge_file = tmp_path / "defect_knowledge.json"
    with open(knowledge_file, "w") as f:
        json.dump(data, f)
    return str(knowledge_file)


class TestKnowledgeEngine:
    """Tests for KnowledgeEngine class."""

    def test_load_knowledge(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        assert engine.knowledge is not None
        assert "defects" in engine.knowledge

    def test_get_defect_info(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        info = engine.get_defect_info("missing_hole")
        assert info is not None
        assert info["name"] == "Missing Hole"
        assert info["severity"] == "high"

    def test_get_defect_info_not_found(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        info = engine.get_defect_info("nonexistent_defect")
        assert info is None

    def test_get_severity(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        assert engine.get_severity("missing_hole") == "high"
        assert engine.get_severity("short") == "critical"
        assert engine.get_severity("unknown_class") == "unknown"

    def test_get_causes(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        causes = engine.get_causes("missing_hole")
        assert isinstance(causes, list)
        assert len(causes) == 2
        assert "Drill bit breakage" in causes

    def test_get_repair_recommendations(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        recs = engine.get_repair_recommendations("missing_hole")
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_get_risks(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        risks = engine.get_risks("short")
        assert isinstance(risks, list)
        assert "Component damage" in risks

    def test_format_for_rag_prompt(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        context = engine.format_for_rag_prompt("missing_hole", 0.95)
        assert isinstance(context, str)
        assert len(context) > 50
        assert "Missing Hole" in context
        assert "HIGH" in context
        assert "95.0%" in context

    def test_format_for_rag_prompt_with_retrieved(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        retrieved = [
            {"label": "missing_hole", "similarity": 0.92},
            {"label": "missing_hole", "similarity": 0.88},
        ]
        context = engine.format_for_rag_prompt("missing_hole", 0.95, retrieved)
        assert "SIMILAR HISTORICAL" in context
        assert "0.92" in context

    def test_get_all_defect_classes(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        classes = engine.get_all_defect_classes()
        assert "missing_hole" in classes
        assert "short" in classes
        assert len(classes) == 2

    def test_get_severity_color(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        color = engine.get_severity_color("short")
        assert color == "#FF0000"

    def test_get_summary_stats(self, sample_knowledge_json):
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        stats = engine.get_summary_stats()
        assert stats["total_defect_types"] == 2
        assert "severity_distribution" in stats

    def test_missing_file_raises_error(self, tmp_path):
        from knowledge.knowledge_engine import KnowledgeEngine
        with pytest.raises(FileNotFoundError):
            KnowledgeEngine(str(tmp_path / "nonexistent.json"))

    def test_normalize_class_name(self, sample_knowledge_json):
        """Test that 'Missing Hole' normalizes to 'missing_hole'."""
        from knowledge.knowledge_engine import KnowledgeEngine
        engine = KnowledgeEngine(sample_knowledge_json)
        info = engine.get_defect_info("Missing Hole")
        assert info is not None
        assert info["code"] == "MH"
