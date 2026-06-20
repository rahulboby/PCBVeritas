"""Tests for synthetic instruction dataset generation."""
import json
import pytest
from pathlib import Path


@pytest.fixture
def knowledge_path():
    return str(Path(__file__).parent.parent / "knowledge" / "defect_knowledge.json")


class TestGenerateDataset:
    def test_generate_small_dataset(self, knowledge_path, tmp_path):
        from llm.fine_tuning.generate_dataset import generate_dataset
        output = str(tmp_path / "test_instructions.json")
        samples = generate_dataset(
            knowledge_path=knowledge_path,
            output_path=output,
            n_samples=30,
            seed=42,
        )
        assert len(samples) == 30
        assert Path(output).exists()

        with open(output) as f:
            loaded = json.load(f)
        assert len(loaded) == 30

    def test_sample_has_required_fields(self, knowledge_path, tmp_path):
        from llm.fine_tuning.generate_dataset import generate_dataset
        output = str(tmp_path / "test_instructions.json")
        samples = generate_dataset(knowledge_path=knowledge_path,
                                   output_path=output, n_samples=10, seed=0)
        for sample in samples:
            assert "instruction" in sample
            assert "input" in sample
            assert "output" in sample
            assert isinstance(sample["instruction"], str)
            assert isinstance(sample["output"], str)
            assert len(sample["instruction"]) > 5
            assert len(sample["output"]) > 20

    def test_all_defects_represented(self, knowledge_path, tmp_path):
        from llm.fine_tuning.generate_dataset import generate_dataset
        output = str(tmp_path / "test_instructions.json")
        samples = generate_dataset(knowledge_path=knowledge_path,
                                   output_path=output, n_samples=100, seed=1)
        classes = [
            "missing_hole", "mouse_bite", "open_circuit",
            "short", "spur", "spurious_copper"
        ]
        all_text = " ".join(s["input"] + s["output"] for s in samples).lower()
        for cls in classes:
            assert cls.replace("_", " ") in all_text or cls in all_text, \
                f"Class '{cls}' not found in dataset"

    def test_explanation_sample_generator(self, knowledge_path):
        import json
        with open(knowledge_path) as f:
            knowledge = json.load(f)
        from llm.fine_tuning.generate_dataset import generate_explanation_sample
        info = knowledge["defects"]["missing_hole"]
        sample = generate_explanation_sample("missing_hole", info)
        assert "instruction" in sample
        assert "output" in sample
        assert "Missing Hole" in sample["output"]

    def test_report_sample_generator(self, knowledge_path):
        import json
        with open(knowledge_path) as f:
            knowledge = json.load(f)
        from llm.fine_tuning.generate_dataset import generate_report_sample
        info = knowledge["defects"]["short"]
        sample = generate_report_sample("short", info, confidence=0.95)
        assert "PCB" in sample["output"] or "Short" in sample["output"]
        assert "95.0" in sample["output"] or "95" in sample["input"]

    def test_comparison_sample_generator(self, knowledge_path):
        import json
        with open(knowledge_path) as f:
            knowledge = json.load(f)
        from llm.fine_tuning.generate_dataset import generate_comparison_sample
        info1 = knowledge["defects"]["missing_hole"]
        info2 = knowledge["defects"]["short"]
        sample = generate_comparison_sample("missing_hole", info1, "short", info2)
        assert "instruction" in sample
        assert "Missing Hole" in sample["output"] or "missing_hole" in sample["output"].lower()
        assert "Short" in sample["output"] or "short" in sample["output"].lower()
