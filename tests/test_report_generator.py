"""Tests for OpenAI-compatible RAG report generation."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest


class FakeOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )
        FakeOpenAI.instances.append(self)

    def _create_completion(self, **payload):
        self.requests.append(payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="# PCB Inspection Report\n\nGenerated.")
                )
            ]
        )


@pytest.fixture(autouse=True)
def reset_fake_openai():
    FakeOpenAI.instances = []


def _llm_config(provider: str = "lm_studio") -> dict:
    from configs.settings import LLM_CONFIG

    config = deepcopy(LLM_CONFIG)
    config["provider"] = provider
    return config


def test_lm_studio_client_uses_configured_endpoint(monkeypatch):
    from llm.inference import report_generator
    from llm.inference.report_generator import PCBReportGenerator

    monkeypatch.setattr(report_generator, "OpenAI", FakeOpenAI)
    config = _llm_config("lm_studio")
    config["providers"]["lm_studio"]["model"] = "loaded-local-model"

    generator = PCBReportGenerator(config=config)
    generator.load_model()

    client = FakeOpenAI.instances[-1]
    assert client.kwargs["base_url"] == "http://localhost:1234/v1"
    assert client.kwargs["api_key"] == "lm-studio"
    assert generator.device_summary() == "api ready: lm_studio / loaded-local-model"


def test_groq_client_uses_env_key_and_endpoint(monkeypatch):
    from llm.inference import report_generator
    from llm.inference.report_generator import PCBReportGenerator

    monkeypatch.setattr(report_generator, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    generator = PCBReportGenerator(config=_llm_config("groq"))
    generator.load_model()

    client = FakeOpenAI.instances[-1]
    assert client.kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert client.kwargs["api_key"] == "test-groq-key"


def test_grok_client_uses_env_key_and_endpoint(monkeypatch):
    from llm.inference import report_generator
    from llm.inference.report_generator import PCBReportGenerator

    monkeypatch.setattr(report_generator, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")

    generator = PCBReportGenerator(config=_llm_config("grok"))
    generator.load_model()

    client = FakeOpenAI.instances[-1]
    assert client.kwargs["base_url"] == "https://api.x.ai/v1"
    assert client.kwargs["api_key"] == "test-xai-key"


def test_rag_prompt_includes_detection_knowledge_and_retrieval(monkeypatch):
    from llm.inference import report_generator
    from llm.inference.report_generator import PCBReportGenerator

    monkeypatch.setattr(report_generator, "OpenAI", FakeOpenAI)

    generator = PCBReportGenerator(config=_llm_config())
    report = generator.generate_report(
        defect_name="missing_hole",
        confidence=0.967,
        severity="high",
        knowledge_context="Stage: Mechanical Drilling\nRepair: Re-drill if board allows.",
        retrieved_cases=[
            {
                "label": "missing_hole",
                "similarity": 0.92,
                "source_image": "PCB_00142.jpg",
            }
        ],
    )

    request = FakeOpenAI.instances[-1].requests[-1]
    user_message = request["messages"][1]["content"]

    assert "Generated" in report
    assert "RAG_CONTEXT_CHUNKS" in user_message
    assert "missing_hole" in user_message
    assert "Mechanical Drilling" in user_message
    assert "PCB_00142.jpg" in user_message
    assert request["model"] == _llm_config()["providers"]["lm_studio"]["model"]


def test_missing_grok_key_raises_clear_error(monkeypatch):
    from llm.inference import report_generator
    from llm.inference.report_generator import PCBReportGenerator

    monkeypatch.setattr(report_generator, "OpenAI", FakeOpenAI)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        PCBReportGenerator(config=_llm_config("grok")).load_model()


def test_pipeline_falls_back_when_llm_api_key_missing(monkeypatch):
    from detector.detector import Detection
    from llm.inference import report_generator
    from pipeline.orchestrator import PCBInspectionPipeline

    monkeypatch.setattr(report_generator, "OpenAI", FakeOpenAI)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    pipeline = PCBInspectionPipeline(
        enable_xai=False,
        enable_retrieval=False,
        enable_llm=True,
    )
    pipeline._report_generator = None
    pipeline.enable_llm = True

    from configs import settings

    original_provider = settings.LLM_CONFIG["provider"]
    settings.LLM_CONFIG["provider"] = "grok"
    try:
        detection = Detection(
            class_name="missing_hole",
            class_id=0,
            confidence=0.9,
            bbox=[10.0, 10.0, 50.0, 50.0],
            bbox_normalized=[0.3, 0.3, 0.2, 0.2],
        )
        report = pipeline._stage_report(
            [detection],
            retrieved_cases={},
            knowledge_contexts={"missing_hole": "High severity."},
        )
    finally:
        settings.LLM_CONFIG["provider"] = original_provider

    assert "Template Mode" in report
    assert "missing_hole" in report.lower() or "Missing Hole" in report
