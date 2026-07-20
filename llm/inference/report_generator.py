"""
RAG-powered PCB inspection report generator.

This module keeps the pipeline-facing report generator API, but sends the
assembled RAG context to an OpenAI-compatible chat endpoint such as LM Studio,
Groq, or xAI/Grok. Endpoint, model, and secret handling live in
configs/settings.py.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only before dependencies install
    OpenAI = None


class PCBReportGenerator:
    """
    Generates expert PCB inspection reports using RAG context and an
    OpenAI-compatible chat completions API.

    Example:
        generator = PCBReportGenerator()
        report = generator.generate_report(
            defect_name="missing_hole",
            confidence=0.967,
            retrieved_cases=[{"label": "missing_hole", "similarity": 0.92}],
            knowledge_context="Defect: Missing Hole | Severity: HIGH...",
        )
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.provider_name = (
            self.config.get("provider")
            or os.getenv("PCB_LLM_PROVIDER")
            or "groq"
        )
        self.provider_config = self._resolve_provider_config()
        self.model_name = self.provider_config.get("model", "local-model")

        self.client: Optional[Any] = None
        self._client_ready = False
        self._warmed_up = False

        logger.info(
            "PCBReportGenerator initialized | "
            f"provider={self.provider_name} | model={self.model_name}"
        )

    def _resolve_provider_config(self) -> dict:
        providers = self.config.get("providers", {})
        provider_config = providers.get(self.provider_name)
        if not provider_config:
            available = ", ".join(sorted(providers)) or "none"
            raise ValueError(
                f"Unknown LLM provider '{self.provider_name}'. "
                f"Available providers: {available}"
            )
        return provider_config

    def _resolve_api_key(self) -> str:
        load_dotenv()

        api_key_env = self.provider_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env) or self.provider_config.get("api_key_default", "")
        requires_key = bool(self.provider_config.get("requires_api_key", True))

        if requires_key and not api_key:
            raise RuntimeError(
                f"Missing API key for LLM provider '{self.provider_name}'. "
                f"Set {api_key_env} in your environment or local .env file."
            )

        return api_key or "local-api-key"

    def load_model(self) -> None:
        """
        Compatibility wrapper used by the pipeline preload step.

        No local model is loaded here. The method validates configuration and
        initializes an OpenAI-compatible API client.
        """
        if self._client_ready:
            return

        if OpenAI is None:
            raise ImportError(
                "The openai package is required for LLM report generation. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            )

        client_cfg = self.config.get("client", {})
        api_key = self._resolve_api_key()
        base_url = self.provider_config.get("base_url")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(client_cfg.get("timeout_seconds", 120)),
            max_retries=int(client_cfg.get("max_retries", 2)),
        )
        self._client_ready = True

        logger.info(
            "LLM API client ready | "
            f"provider={self.provider_name} | model={self.model_name} | base_url={base_url}"
        )

    def warmup(self) -> None:
        """
        Optionally run a tiny API request, disabled by default to avoid startup
        latency and paid-token usage.
        """
        if self._warmed_up:
            return

        self.load_model()

        if self.config.get("client", {}).get("warmup_request", False):
            self._generate_from_messages(
                [
                    {"role": "system", "content": "You are a PCB inspection assistant."},
                    {"role": "user", "content": "Reply with: ready"},
                ],
                max_tokens=4,
            )

        self._warmed_up = True
        logger.info("LLM API client warmup complete")

    def device_summary(self) -> str:
        """Human-readable LLM status for the Streamlit sidebar."""
        if not self._client_ready:
            return "api client not initialized"
        return f"api ready: {self.provider_name} / {self.model_name}"

    def _generate_from_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat messages to the configured OpenAI-compatible endpoint."""
        if not self._client_ready:
            self.load_model()

        gen_cfg = self.config.get("generation", {})
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens or gen_cfg.get("max_tokens", 700)),
            "temperature": float(gen_cfg.get("temperature", 0.3)),
            "top_p": float(gen_cfg.get("top_p", 0.9)),
        }

        logger.info(
            "Generating LLM report via API | "
            f"provider={self.provider_name} | model={self.model_name}"
        )
        completion = self.client.chat.completions.create(**payload)
        content = completion.choices[0].message.content
        report = self._coerce_text_content(content)

        logger.info(f"LLM report generated | length={len(report.split())} words")
        return report.strip()

    @staticmethod
    def _coerce_text_content(content: Any) -> str:
        """Normalize OpenAI-compatible response content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _format_retrieved_cases(self, retrieved_cases: Optional[list[dict]]) -> str:
        if not retrieved_cases:
            return "No similar historical cases were retrieved above the similarity threshold."

        lines = []
        for i, case in enumerate(retrieved_cases, 1):
            label = case.get("label", "unknown").replace("_", " ").title()
            similarity = case.get("similarity", 0.0)
            source = case.get("source_image") or case.get("image_path") or case.get("crop_path", "N/A")
            lines.append(f"{i}. {label} | similarity={similarity:.2f} | source={source}")
        return "\n".join(lines)

    def _build_rag_prompt(
        self,
        defect_name: str,
        confidence: float,
        knowledge_context: str,
        retrieved_cases: Optional[list[dict]] = None,
        severity: str = "unknown",
    ) -> list[dict[str, str]]:
        """
        Build chat messages from detection, retrieval, and knowledge chunks.
        """
        system_prompt = self.config.get(
            "system_prompt",
            "You are an expert PCB inspection engineer. Generate detailed inspection reports.",
        )

        retrieved_context = self._format_retrieved_cases(retrieved_cases)
        user_message = f"""Generate a comprehensive PCB inspection report for this detected defect.

RAG_CONTEXT_CHUNKS:

[Detection]
- Defect class: {defect_name}
- Confidence: {confidence:.1%}
- Severity: {severity.upper()}

[Knowledge Base]
{knowledge_context}

[Retrieved Similar Cases]
{retrieved_context}

Report requirements:
- Use markdown.
- Include Executive Summary, Technical Analysis, Root Cause Assessment, Risk Assessment, Corrective Actions, and Recommendations.
- Ground claims in the RAG context above.
- Be concise, technical, and actionable."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def generate_report(
        self,
        defect_name: str,
        confidence: float,
        knowledge_context: str,
        retrieved_cases: Optional[list[dict]] = None,
        severity: str = "unknown",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate an inspection report for a single detected defect.

        max_new_tokens is kept for backward compatibility and maps to the
        OpenAI-compatible max_tokens request field.
        """
        messages = self._build_rag_prompt(
            defect_name=defect_name,
            confidence=confidence,
            knowledge_context=knowledge_context,
            retrieved_cases=retrieved_cases,
            severity=severity,
        )
        return self._generate_from_messages(messages, max_tokens=max_new_tokens)

    def generate_multi_defect_report(
        self,
        detections: list[dict],
        knowledge_engine: "KnowledgeEngine",
        retrieved_cases_map: Optional[dict[int, list[dict]]] = None,
    ) -> str:
        """Generate one consolidated report for all detected defects."""
        if not detections:
            return "# PCB Inspection Report\n\n**Result: No defects detected.**\n\nThe PCB passed visual inspection."

        defect_chunks = []
        severity_counts: dict[str, int] = {}
        for idx, det in enumerate(detections):
            class_name = det.get("class_name", "unknown")
            confidence = float(det.get("confidence", 0.0))
            bbox = det.get("bbox", [])
            retrieved = (retrieved_cases_map or {}).get(idx, [])
            severity = knowledge_engine.get_severity(class_name)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            knowledge_context = knowledge_engine.format_for_rag_prompt(
                class_name, confidence, retrieved
            )

            defect_chunks.append(
                f"[Defect {idx + 1}: {class_name}]\n"
                f"Detection confidence: {confidence:.1%}\n"
                f"Bounding box: {bbox}\n"
                f"Severity: {severity.upper()}\n\n"
                f"Knowledge chunk:\n{knowledge_context}\n\n"
                f"Retrieved similar cases:\n{self._format_retrieved_cases(retrieved)}"
            )

        severity_summary = ", ".join(
            f"{count} {severity}" for severity, count in sorted(severity_counts.items())
        )
        user_message = f"""Generate one professional PCB inspection report for this image.

RAG_CONTEXT_CHUNKS:

[Inspection Summary]
- Total defects detected: {len(detections)}
- Severity summary: {severity_summary}

{chr(10).join(defect_chunks)}

Report requirements:
- Use markdown.
- Include Executive Summary, Defect Findings, Root Cause Assessment, Risk Assessment, Corrective Actions, and Recommendations.
- Consolidate repeated defects instead of repeating identical root-cause text.
- Ground every technical claim in the detection, retrieval, and knowledge chunks above."""

        messages = [
            {"role": "system", "content": self.config.get("system_prompt", "You are a PCB inspection expert.")},
            {"role": "user", "content": user_message},
        ]
        return self._generate_from_messages(messages)

    def is_loaded(self) -> bool:
        """Check if the API client is initialized and ready."""
        return self._client_ready
