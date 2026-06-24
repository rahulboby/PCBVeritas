"""
RAG-Powered PCB Inspection Report Generator
============================================
PURPOSE:
    Combines defect detection results, retrieved similar cases, and
    knowledge base information to generate comprehensive expert reports
    using the fine-tuned Qwen2.5 model.

    This is the RAG (Retrieval-Augmented Generation) heart of the system.
    RAG = Detection Context + Retrieved Cases + Knowledge → LLM → Report

INPUT:
    - List of detected defects (from detector)
    - Retrieved similar cases (from FAISS)
    - Knowledge base context (from KnowledgeEngine)

OUTPUT:
    - Markdown-formatted inspection report
    - Structured defect summary (JSON-serializable)

HOW RAG WORKS:
    1. Detection: YOLOv8 identifies "missing_hole" at 96.7% confidence
    2. Retrieval: FAISS finds 3 similar historical defects (visual match)
    3. Knowledge: Engine retrieves causes, risks, repair procedures
    4. Prompt: Combine all context into a rich prompt for the LLM
    5. Generation: Fine-tuned Qwen generates the expert report

WHY RAG IS BETTER THAN PURE LLM:
    - Without RAG: Model relies only on training knowledge
    - With RAG:    Model grounds response in actual detection data
                   + specific historical cases + domain knowledge
    - Result: More accurate, specific, and actionable reports

CONNECTS TO:
    - pipeline/orchestrator.py: Called as final pipeline step
    - app/pages/report_page.py: Report displayed in Streamlit
"""

from pathlib import Path
from typing import Optional
import yaml
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class PCBReportGenerator:
    """
    Generates expert PCB inspection reports using a fine-tuned Qwen model
    with Retrieval-Augmented Generation (RAG).

    Supports both the fine-tuned LoRA model and the base model as fallback.

    Example:
        generator = PCBReportGenerator()
        report = generator.generate_report(
            detections=[{'class_name': 'missing_hole', 'confidence': 0.967}],
            retrieved_cases=[{'label': 'missing_hole', 'similarity': 0.92}],
            knowledge_context="Defect: Missing Hole | Severity: HIGH..."
        )
        print(report)
    """

    def __init__(
        self,
        config_path: str = "configs/llm.yaml",
        ft_config_path: str = "configs/fine_tuning.yaml",
        use_fine_tuned: bool = True,
    ) -> None:
        """
        Initialize the report generator.

        Args:
            config_path: LLM configuration path.
            ft_config_path: Fine-tuning configuration path.
            use_fine_tuned: Load fine-tuned LoRA model if True, else base model.
        """
        self.config = self._load_config(config_path)
        self.ft_config = self._load_config(ft_config_path)
        self.use_fine_tuned = use_fine_tuned

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._model_loaded = False

        logger.info(
            f"PCBReportGenerator initialized | "
            f"fine_tuned={'yes' if use_fine_tuned else 'no'}"
        )

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {path}")
            return {}

    def load_model(self) -> None:
        """
        Load the tokenizer and language model.

        Tries fine-tuned LoRA model first, falls back to base model.
        Lazy loading — called on first use.
        """
        if self._model_loaded:
            return

        base_dir = self.ft_config.get("base_model", {}).get(
            "local_dir", "models/llm/qwen2.5-1.5b"
        )
        fine_tuned_dir = self.ft_config.get("training", {}).get(
            "output_dir", "models/llm/fine_tuned_qwen"
        )
        model_name = self.ft_config.get("base_model", {}).get(
            "name", "Qwen/Qwen2.5-1.5B-Instruct"
        )

        # --- Load tokenizer ---
        load_path = base_dir if Path(base_dir).exists() else model_name
        logger.info(f"Loading tokenizer from: {load_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Load model ---
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Loading base model: {load_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )

        # --- Apply LoRA weights if available ---
        fine_tuned_path = Path(fine_tuned_dir)
        if self.use_fine_tuned and fine_tuned_path.exists():
            logger.info(f"Loading LoRA adapter from: {fine_tuned_dir}")
            try:
                self.model = PeftModel.from_pretrained(base_model, fine_tuned_dir)
                self.model = self.model.merge_and_unload()  # Merge for faster inference
                logger.info("Fine-tuned LoRA model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter: {e}. Using base model.")
                self.model = base_model
        else:
            logger.info("Using base Qwen model (not fine-tuned)")
            self.model = base_model

        self.model.eval()
        self._model_loaded = True

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"Model ready | {total_params:.2f}B parameters")

    def _build_rag_prompt(
        self,
        defect_name: str,
        confidence: float,
        knowledge_context: str,
        retrieved_cases: Optional[list[dict]] = None,
        severity: str = "unknown",
    ) -> list[dict]:
        """
        Build the RAG prompt as a chat messages list.

        The prompt combines:
        1. System message: Expert role definition
        2. User message: Detection data + knowledge + retrieved cases

        Args:
            defect_name: Detected defect class name.
            confidence: Detection confidence (0-1).
            knowledge_context: Pre-formatted knowledge base context.
            retrieved_cases: List of similar historical cases from FAISS.
            severity: Severity level string.

        Returns:
            List of message dicts for chat template.
        """
        system_prompt = self.config.get(
            "system_prompt",
            "You are an expert PCB inspection engineer. Generate detailed inspection reports."
        )

        # Build retrieved cases section
        retrieved_section = ""
        if retrieved_cases:
            retrieved_section = "\n\nSIMILAR HISTORICAL CASES (from database):\n"
            for i, case in enumerate(retrieved_cases, 1):
                label = case.get("label", "Unknown").replace("_", " ").title()
                sim = case.get("similarity", 0)
                retrieved_section += f"  Case {i}: {label} (visual similarity: {sim:.2f})\n"
            retrieved_section += (
                f"\nNote: {len(retrieved_cases)} similar cases confirm this defect "
                f"pattern has been observed in production before."
            )

        user_message = f"""Generate a comprehensive PCB inspection report for the following detected defect.

DETECTION DATA:
===============
{knowledge_context}

Severity: {severity.upper()}
Confidence: {confidence:.1%}
{retrieved_section}

Please produce a structured inspection report including:
1. Executive Summary
2. Technical Analysis
3. Root Cause Assessment
4. Risk Assessment
5. Corrective Actions
6. Recommendations

Format as a professional engineering report using markdown."""

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
        Generate an inspection report for a detected defect.

        Args:
            defect_name: Detected defect class (e.g., "missing_hole").
            confidence: Detection confidence score (0-1).
            knowledge_context: Context from KnowledgeEngine.format_for_rag_prompt().
            retrieved_cases: Similar historical defects from FAISS search.
            severity: Severity level string.
            max_new_tokens: Override for maximum response length.

        Returns:
            Formatted inspection report as markdown string.
        """
        # Lazy load model on first call
        if not self._model_loaded:
            logger.info("Loading LLM for first time (this may take ~30 seconds)...")
            self.load_model()

        # Build prompt
        messages = self._build_rag_prompt(
            defect_name=defect_name,
            confidence=confidence,
            knowledge_context=knowledge_context,
            retrieved_cases=retrieved_cases,
            severity=severity,
        )

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("tokenizer", {}).get("max_input_length", 2048),
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generation config
        gen_config = self.config.get("generation", {})
        max_tokens = max_new_tokens or gen_config.get("max_new_tokens", 512)

        logger.info(f"Generating report for: {defect_name} (max_tokens={max_tokens})")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=gen_config.get("temperature", 0.3),
                top_p=gen_config.get("top_p", 0.9),
                top_k=gen_config.get("top_k", 50),
                repetition_penalty=gen_config.get("repetition_penalty", 1.1),
                do_sample=gen_config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (not the input)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        report = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        logger.info(f"Report generated | length={len(report.split())} words")
        return report.strip()

    def generate_multi_defect_report(
        self,
        detections: list[dict],
        knowledge_engine: "KnowledgeEngine",
        retrieved_cases_map: Optional[dict[int, list[dict]]] = None,
    ) -> str:
        """
        Generate a consolidated report for multiple detected defects.

        Args:
            detections: List of detection dicts (with class_name, confidence).
            knowledge_engine: KnowledgeEngine instance for context.
            retrieved_cases_map: Dict mapping detection index to retrieved cases.

        Returns:
            Full multi-defect inspection report.
        """
        from knowledge.knowledge_engine import KnowledgeEngine

        if not detections:
            return "# PCB Inspection Report\n\n**Result: No defects detected.**\n\nThe PCB passed visual inspection."

        if not self._model_loaded:
            self.load_model()

        sections = []

        # Header
        sections.append("# PCB AUTOMATED INSPECTION REPORT")
        sections.append(f"\n**Total Defects Detected:** {len(detections)}")

        # Severity summary
        severity_counts: dict[str, int] = {}
        for det in detections:
            sev = knowledge_engine.get_severity(det.get("class_name", ""))
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if "critical" in severity_counts:
            sections.append(f"\n🔴 **CRITICAL DEFECTS: {severity_counts['critical']}** — IMMEDIATE ACTION REQUIRED")
        if "high" in severity_counts:
            sections.append(f"🟠 **HIGH SEVERITY: {severity_counts['high']}** — Engineering review required")
        if "medium" in severity_counts:
            sections.append(f"🟡 **MEDIUM SEVERITY: {severity_counts['medium']}** — Risk assessment required")
        if "low" in severity_counts:
            sections.append(f"🟢 **LOW SEVERITY: {severity_counts['low']}** — Document and monitor")

        sections.append("\n---\n")

        # Per-defect reports
        for idx, det in enumerate(detections):
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0.0)
            retrieved = (retrieved_cases_map or {}).get(idx, [])

            knowledge_context = knowledge_engine.format_for_rag_prompt(
                class_name, confidence, retrieved
            )
            severity = knowledge_engine.get_severity(class_name)

            sections.append(f"## Defect {idx+1}: {class_name.replace('_', ' ').title()}")
            sections.append(f"**Confidence:** {confidence:.1%} | **Severity:** {severity.upper()}\n")

            report = self.generate_report(
                defect_name=class_name,
                confidence=confidence,
                knowledge_context=knowledge_context,
                retrieved_cases=retrieved,
                severity=severity,
            )
            sections.append(report)
            sections.append("\n---\n")

        # Footer
        sections.append("*Report generated by PCB-VLM-XAI Automated Inspection System*")
        sections.append("*Powered by: YOLOv8s + SigLIP + FAISS + LoRA Fine-tuned Qwen2.5*")

        return "\n".join(sections)

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded


from typing import Optional
