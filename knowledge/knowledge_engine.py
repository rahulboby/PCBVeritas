"""
Knowledge Engine for PCB Defect Analysis
=========================================
PURPOSE:
    Provides structured industrial knowledge about PCB defects.
    Answers questions like: "What causes a short circuit?" or
    "How should I repair a missing hole defect?"

INPUT:
    Defect class name (e.g., "missing_hole", "short")

OUTPUT:
    Structured knowledge dict containing severity, causes,
    risks, repair recommendations, and manufacturing context.

HOW IT WORKS:
    Loads a JSON knowledge base containing expert-curated
    information about each PCB defect class. Provides methods
    to retrieve and format this information for RAG prompts.

CONNECTS TO:
    - pipeline/orchestrator.py: Provides knowledge for LLM prompt
    - llm/fine_tuning/generate_dataset.py: Source for synthetic data
    - app/pages/: Displays knowledge in Streamlit UI
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class KnowledgeEngine:
    """
    Retrieves structured PCB defect knowledge for RAG-based report generation.
    
    The knowledge engine acts as the 'expert system' layer - it knows all the
    manufacturing details, risk factors, and repair procedures for each defect
    type. This knowledge is combined with visual detection results to generate
    comprehensive inspection reports.
    """

    def __init__(self, knowledge_path: str = "knowledge/defect_knowledge.json") -> None:
        """
        Initialize the knowledge engine by loading the JSON knowledge base.

        Args:
            knowledge_path: Path to the defect_knowledge.json file.
        """
        self.knowledge_path = Path(knowledge_path)
        self.knowledge: dict[str, Any] = {}
        self._load_knowledge()
        logger.info(f"KnowledgeEngine initialized with {len(self.knowledge.get('defects', {}))} defect types")

    def _load_knowledge(self) -> None:
        """Load and validate the knowledge base from JSON."""
        if not self.knowledge_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at: {self.knowledge_path}\n"
                "Please ensure knowledge/defect_knowledge.json exists."
            )
        with open(self.knowledge_path, "r") as f:
            self.knowledge = json.load(f)
        
        # Validate structure
        required_keys = ["defects", "severity_levels"]
        for key in required_keys:
            if key not in self.knowledge:
                raise ValueError(f"Knowledge base missing required key: '{key}'")
        
        logger.info(f"Knowledge base v{self.knowledge.get('version', 'unknown')} loaded")

    def get_defect_info(self, defect_class: str) -> Optional[dict[str, Any]]:
        """
        Retrieve complete knowledge for a specific defect class.

        Args:
            defect_class: Class name like "missing_hole", "open_circuit", etc.
                         Can also accept display names like "Missing Hole".

        Returns:
            Dictionary with all defect knowledge, or None if not found.
        """
        # Normalize input: "Missing Hole" -> "missing_hole"
        normalized = defect_class.lower().replace(" ", "_").replace("-", "_")
        
        defects = self.knowledge.get("defects", {})
        
        if normalized in defects:
            return defects[normalized]
        
        # Try fuzzy match
        for key in defects:
            if normalized in key or key in normalized:
                logger.warning(f"Fuzzy matched '{defect_class}' to '{key}'")
                return defects[key]
        
        logger.warning(f"No knowledge found for defect class: '{defect_class}'")
        return None

    def get_severity(self, defect_class: str) -> str:
        """
        Get severity level for a defect class.

        Args:
            defect_class: Defect class name.

        Returns:
            Severity string: 'critical', 'high', 'medium', or 'low'.
        """
        info = self.get_defect_info(defect_class)
        if info:
            return info.get("severity", "unknown")
        return "unknown"

    def get_causes(self, defect_class: str) -> list[str]:
        """
        Get list of manufacturing causes for a defect.

        Args:
            defect_class: Defect class name.

        Returns:
            List of cause strings.
        """
        info = self.get_defect_info(defect_class)
        if info:
            return info.get("causes", [])
        return []

    def get_repair_recommendations(self, defect_class: str) -> list[str]:
        """
        Get repair and corrective action recommendations.

        Args:
            defect_class: Defect class name.

        Returns:
            List of recommendation strings.
        """
        info = self.get_defect_info(defect_class)
        if info:
            return info.get("repair_recommendations", [])
        return []

    def get_risks(self, defect_class: str) -> list[str]:
        """
        Get potential risk factors for a defect.

        Args:
            defect_class: Defect class name.

        Returns:
            List of risk strings.
        """
        info = self.get_defect_info(defect_class)
        if info:
            return info.get("potential_risks", [])
        return []

    def format_for_rag_prompt(
        self,
        defect_class: str,
        confidence: float,
        retrieved_cases: Optional[list[dict]] = None,
    ) -> str:
        """
        Format knowledge into a prompt-ready context string for the LLM.
        
        This is the core RAG (Retrieval-Augmented Generation) function.
        It combines structured knowledge with retrieved similar cases to
        build a rich context that the LLM can use to generate expert reports.

        Args:
            defect_class: The detected defect class name.
            confidence: Detection confidence score (0.0 to 1.0).
            retrieved_cases: Optional list of similar historical defects from FAISS.

        Returns:
            Formatted context string ready for LLM prompt injection.
        """
        info = self.get_defect_info(defect_class)
        if not info:
            return f"Detected defect: {defect_class} (confidence: {confidence:.1%}). No detailed knowledge available."

        # Format severity info
        severity = info.get("severity", "unknown").upper()
        severity_info = self.knowledge.get("severity_levels", {}).get(
            info.get("severity", ""), {}
        )

        # Build causes string
        causes_str = "\n".join(f"  - {c}" for c in info.get("causes", [])[:4])
        
        # Build risks string
        risks_str = "\n".join(f"  - {r}" for r in info.get("potential_risks", [])[:3])
        
        # Build recommendations string
        recs_str = "\n".join(f"  - {r}" for r in info.get("repair_recommendations", [])[:4])

        # Build retrieved cases context
        retrieved_context = ""
        if retrieved_cases:
            retrieved_context = "\n\nSIMILAR HISTORICAL CASES RETRIEVED:\n"
            for i, case in enumerate(retrieved_cases, 1):
                retrieved_context += (
                    f"Case {i}: {case.get('label', 'Unknown')} "
                    f"(similarity: {case.get('similarity', 0):.2f})\n"
                )

        # Assemble full context
        context = f"""
DEFECT INSPECTION CONTEXT:

Detected Defect: {info.get('name', defect_class)}
Detection Confidence: {confidence:.1%}
Defect Code: {info.get('code', 'N/A')}
Severity Level: {severity}
Category: {info.get('category', 'N/A').upper()}

TECHNICAL DESCRIPTION:
{info.get('description', 'No description available.')}

MANUFACTURING PROCESS:
Stage: {info.get('manufacturing_process', {}).get('stage', 'N/A')}
{info.get('manufacturing_process', {}).get('process_description', '')}

ROOT CAUSE ANALYSIS - Likely Manufacturing Causes:
{causes_str}

RISK ASSESSMENT:
{risks_str}

IPC STANDARD REFERENCE: {info.get('ipc_standard', 'N/A')}

CORRECTIVE ACTIONS AND REPAIR RECOMMENDATIONS:
{recs_str}
{retrieved_context}
"""
        return context.strip()

    def get_all_defect_classes(self) -> list[str]:
        """
        Get list of all defect class names in the knowledge base.

        Returns:
            List of defect class name strings.
        """
        return list(self.knowledge.get("defects", {}).keys())

    def get_severity_color(self, defect_class: str) -> str:
        """
        Get the hex color code associated with a defect's severity level.

        Args:
            defect_class: Defect class name.

        Returns:
            Hex color string like '#FF0000'.
        """
        severity = self.get_severity(defect_class)
        severity_info = self.knowledge.get("severity_levels", {}).get(severity, {})
        return severity_info.get("color_code", "#FFFFFF")

    def get_summary_stats(self) -> dict[str, Any]:
        """
        Get summary statistics about the knowledge base.

        Returns:
            Dictionary with counts and distribution info.
        """
        defects = self.knowledge.get("defects", {})
        severity_counts: dict[str, int] = {}
        
        for defect_info in defects.values():
            sev = defect_info.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_defect_types": len(defects),
            "severity_distribution": severity_counts,
            "version": self.knowledge.get("version", "unknown"),
        }


if __name__ == "__main__":
    # Quick demo / smoke test
    engine = KnowledgeEngine()
    
    print("=== PCB Knowledge Engine Demo ===\n")
    print(f"Stats: {engine.get_summary_stats()}\n")
    
    for defect in ["missing_hole", "open_circuit", "short"]:
        print(f"\n{'='*50}")
        print(f"Defect: {defect.upper()}")
        print(f"Severity: {engine.get_severity(defect)}")
        print(f"Top cause: {engine.get_causes(defect)[0] if engine.get_causes(defect) else 'N/A'}")
        print(f"Color: {engine.get_severity_color(defect)}")
    
    print("\n=== RAG Prompt Context ===")
    print(engine.format_for_rag_prompt("missing_hole", 0.967))
