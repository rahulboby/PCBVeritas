"""
Synthetic Instruction Dataset Generator for PCB LLM Fine-Tuning
================================================================
PURPOSE:
    Automatically generates hundreds of instruction-following training
    examples from the knowledge base. These examples teach the Qwen model
    to generate expert PCB inspection reports.

WHY SYNTHETIC DATA?
    We don't have thousands of real PCB inspection reports to fine-tune on.
    Instead, we use the structured knowledge base to generate diverse,
    high-quality instruction pairs that cover all defect types and scenarios.

    This is called "self-instruct" or "knowledge distillation to text" —
    converting structured data into natural language training examples.

INPUT:
    knowledge/defect_knowledge.json

OUTPUT:
    data/fine_tuning/pcb_instructions.json
    Format: [{"instruction": ..., "input": ..., "output": ...}, ...]

INSTRUCTION TEMPLATES:
    We create 10+ diverse question types per defect:
    1. Explain this defect
    2. What causes this defect?
    3. How severe is this defect?
    4. What are the risks?
    5. How should this be repaired?
    6. Write an inspection report
    7. What manufacturing process caused this?
    8. Classify the severity
    9. Compare two defects
    10. Predict consequences

USAGE:
    python llm/fine_tuning/generate_dataset.py
    python llm/fine_tuning/generate_dataset.py --samples 800 --output data/fine_tuning/
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any
from loguru import logger
from rich.console import Console

console = Console()

# Import knowledge engine
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge.knowledge_engine import KnowledgeEngine


# ============================================================
# Instruction templates (diverse question formulations)
# ============================================================

EXPLANATION_TEMPLATES = [
    "Explain the {defect} defect found on this PCB.",
    "What is a {defect} defect in PCB manufacturing?",
    "Describe the {defect} defect type in detail.",
    "I've found a {defect} on my PCB. What does this mean?",
    "As a PCB inspector, explain what {defect} is.",
    "Provide a technical description of the {defect} PCB defect.",
    "What are the visual characteristics of a {defect} defect?",
]

CAUSE_TEMPLATES = [
    "What causes {defect} defects in PCB manufacturing?",
    "Why do {defect} defects occur on printed circuit boards?",
    "List the manufacturing causes of {defect} in PCB production.",
    "What process failures lead to {defect} defects?",
    "Identify the root causes of {defect} in PCB fabrication.",
    "What went wrong in manufacturing to cause this {defect}?",
    "What manufacturing process errors result in {defect}?",
]

SEVERITY_TEMPLATES = [
    "How severe is a {defect} defect?",
    "What is the severity level of {defect} in PCB quality control?",
    "Should I reject a PCB with a {defect} defect?",
    "Rate the criticality of {defect} in electronics manufacturing.",
    "Is {defect} a critical defect or a minor concern?",
]

RISK_TEMPLATES = [
    "What are the risks of {defect} in a PCB?",
    "What problems can {defect} cause in an electronic circuit?",
    "What are the consequences of {defect} going undetected?",
    "How does {defect} affect circuit reliability?",
    "What failures can result from {defect} in electronics?",
]

REPAIR_TEMPLATES = [
    "How should {defect} be repaired?",
    "What corrective actions should be taken for {defect}?",
    "Provide repair recommendations for a PCB with {defect}.",
    "What steps should a manufacturing engineer take to fix {defect}?",
    "How can {defect} be prevented or corrected in production?",
    "What process improvements address {defect} defects?",
]

REPORT_TEMPLATES = [
    "Write an inspection report for a PCB with {defect} detected at {confidence}% confidence.",
    "Generate a quality control report for {defect} defect.",
    "Create a manufacturing defect report for {defect}.",
    "Write a technical report documenting {defect} in PCB inspection.",
    "Produce an IQC (Incoming Quality Control) report for {defect}.",
    "Document this {defect} defect for the manufacturing team.",
]

COMPARISON_TEMPLATES = [
    "Compare {defect1} and {defect2} defects in terms of severity.",
    "What is the difference between {defect1} and {defect2}?",
    "Which is more serious: {defect1} or {defect2}?",
    "How do {defect1} and {defect2} differ in their manufacturing causes?",
]

MANUFACTURING_TEMPLATES = [
    "At which stage of PCB manufacturing does {defect} occur?",
    "Describe the manufacturing process that leads to {defect}.",
    "How does the {defect} defect relate to the etching process?",
    "What production step needs improvement to reduce {defect}?",
    "Explain the PCB fabrication failure that creates {defect}.",
]


def generate_explanation_sample(defect_name: str, info: dict) -> dict:
    """Generate an explanation instruction sample."""
    template = random.choice(EXPLANATION_TEMPLATES)
    instruction = template.format(defect=defect_name.replace("_", " "))

    output = f"""The {info['name']} defect is a {info['category']} type PCB defect with {info['severity']} severity.

**Definition:**
{info['description']}

**Visual Characteristics:**
{chr(10).join(f"• {v}" for v in info.get('visual_characteristics', [])[:3])}

**Severity Assessment:**
{info['name']} is classified as {info['severity'].upper()} severity. {info.get('severity_rationale', '')}

**IPC Standard Reference:** {info.get('ipc_standard', 'See IPC-6012D')}"""

    return {
        "instruction": instruction,
        "input": f"Defect type: {info['name']}",
        "output": output,
    }


def generate_cause_sample(defect_name: str, info: dict) -> dict:
    """Generate a root cause analysis instruction sample."""
    template = random.choice(CAUSE_TEMPLATES)
    instruction = template.format(defect=defect_name.replace("_", " "))

    causes = info.get("causes", [])
    mfg = info.get("manufacturing_process", {})

    output = f"""**Manufacturing Stage:** {mfg.get('stage', 'Multiple stages')}

**Process Description:**
{mfg.get('process_description', 'Not specified')}

**Primary Causes of {info['name']}:**
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(causes))}

**Contributing Factors:**
{chr(10).join(f"• {f}" for f in mfg.get('contributing_factors', []))}

Understanding these causes enables targeted process improvements to eliminate {info['name']} defects."""

    return {
        "instruction": instruction,
        "input": f"Defect: {info['name']} | Code: {info.get('code', 'N/A')}",
        "output": output,
    }


def generate_severity_sample(defect_name: str, info: dict) -> dict:
    """Generate a severity assessment instruction sample."""
    template = random.choice(SEVERITY_TEMPLATES)
    instruction = template.format(defect=defect_name.replace("_", " "))

    severity = info["severity"].upper()
    severity_actions = {
        "CRITICAL": "Mandatory rejection. No rework acceptable. Immediate process hold required.",
        "HIGH": "Engineering disposition required. Rework evaluation needed before accept/reject.",
        "MEDIUM": "Risk-based disposition. Accept with documentation if within IPC limits.",
        "LOW": "Accept with notation. Monitor for frequency trends.",
    }

    output = f"""**Severity Level: {severity}**

{info.get('severity_rationale', '')}

**Required Action:** {severity_actions.get(severity, 'Engineering review required.')}

**Risk Assessment:**
{chr(10).join(f"• {r}" for r in info.get('potential_risks', []))}

**IPC Classification:** {info.get('ipc_standard', 'Refer to IPC-6012D')}

{"⚠️ CRITICAL DEFECT: This defect results in immediate board failure and must never be accepted." if severity == "CRITICAL" else ""}"""

    return {
        "instruction": instruction,
        "input": f"Defect: {info['name']} | Severity: {info['severity']}",
        "output": output,
    }


def generate_risk_sample(defect_name: str, info: dict) -> dict:
    """Generate a risk analysis instruction sample."""
    template = random.choice(RISK_TEMPLATES)
    instruction = template.format(defect=defect_name.replace("_", " "))

    risks = info.get("potential_risks", [])

    output = f"""**Risk Analysis for {info['name']}:**

**Immediate Risks:**
{chr(10).join(f"• {r}" for r in risks[:3])}

**Long-term Reliability Concerns:**
{chr(10).join(f"• {r}" for r in risks[3:])}

**Overall Risk Rating:** {info['severity'].upper()}

{info.get('severity_rationale', '')}

**Recommended Action:**
Boards with {info['name']} defects should undergo {
"immediate rejection and root cause investigation" if info['severity'] in ['critical', 'high'] 
else "engineering disposition review and risk assessment"
}."""

    return {
        "instruction": instruction,
        "input": f"Detected defect: {info['name']}",
        "output": output,
    }


def generate_repair_sample(defect_name: str, info: dict) -> dict:
    """Generate a repair/corrective action instruction sample."""
    template = random.choice(REPAIR_TEMPLATES)
    instruction = template.format(defect=defect_name.replace("_", " "))

    recs = info.get("repair_recommendations", [])

    output = f"""**Corrective Actions for {info['name']}:**

**Immediate Board-Level Actions:**
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(recs[:3]))}

**Process-Level Corrections:**
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(recs[3:]))}

**Inspection Procedure:**
{chr(10).join(f"• {p}" for p in info.get('inspection_procedure', [])[:4])}

**Prevention:** Address the root cause in the manufacturing process to prevent recurrence. 
Implement SPC (Statistical Process Control) monitoring at the {
info.get('manufacturing_process', {}).get('stage', 'relevant manufacturing')
} stage."""

    return {
        "instruction": instruction,
        "input": f"PCB defect requiring corrective action: {info['name']}",
        "output": output,
    }


def generate_report_sample(
    defect_name: str, info: dict, confidence: float = None
) -> dict:
    """Generate a full inspection report instruction sample."""
    if confidence is None:
        confidence = round(random.uniform(0.70, 0.99), 3)
    conf_pct = round(confidence * 100, 1)

    template = random.choice(REPORT_TEMPLATES)
    instruction = template.format(
        defect=defect_name.replace("_", " "),
        confidence=conf_pct,
    )

    causes_top = info.get("causes", [])[:2]
    risks_top = info.get("potential_risks", [])[:2]
    recs_top = info.get("repair_recommendations", [])[:3]

    output = f"""# PCB DEFECT INSPECTION REPORT

## Detection Summary
| Field | Value |
|-------|-------|
| Defect Type | {info['name']} |
| Defect Code | {info.get('code', 'N/A')} |
| Detection Confidence | {conf_pct}% |
| Severity Level | **{info['severity'].upper()}** |
| Category | {info.get('category', 'N/A').title()} |

## Technical Description
{info['description']}

## Root Cause Analysis
**Manufacturing Stage:** {info.get('manufacturing_process', {}).get('stage', 'N/A')}

Most probable causes:
{chr(10).join(f"• {c}" for c in causes_top)}

## Risk Assessment
{chr(10).join(f"• {r}" for r in risks_top)}

## Disposition
{"🔴 **REJECT** — This defect results in functional failure. Board must be rejected." if info['severity'] == 'critical' else 
 "🟠 **ENGINEERING REVIEW REQUIRED** — High severity defect requires disposition." if info['severity'] == 'high' else
 "🟡 **CONDITIONAL ACCEPT** — Review against IPC specification limits." if info['severity'] == 'medium' else
 "🟢 **ACCEPT WITH DOCUMENTATION** — Monitor for recurrence trends."}

## Corrective Actions
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(recs_top))}

## Standards Reference
{info.get('ipc_standard', 'IPC-6012D')}

---
*Report generated by PCB-VLM-XAI Automated Inspection System*"""

    return {
        "instruction": instruction,
        "input": (
            f"Detected: {info['name']} | "
            f"Confidence: {conf_pct}% | "
            f"Severity: {info['severity']}"
        ),
        "output": output,
    }


def generate_comparison_sample(
    defect1_name: str, info1: dict,
    defect2_name: str, info2: dict,
) -> dict:
    """Generate a comparison instruction sample between two defects."""
    template = random.choice(COMPARISON_TEMPLATES)
    instruction = template.format(
        defect1=defect1_name.replace("_", " "),
        defect2=defect2_name.replace("_", " "),
    )

    output = f"""**Comparison: {info1['name']} vs {info2['name']}**

| Aspect | {info1['name']} | {info2['name']} |
|--------|{'—'*len(info1['name'])}|{'—'*len(info2['name'])}|
| Severity | {info1['severity'].upper()} | {info2['severity'].upper()} |
| Category | {info1.get('category','N/A').title()} | {info2.get('category','N/A').title()} |
| IPC Standard | {info1.get('ipc_standard','N/A')} | {info2.get('ipc_standard','N/A')} |

**{info1['name']}:**
{info1['description'][:200]}...

**{info2['name']}:**
{info2['description'][:200]}...

**Key Difference:**
{info1['name']} primarily affects {info1.get('category','the PCB')} with {info1['severity']} severity, 
while {info2['name']} is a {info2.get('category','PCB')} issue with {info2['severity']} severity. 
{"Both are critical defects requiring immediate rejection." if info1['severity'] == 'critical' and info2['severity'] == 'critical' else
 f"{info1['name']} is {'more' if info1['severity'] in ['critical','high'] else 'less'} severe in most applications."}"""

    return {
        "instruction": instruction,
        "input": f"Compare: {info1['name']} and {info2['name']}",
        "output": output,
    }


def generate_manufacturing_sample(defect_name: str, info: dict) -> dict:
    """Generate a manufacturing process explanation sample."""
    template = random.choice(MANUFACTURING_TEMPLATES)
    instruction = template.format(defect=defect_name.replace("_", " "))

    mfg = info.get("manufacturing_process", {})

    output = f"""**Manufacturing Analysis: {info['name']}**

**Process Stage:** {mfg.get('stage', 'Multiple stages')}

**Process Description:**
{mfg.get('process_description', 'The defect occurs during PCB fabrication.')}

**Contributing Process Factors:**
{chr(10).join(f"• {f}" for f in mfg.get('contributing_factors', []))}

**Process Improvement Recommendations:**
To eliminate {info['name']} at the source:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(info.get('repair_recommendations', [])[-3:]))}

Implementing Statistical Process Control (SPC) at the {mfg.get('stage', 'relevant')} stage 
will help detect process drift before defects reach unacceptable levels."""

    return {
        "instruction": instruction,
        "input": f"Defect: {info['name']} | Stage: {mfg.get('stage', 'N/A')}",
        "output": output,
    }


def generate_dataset(
    knowledge_path: str = "knowledge/defect_knowledge.json",
    output_path: str = "data/fine_tuning/pcb_instructions.json",
    n_samples: int = 500,
    seed: int = 42,
) -> list[dict]:
    """
    Generate the complete synthetic instruction dataset.

    Args:
        knowledge_path: Path to knowledge base JSON.
        output_path: Where to save the generated dataset.
        n_samples: Target number of instruction samples.
        seed: Random seed for reproducibility.

    Returns:
        List of instruction sample dicts.
    """
    random.seed(seed)

    console.rule("[bold blue]Synthetic Instruction Dataset Generation")

    engine = KnowledgeEngine(knowledge_path)
    defects = engine.knowledge["defects"]
    defect_names = list(defects.keys())

    all_samples = []

    # --- Generate samples for each defect type ---
    generators = [
        generate_explanation_sample,
        generate_cause_sample,
        generate_severity_sample,
        generate_risk_sample,
        generate_repair_sample,
        generate_manufacturing_sample,
    ]

    # Base samples: all generators × all defects
    for defect_name, info in defects.items():
        for generator in generators:
            sample = generator(defect_name, info)
            all_samples.append(sample)

        # Multiple report samples with different confidence levels
        for conf in [0.72, 0.83, 0.91, 0.96, 0.99]:
            sample = generate_report_sample(defect_name, info, confidence=conf)
            all_samples.append(sample)

    # Comparison samples: all pairs
    for i, (name1, info1) in enumerate(defects.items()):
        for name2, info2 in list(defects.items())[i+1:]:
            sample = generate_comparison_sample(name1, info1, name2, info2)
            all_samples.append(sample)

    console.print(f"Base samples generated: {len(all_samples)}")

    # --- Augment by paraphrasing instruction variety ---
    # Add more report samples with random confidence values
    while len(all_samples) < n_samples:
        defect_name = random.choice(defect_names)
        info = defects[defect_name]
        gen = random.choice(generators + [generate_report_sample])
        
        if gen == generate_report_sample:
            sample = generate_report_sample(defect_name, info)
        else:
            sample = gen(defect_name, info)
        
        all_samples.append(sample)

    # Shuffle for training
    random.shuffle(all_samples)
    all_samples = all_samples[:n_samples]

    # --- Save dataset ---
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=True)

    console.print(f"\n[bold green]Dataset saved: {output_file}[/bold green]")
    console.print(f"Total samples: {len(all_samples)}")

    # Stats
    instruction_lengths = [len(s["output"].split()) for s in all_samples]
    console.print(f"Avg output words: {sum(instruction_lengths)/len(instruction_lengths):.0f}")
    console.print(f"Max output words: {max(instruction_lengths)}")

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PCB Instruction Dataset")
    parser.add_argument("--knowledge", default="knowledge/defect_knowledge.json")
    parser.add_argument("--output", default="data/fine_tuning/pcb_instructions.json")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        knowledge_path=args.knowledge,
        output_path=args.output,
        n_samples=args.samples,
        seed=args.seed,
    )
