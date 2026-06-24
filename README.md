# PCBVeritas 

## Explainable Vision-Language PCB Inspection System with Retrieval-Augmented Defect Reasoning

[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-ff4b4b.svg)](https://streamlit.io/)

A production-quality AI system for automated PCB (Printed Circuit Board) defect inspection combining **YOLOv8 object detection**, **Grad-CAM explainability**, **SigLIP + FAISS retrieval**, and **LoRA fine-tuned Qwen2.5** for expert report generation — all running **fully offline** on consumer hardware.

---

## Project Objectives

* Detect PCB manufacturing defects with high accuracy.
* Localize defects using bounding boxes.
* Provide visual explanations for model predictions.
* Retrieve visually similar historical defect cases.
* Generate grounded inspection reports using retrieved evidence and domain knowledge.
* Explore the integration of object detection, retrieval systems, and language models in an industrial AI workflow.

---

## System Architecture

```text
PCB Image
    │
    ▼
YOLOv8s Detector
    │
    ├── Bounding Boxes
    ├── Class Labels
    ├── Confidence Scores
    │
    ├─────────────┐
    ▼             ▼
Grad-CAM       SigLIP
    │             │
    ▼             ▼
Heatmaps      FAISS Retrieval
                   │
                   ▼
            Similar Cases
                   │
                   ▼
           Knowledge Base
                   │
                   ▼
         LoRA-Tuned Qwen2.5
                   │
                   ▼
          Inspection Report
```

---

## Defect Classes

The detector identifies six PCB defect categories:

| Class           | Description                         |
| --------------- | ----------------------------------- |
| Missing Hole    | Missing drilled via or through-hole |
| Mouse Bite      | Irregular conductor edge damage     |
| Open Circuit    | Broken electrical connection        |
| Short Circuit   | Unintended conductive bridge        |
| Spur            | Small copper protrusion             |
| Spurious Copper | Unwanted copper deposit             |

---

## Dataset Construction

Two datasets were used during development.

### Dataset 1: PCB_DATASET

https://www.kaggle.com/datasets/akhatova/pcb-defects

* Pascal VOC XML annotations
* Images stored in class-specific folders
* Original rotation-based augmentations were intentionally ignored
* Converted into YOLO format using:

```text
data/raw/prepare_pcb_dataset_split.py
```

This script:

* Parses Pascal VOC XML annotations
* Converts annotations to YOLO format
* Creates train/validation/test splits
* Generates:

```text
data/raw/PCB_DATASET_SPLIT
```

### Dataset 2: pcb-defect-dataset

https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset

* Native YOLO-format dataset
* Used directly without annotation conversion

### Dataset Merge

The two datasets were merged using:

```text
data/raw/merge_pcb_yolo_datasets.py
```

This script:

* Merges both datasets
* Automatically remaps class IDs using class names
* Preserves a unified label schema
* Generates:

```text
data/splits
```

Final dataset statistics:

| Split      | Images |
| ---------- | ------ |
| Train      | 9,088  |
| Validation | 1,135  |
| Test       | 1,138  |

The merged dataset contains approximately 11,361 PCB images.

---

## Object Detection

The object detection component uses YOLOv8s.

Training configuration:

* Framework: Ultralytics YOLOv8
* Model: YOLOv8s
* Epochs: 100
* Dataset: Merged PCB dataset
* Input format: Bounding-box detection

Training runtime on RTX 4050:

* Average epoch time: ~2 minutes 40 seconds
* Total training time: ~4.5 hours

YOLOv8 built-in augmentation was used during training. Pre-generated rotation images from the original dataset were excluded.

---

## Explainability

To improve transparency, Grad-CAM visualizations are generated for detected defects.

The explainability module highlights image regions that contributed most strongly to detector predictions and provides a visual inspection aid for model validation.

---

## Retrieval-Augmented Defect Reasoning

Visual retrieval is performed using:

* SigLIP image embeddings
* FAISS similarity search

For every detected defect:

1. SigLIP generates an image embedding.
2. FAISS retrieves similar historical examples.
3. Retrieved examples are passed into the report-generation pipeline.

This grounds generated reports using previously observed defect cases.

---

## Report Generation

Inspection reports are generated using:

* Qwen2.5-1.5B-Instruct
* LoRA fine-tuning
* Structured defect knowledge base

The language model does not perform defect detection.

Its role is to synthesize:

* Detector outputs
* Retrieved defect examples
* Domain knowledge

into a human-readable inspection report.

---

## Technology Stack

| Component         | Technology   |
| ----------------- | ------------ |
| Object Detection  | YOLOv8s      |
| Explainability    | Grad-CAM     |
| Visual Embeddings | SigLIP       |
| Retrieval         | FAISS        |
| Language Model    | Qwen2.5-1.5B |
| Fine-Tuning       | LoRA (PEFT)  |
| Framework         | PyTorch      |
| Interface         | Streamlit    |

---

## Repository Structure

```text
PCBVeritas/
├── detector/
├── retrieval/
├── xai/
├── knowledge/
├── llm/
├── pipeline/
├── app/
├── data/
├── configs/
├── docs/
└── tests/
```

Detailed implementation notes, setup instructions, and training procedures are documented in the `docs/` directory.
