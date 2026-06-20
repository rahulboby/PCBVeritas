# PCBVeritas 🔬

## Explainable Vision-Language PCB Inspection System with Retrieval-Augmented Defect Reasoning

[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-ff4b4b.svg)](https://streamlit.io/)

A production-quality AI system for automated PCB (Printed Circuit Board) defect inspection combining **YOLOv8 object detection**, **Grad-CAM explainability**, **SigLIP + FAISS retrieval**, and **LoRA fine-tuned Qwen2.5** for expert report generation — all running **fully offline** on consumer hardware.

---

## 📸 System Overview

```
PCB Image
    │
    ▼
YOLOv8s Detection ──────────────┐
    │                            │
    ├── Bounding Boxes           │
    ├── Class Labels             │  Detection
    └── Confidence Scores        │
    │                            │
    ├──────────────┐             │
    ▼              ▼             │
Grad-CAM        SigLIP          │
    │              │             │
    ▼              ▼             │
Heatmaps      FAISS Search ─────┤
                   │             │
                   ▼             │
             Similar Cases       │
                   │             │
                   ▼             │
          Knowledge Base ────────┤
                   │             │
                   ▼             │
        LoRA Fine-Tuned Qwen ────┘
                   │
                   ▼
        Inspection Report
```

---

## ✨ Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| **Defect Detection** | YOLOv8s | Localize and classify 6 PCB defect types |
| **Explainability** | Grad-CAM + EigenCAM | Heatmaps showing model attention regions |
| **Visual Retrieval** | SigLIP + FAISS | Find similar historical defects (top-3) |
| **Expert Reports** | Qwen2.5-1.5B + LoRA | RAG-powered inspection report generation |
| **Knowledge Base** | Structured JSON | Industrial defect knowledge (causes, risks, repair) |
| **Web Interface** | Streamlit | Polished multi-page inspection dashboard |
| **Offline Operation** | Local models | No API calls, no internet required |

---

## 🎯 Detected Defect Classes

| Class | Code | Severity | Description |
|-------|------|----------|-------------|
| Missing Hole | MH | 🟠 High | Absent drilled via or through-hole |
| Mouse Bite | MB | 🟡 Medium | Irregular notches on conductor edges |
| Open Circuit | OC | 🔴 Critical | Complete break in conductor path |
| Short Circuit | SC | 🔴 Critical | Unintended conductor bridge |
| Spur | SP | 🟢 Low | Small copper protrusion from trace |
| Spurious Copper | SCu | 🟡 Medium | Unintended copper deposits |

---

## 🏗️ Architecture

```
PCB-VLM-XAI/
├── detector/          # YOLOv8s detection pipeline
│   ├── detector.py    # Core inference class
│   ├── train.py       # Training script
│   └── inference.py   # Standalone inference
├── xai/               # Explainability module
│   ├── grad_cam.py    # Grad-CAM + EigenCAM
│   └── visualizer.py  # Multi-panel visualization
├── retrieval/         # Visual similarity search
│   ├── embedder.py    # SigLIP feature extraction
│   ├── faiss_search.py # Vector index search
│   └── build_index.py # Index construction
├── knowledge/         # Industrial defect knowledge
│   ├── knowledge_engine.py # Knowledge retrieval API
│   └── defect_knowledge.json # Expert knowledge base
├── llm/               # Language model
│   ├── fine_tuning/   # LoRA training pipeline
│   └── inference/     # Report generation
├── pipeline/          # End-to-end orchestration
│   └── orchestrator.py
├── app/               # Streamlit web interface
│   └── main.py
├── scripts/           # Dataset preparation scripts
├── configs/           # YAML configurations
├── tests/             # Test suite
└── docs/              # Technical documentation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA 11.8+ (recommended: RTX 4050+)
- 24 GB RAM recommended
- 20 GB disk space

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/PCB-VLM-XAI.git
cd PCB-VLM-XAI
```

### 2. Install Dependencies
```bash
bash install.sh
```

Or manually:
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade "pip>=24.0" "setuptools>=68.2.2" "wheel>=0.42.0"
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Download Dataset
Download the PCB Defect Dataset and extract to `data/raw/PCB_DATASET/`

> Dataset: [PCB Defect Dataset (PKUSZ Robotics Lab)](https://robotics.pkusz.edu.cn/resources/dataset/)

### 4. Prepare Data
```bash
python scripts/validate_dataset.py
python scripts/prepare_dataset.py
python scripts/analyze_dataset.py
```

### 5. Train Detector
```bash
python detector/train.py
```

### 6. Build Retrieval Index
```bash
python retrieval/build_index.py
```

### 7. Generate Fine-Tuning Data & Train LLM
```bash
python llm/fine_tuning/generate_dataset.py
python llm/fine_tuning/train_lora.py
```

### 8. Launch App
```bash
streamlit run app/main.py
```

---

## 📊 Expected Results

After training on the full PCB dataset:

| Metric | Expected Value |
|--------|---------------|
| mAP50 | > 0.90 |
| mAP50-95 | > 0.65 |
| Inference speed | ~15ms/image (GPU) |
| Report generation | ~8-15s/image |
| FAISS retrieval | <1ms |

---

## 🛠️ Configuration

All configurations are in `configs/`:

```bash
configs/
├── training.yaml       # YOLOv8 training hyperparameters
├── inference.yaml      # Detection thresholds & output settings
├── retrieval.yaml      # SigLIP & FAISS configuration
├── llm.yaml            # Qwen model & generation settings
├── fine_tuning.yaml    # LoRA training configuration
├── xai.yaml            # Grad-CAM settings
└── streamlit.yaml      # Web app configuration
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=html
```

---

## 📚 Documentation

| Document | Audience | Description |
|----------|----------|-------------|
| `README.md` | Everyone | Project overview (this file) |
| `INSTRUCTIONS.md` | Project owner | Complete operation guide |
| `PROJECT_MAP.md` | Developers | Module dependency & data flow |
| `docs/01_object_detection.md` | Learners | Object detection fundamentals |
| `docs/02_yolov8_architecture.md` | Learners | YOLOv8 deep dive |
| `docs/03_gradcam.md` | Learners | Grad-CAM mathematics |
| `docs/08_lora.md` | Learners | LoRA fine-tuning theory |
| `docs/09_retrieval_augmented_reasoning.md` | Learners | RAG explained |
| `docs/12_gpu_optimization.md` | Developers | VRAM optimization guide |

---

## 🔬 Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Object Detection | Ultralytics YOLOv8 | 8.2.18 |
| Deep Learning | PyTorch | 2.2.2 |
| Explainability | pytorch-grad-cam | 1.5.0 |
| Vision Embeddings | HuggingFace Transformers (SigLIP) | 4.40.2 |
| Vector Search | FAISS | 1.7.4 |
| LLM | Qwen2.5-1.5B-Instruct | via Transformers |
| Fine-Tuning | PEFT (LoRA) | 0.11.1 |
| Web Interface | Streamlit | 1.33.0 |
| Experiment Tracking | MLflow | 2.13.0 |

---

## 💡 Future Work

- [ ] Multi-GPU training support
- [ ] ONNX export for edge deployment
- [ ] Online learning from user feedback
- [ ] Support for additional PCB defect classes
- [ ] 3D PCB inspection with depth cameras
- [ ] Integration with manufacturing MES systems
- [ ] Real-time video stream inspection
- [ ] Confidence calibration module

---

## 📄 License

MIT License — See [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- PCB Dataset: [PKUSZ Robotics Laboratory](https://robotics.pkusz.edu.cn/)
- YOLOv8: [Ultralytics](https://ultralytics.com/)
- SigLIP: [Google Research](https://arxiv.org/abs/2303.15343)
- Qwen: [Alibaba Cloud](https://github.com/QwenLM/Qwen2.5)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
