# PCBVeritas

## Explainable PCB Inspection With Retrieval-Augmented Reports

PCBVeritas is an AI-assisted PCB inspection system that combines YOLOv8 object
detection, Grad-CAM explainability, SigLIP + FAISS visual retrieval, and a
RAG-based LLM report generator. The LLM is called through the OpenAI Python SDK
against either a local LM Studio server or the xAI/Grok API.

## Pipeline

```text
PCB image
  -> YOLOv8s detector
  -> Grad-CAM / EigenCAM visual explanation
  -> SigLIP crop embeddings
  -> FAISS similar-case retrieval
  -> PCB defect knowledge base
  -> OpenAI-compatible LLM API
  -> Markdown inspection report
```

## Defect Classes

| Class | Description |
| --- | --- |
| Missing Hole | Missing drilled via or through-hole |
| Mouse Bite | Irregular conductor edge damage |
| Open Circuit | Broken electrical connection |
| Short Circuit | Unintended conductive bridge |
| Spur | Small copper protrusion |
| Spurious Copper | Unwanted copper deposit |

## Configuration

All project settings are centralized in `configs/settings.py`.

- `TRAINING_CONFIG`: YOLOv8s detector training settings.
- `INFERENCE_CONFIG`: detector inference thresholds and output paths.
- `RETRIEVAL_CONFIG`: SigLIP, FAISS, crop, and retrieval settings.
- `XAI_CONFIG`: Grad-CAM/EigenCAM settings.
- `LLM_CONFIG`: provider, API endpoint, model name, secret env var, generation settings, and system prompt.

YOLO still uses `data/splits/dataset.yaml` as the Ultralytics dataset manifest;
that is dataset metadata, not a project config file.

## Installation and Deployment

This project uses a single requirements-based workflow. Streamlit Community Cloud
will install dependencies from `requirements.txt`, so there is no longer any need
for `environment.yml`.

### Local development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Local RTX 4050 note

The default requirements install CPU-friendly PyTorch wheels, which keep the
retrieval and embedding stack off the GPU. If you want GPU acceleration for the
YOLO detector locally, install the CUDA-enabled wheels manually after the base
requirements install:

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

### Streamlit Community Cloud

- Keep `requirements.txt` in the repository root.
- Do not add `environment.yml`.
- Set the LLM API key as a secret in the hosting platform (for example
  `GROQ_API_KEY` or `XAI_API_KEY`).

## LLM Setup

Copy `.env.example` to `.env` for local development and set the secret needed by
the provider selected in `LLM_CONFIG`.

```bash
XAI_API_KEY=
LM_STUDIO_API_KEY=lm-studio
```

`.env` is ignored by Git. For GitHub or cloud deployments, set the same
environment variable in the hosting platform's secret manager.

Default LLM provider:

```python
LLM_CONFIG["provider"] = "lm_studio"
```

To use Grok instead, change the provider and ensure `XAI_API_KEY` is available:

```python
LLM_CONFIG["provider"] = "grok"
```

## Common Commands

```bash
python scripts/prepare_dataset.py
python detector/train.py
python retrieval/build_index.py
streamlit run app/app.py
```

## Technology Stack

| Component | Technology |
| --- | --- |
| Object Detection | YOLOv8s |
| Explainability | Grad-CAM / EigenCAM |
| Visual Embeddings | SigLIP |
| Retrieval | FAISS |
| Report Generation | RAG + OpenAI-compatible LLM API |
| Interface | Streamlit |

## Notes

- Fine-tuning code and synthetic fine-tuning data are intentionally removed.
- RAG remains active: detector outputs, retrieved similar cases, and the PCB
  knowledge base are passed to the configured LLM API as prompt context.
- `retrieval/build_index.py` remains the index build entrypoint.
- YOLO inference is allowed to use the GPU locally when available, but the
  retrieval pipeline, SigLIP embeddings, and supporting components default to
  CPU so you keep as much VRAM as possible free for LM Studio.
