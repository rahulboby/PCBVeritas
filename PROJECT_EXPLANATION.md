# PCBVeritas Project Explanation

## 1. What This Project Does

PCBVeritas is an explainable PCB defect inspection pipeline. It takes a PCB
image, detects manufacturing defects with YOLOv8s, explains the visual decision
with Grad-CAM or EigenCAM, retrieves visually similar historical defects with
SigLIP + FAISS, enriches the result with a PCB defect knowledge base, and sends
that RAG context to an OpenAI-compatible LLM endpoint to generate a markdown
inspection report.

The current project is RAG-only for LLM reporting. Fine-tuning, LoRA adapters,
and synthetic fine-tuning data are intentionally removed.

End-to-end flow:

```text
PCB image
  -> YOLOv8s detector
  -> detected class, confidence, bbox, crop
  -> optional Grad-CAM / EigenCAM explanation
  -> SigLIP crop embedding
  -> FAISS similar-case retrieval
  -> defect knowledge lookup
  -> RAG prompt + system prompt
  -> OpenAI-compatible LLM API
  -> markdown inspection report
```

## 2. Main Configuration

All project settings live in `configs/settings.py`.

Important dictionaries:

| Config | Purpose |
| --- | --- |
| `STREAMLIT_CONFIG` | App title, pages, upload rules, display settings |
| `INFERENCE_CONFIG` | YOLOv8s inference weights, confidence thresholds, classes |
| `TRAINING_CONFIG` | YOLOv8s training hyperparameters and dataset path |
| `RETRIEVAL_CONFIG` | SigLIP model, FAISS paths, crop settings, top-k retrieval |
| `XAI_CONFIG` | Grad-CAM/EigenCAM target layer and visualization settings |
| `LLM_CONFIG` | LLM provider, endpoint, model, API key env var, generation settings, system prompt |

YOLO still requires `data/splits/dataset.yaml`. That file is an Ultralytics
dataset manifest, not a project config file. Training and runtime settings are
read from `configs/settings.py`.

## 3. Dataset Sources

Two PCB defect datasets were used.

### Dataset 1: PKU PCB_DATASET

Expected location:

```text
data/raw/PCB_DATASET/
  images/
  Annotations/
  rotation/      optional/raw augmentation folder
  PCB_USED/      optional/source folder
```

The annotation format is Pascal VOC XML. Each XML file stores object class names
and absolute bounding boxes:

```text
xmin, ymin, xmax, ymax
```

The project converts these boxes to YOLO format:

```text
class_id center_x center_y width height
```

All YOLO coordinates are normalized to the image width and height.

Canonical class order:

| ID | Class |
| --- | --- |
| 0 | `missing_hole` |
| 1 | `mouse_bite` |
| 2 | `open_circuit` |
| 3 | `short` |
| 4 | `spur` |
| 5 | `spurious_copper` |

### Dataset 2: pcb-defect-dataset

Expected location:

```text
data/raw/pcb-defect-dataset/
  data.yaml
  train/
  val/
  test/
```

This dataset is already in YOLO format. Its class order can differ from the
canonical order, so the merge step remaps class IDs by class name.

Current secondary dataset class order from `data/raw/pcb-defect-dataset/data.yaml`:

| Source ID | Class |
| --- | --- |
| 0 | `mouse_bite` |
| 1 | `spur` |
| 2 | `missing_hole` |
| 3 | `short` |
| 4 | `open_circuit` |
| 5 | `spurious_copper` |

## 4. Dataset Preparation Walkthrough

### Step 1: Create and activate the environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip>=24.0" "setuptools>=68.2.2" "wheel>=0.42.0"
python -m pip install -r requirements.txt
```

Linux/macOS/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=24.0" "setuptools>=68.2.2" "wheel>=0.42.0"
python -m pip install -r requirements.txt
```

`requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cu118`
so PyTorch CUDA wheels can be found while normal packages such as `openai` still
come from PyPI.

### Step 2: Convert the PKU XML dataset to YOLO

The conversion script is:

```text
data/raw/prepare_pcb_dataset_split.py
```

Run:

PowerShell:

```powershell
python data/raw/prepare_pcb_dataset_split.py --raw-dir data/raw/PCB_DATASET --output-dir data/raw/PCB_DATASET_SPLIT
```

Bash:

```bash
python data/raw/prepare_pcb_dataset_split.py \
  --raw-dir data/raw/PCB_DATASET \
  --output-dir data/raw/PCB_DATASET_SPLIT
```

What this script does:

1. Reads images from `data/raw/PCB_DATASET/images`.
2. Reads XML annotations from `data/raw/PCB_DATASET/Annotations`.
3. Ignores `rotation/` and `PCB_USED/` so augmentation is not baked into the validation/test data.
4. Normalizes class names into the canonical six-class schema.
5. Converts each Pascal VOC bbox from `[xmin, ymin, xmax, ymax]` to YOLO `[cx, cy, w, h]`.
6. Splits data into train/val/test with an 80/10/10 default.
7. Writes:

```text
data/raw/PCB_DATASET_SPLIT/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
  dataset.yaml
  split_metadata.json
```

### Step 3: Merge the converted PKU dataset with the native YOLO dataset

The merge script is:

```text
data/raw/merge_pcb_yolo_datasets.py
```

Run:

PowerShell:

```powershell
python data/raw/merge_pcb_yolo_datasets.py --primary-yaml data/raw/PCB_DATASET_SPLIT/dataset.yaml --secondary-yaml data/raw/pcb-defect-dataset/data.yaml --output-dir data/splits
```

Bash:

```bash
python data/raw/merge_pcb_yolo_datasets.py \
  --primary-yaml data/raw/PCB_DATASET_SPLIT/dataset.yaml \
  --secondary-yaml data/raw/pcb-defect-dataset/data.yaml \
  --output-dir data/splits
```

What this script does:

1. Reads the canonical class names from the primary dataset YAML.
2. Reads the native YOLO dataset names from the secondary dataset YAML.
3. Builds a secondary-to-primary class ID remap by class name.
4. Copies train/val/test images and labels from both datasets.
5. Renames collisions safely by adding source tags.
6. Writes the final Ultralytics dataset manifest:

```text
data/splits/dataset.yaml
```

7. Writes merge details:

```text
data/splits/merge_metadata.json
```

The final training data path used by YOLO is:

```text
data/splits/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
  dataset.yaml
```

## 5. Detector Training

The detector is YOLOv8s from Ultralytics.

Training entrypoint:

```text
detector/train.py
```

Command:

```bash
python detector/train.py
```

Resume interrupted training:

```bash
python detector/train.py --resume
```

Validate a trained checkpoint:

```bash
python detector/train.py --validate-only --weights models/detector/best.pt
```

How configuration is loaded:

- `detector/train.py` imports `TRAINING_CONFIG` from `configs/settings.py`.
- `TRAINING_CONFIG["dataset"]["path"]` points to `data/splits/dataset.yaml`.
- The script refreshes the absolute path inside `dataset.yaml` if the project folder was renamed or moved.
- Best weights are copied to:

```text
models/detector/best.pt
```

Training outputs:

```text
runs/detect/pcb_defect_detector/
  weights/best.pt
  weights/last.pt
  results.csv
  confusion_matrix.png
  PR_curve.png
```

## 6. Single-Image Detector Inference

Entrypoint:

```text
detector/inference.py
```

Command:

```bash
python detector/inference.py --image path/to/pcb.jpg --save outputs/detections/
```

The detector returns:

- `class_name`
- `class_id`
- `confidence`
- absolute bbox `[x1, y1, x2, y2]`
- normalized bbox `[cx, cy, w, h]`
- optional defect crop image

## 7. Retrieval Index Build

Retrieval uses SigLIP image embeddings and FAISS vector search.

Entrypoint:

```text
retrieval/build_index.py
```

Command:

```bash
python retrieval/build_index.py
```

Force a full crop/index rebuild:

```bash
python retrieval/build_index.py --force-rebuild
```

What happens:

1. Reads YOLO labels from `data/splits/train/labels`.
2. Reads matching images from `data/splits/train/images`.
3. Converts YOLO normalized bboxes back to pixel crop coordinates.
4. Adds padding from `RETRIEVAL_CONFIG["crops"]["padding"]`.
5. Saves crops under `data/processed/crops`.
6. Embeds each crop with SigLIP (`google/siglip-base-patch16-224`).
7. L2-normalizes embeddings.
8. Builds a FAISS `IndexFlatIP`, which acts like cosine similarity for normalized vectors.
9. Writes:

```text
data/embeddings/faiss_index.bin
data/embeddings/metadata.json
```

Runtime retrieval:

1. The pipeline crops each newly detected defect.
2. `retrieval/embedder.py` embeds the crop.
3. `retrieval/faiss_search.py` searches for the top-k nearest historical cases.
4. Returned cases include labels, similarity scores, source paths, crop paths, and rank.

## 8. Knowledge Base

Knowledge file:

```text
knowledge/defect_knowledge.json
```

Runtime API:

```text
knowledge/knowledge_engine.py
```

The knowledge engine provides:

- defect description
- severity
- manufacturing process stage
- likely causes
- potential risks
- inspection procedure
- repair recommendations
- IPC standard reference

For report generation, `KnowledgeEngine.format_for_rag_prompt(...)` converts this
structured JSON into prompt-ready text.

## 9. RAG Report Generation

Report generator:

```text
llm/inference/report_generator.py
```

The project no longer loads a local HuggingFace LLM inside Python. Instead,
`PCBReportGenerator` initializes an OpenAI-compatible client using `LLM_CONFIG`.

Default provider:

```python
LLM_CONFIG["provider"] = "lm_studio"
```

Default LM Studio endpoint:

```text
http://localhost:1234/v1
```

Grok/xAI endpoint:

```text
https://api.x.ai/v1
```

To switch providers, edit only `configs/settings.py`:

```python
LLM_CONFIG["provider"] = "grok"
LLM_CONFIG["providers"]["grok"]["model"] = "grok-4.5"
```

The LLM prompt contains:

1. A system prompt from `LLM_CONFIG["system_prompt"]`.
2. Detection chunks: class, confidence, bbox, severity.
3. Knowledge-base chunks from `KnowledgeEngine`.
4. Retrieved similar-case chunks from FAISS metadata.
5. Report requirements telling the LLM to write a grounded markdown report.

The API call uses:

```python
client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=...,
    temperature=...,
    top_p=...,
)
```

## 10. Secret Handling

Never store real API keys in source files.

Local setup:

```bash
copy .env.example .env
```

or on Bash:

```bash
cp .env.example .env
```

`.env.example` contains placeholders only:

```text
XAI_API_KEY=
LM_STUDIO_API_KEY=lm-studio
```

`.env` is ignored by Git through `.gitignore`.

Deployment:

- Put `XAI_API_KEY` or `LM_STUDIO_API_KEY` in the deployment platform's secret manager.
- Keep `configs/settings.py` as the single place that names which env var to read.
- For LM Studio, the API key is usually just a placeholder because the local server does not require a real hosted key.

## 11. Full Pipeline Runtime

Orchestrator:

```text
pipeline/orchestrator.py
```

The main runtime class is:

```python
PCBInspectionPipeline
```

Stages:

1. Detection: `detector/detector.py`
2. XAI: `xai/grad_cam.py` and `xai/visualizer.py`
3. Retrieval: `retrieval/embedder.py` and `retrieval/faiss_search.py`
4. Knowledge: `knowledge/knowledge_engine.py`
5. Report: `llm/inference/report_generator.py`

The final object is `PipelineResult`, which contains:

- original image
- detections
- detection visualization
- heatmap and overlay
- retrieved similar cases
- knowledge contexts
- markdown report
- stage timings
- success/error status

If the LLM API is unavailable, the pipeline falls back to a template report so
detection, XAI, and retrieval can still run.

## 12. Streamlit App

Entrypoint:

```text
app/app.py
```

Command:

```bash
streamlit run app/app.py
```

Pages:

| Page | Purpose |
| --- | --- |
| Upload & Detect | Upload a PCB image and run inspection |
| XAI Visualizations | View Grad-CAM heatmaps and overlays |
| Similar Defects | View FAISS retrieved historical cases |
| Knowledge Insights | Read defect causes, risks, and repairs |
| Inspection Report | View/download the RAG-generated markdown report |

## 13. Recommended First-Time Run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip>=24.0" "setuptools>=68.2.2" "wheel>=0.42.0"
python -m pip install -r requirements.txt
copy .env.example .env
```

Convert and merge datasets:

```bash
python data/raw/prepare_pcb_dataset_split.py --raw-dir data/raw/PCB_DATASET --output-dir data/raw/PCB_DATASET_SPLIT
python data/raw/merge_pcb_yolo_datasets.py --primary-yaml data/raw/PCB_DATASET_SPLIT/dataset.yaml --secondary-yaml data/raw/pcb-defect-dataset/data.yaml --output-dir data/splits
```

Train and index:

```bash
python detector/train.py
python retrieval/build_index.py
```

Run app:

```bash
streamlit run app/app.py
```

## 14. Testing

Focused test command:

```bash
python -m pytest tests/test_knowledge.py tests/test_pipeline.py tests/test_retrieval.py tests/test_detector.py tests/test_report_generator.py
```

Report-generator tests mock the OpenAI client and do not call the network.

## 15. Troubleshooting

### `openai>=1.40.0` cannot be found

Make sure `requirements.txt` uses `--extra-index-url`, not `--index-url`, for
the PyTorch CUDA wheel source. `--index-url` replaces PyPI and prevents pip from
finding normal packages such as `openai`.

### LM Studio report generation fails

1. Start LM Studio.
2. Load a chat/instruct model.
3. Start the local server.
4. Confirm the base URL in `settings.py` is `http://localhost:1234/v1`.
5. Keep `LM_STUDIO_API_KEY=lm-studio` in `.env`.

### Grok report generation fails

1. Set `LLM_CONFIG["provider"] = "grok"` in `configs/settings.py`.
2. Put `XAI_API_KEY=...` in `.env` or your deployment secret manager.
3. Confirm the model name in `settings.py` is available to your xAI account.

### FAISS index missing

Run:

```bash
python retrieval/build_index.py
```

Expected files:

```text
data/embeddings/faiss_index.bin
data/embeddings/metadata.json
```

### Detector weights missing

Train the detector or place trained weights at:

```text
models/detector/best.pt
```

## 16. What Was Removed

The project no longer includes:

- `llm/fine_tuning/`
- `data/fine_tuning/pcb_instructions.json`
- local Qwen model loading in `report_generator.py`
- PEFT/LoRA dependencies
- fine-tuning tests
- `FINE_TUNING_CONFIG`

The active LLM path is RAG through an OpenAI-compatible API.
