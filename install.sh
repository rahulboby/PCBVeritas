#!/usr/bin/env bash
# ============================================================
# PCB-VLM-XAI Installation Script
# Run: bash install.sh
# 
# NOTE: This script is for Linux/macOS/WSL only.
# Windows users: Use PowerShell instead (see INSTRUCTIONS.md)
# ============================================================

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  PCB-VLM-XAI Installation Script${NC}"
echo -e "${BLUE}  Explainable Vision-Language PCB Inspection System${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# --- Check Python ---
echo -e "${YELLOW}[1/7] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Found Python $python_version${NC}"

# --- Check CUDA ---
echo -e "${YELLOW}[2/7] Checking CUDA availability...${NC}"
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}Found CUDA $cuda_version${NC}"
else
    echo -e "${RED}WARNING: CUDA not found. Will use CPU mode.${NC}"
fi

# --- Create virtual environment ---
echo -e "${YELLOW}[3/7] Creating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# --- Activate venv ---
source .venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# --- Upgrade packaging tools ---
echo -e "${YELLOW}[4/7] Upgrading packaging tools...${NC}"
python3 -m pip install --upgrade "pip>=24.0" "setuptools>=68.2.2" "wheel>=0.42.0"

# --- Install project requirements ---
echo -e "${YELLOW}[5/7] Installing project requirements...${NC}"
python3 -m pip install -r requirements.txt

# --- Ensure headless OpenCV is the active runtime ---
echo -e "${YELLOW}[5b/7] Ensuring OpenCV headless is active...${NC}"
python3 -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
python3 -m pip install --force-reinstall --no-deps opencv-python-headless==4.13.0.92

# --- Optional local GPU note ---
echo -e "${YELLOW}[6/7] Optional: install CUDA-enabled PyTorch locally for RTX 4050...${NC}"
echo -e "${BLUE}If you want GPU acceleration for YOLO on a local NVIDIA card, run:${NC}"
echo -e "${BLUE}python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118${NC}"

# --- Create directory structure ---
echo -e "${YELLOW}[7/7] Creating data directories...${NC}"
mkdir -p data/{raw/PCB_DATASET,processed,splits,embeddings}
mkdir -p logs/{training,inference,errors}
mkdir -p models/{detector,embeddings,llm}
mkdir -p outputs/{detections,heatmaps,reports,retrieved}

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Download PCB dataset to: ${BLUE}data/raw/PCB_DATASET/${NC}"
echo -e "  2. Validate dataset:        ${BLUE}python scripts/validate_dataset.py${NC}"
echo -e "  3. Prepare dataset:         ${BLUE}python scripts/prepare_dataset.py${NC}"
echo -e "  4. Train detector:          ${BLUE}python detector/train.py${NC}"
echo -e "  5. Generate embeddings:     ${BLUE}python retrieval/build_index.py${NC}"
echo -e "  6. Configure LLM secrets:   ${BLUE}cp .env.example .env${NC}"
echo -e "  7. Launch app:              ${BLUE}streamlit run app/app.py${NC}"
echo ""
echo -e "See ${BLUE}INSTRUCTIONS.md${NC} for detailed step-by-step guide."
