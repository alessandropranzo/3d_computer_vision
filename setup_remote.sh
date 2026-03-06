#!/usr/bin/env bash
#
# Setup CF-3DGS on a remote Linux server with an NVIDIA GPU.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA >= 11.6
#   - Anaconda or Miniconda installed
#   - Git with SSH access to GitHub
#
# Usage:
#   chmod +x setup_remote.sh
#   ./setup_remote.sh
#
# After setup, transfer your frames and run:
#   conda activate cf3dgs
#   cd CF-3DGS
#   python run_cf3dgs.py -s ./data/my_room/ --mode train --data_type custom

set -euo pipefail

ENV_NAME="cf3dgs"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.0.0"
TORCHVISION_VERSION="0.15.0"
CUDA_VERSION="11.7"

echo "=== CF-3DGS Remote Setup ==="
echo ""

# ── 1. Create conda environment ──────────────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating..."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate (works inside a script with conda init)
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Python: $(python --version)"
echo ""

# ── 2. Install CUDA toolkit and PyTorch ──────────────────────────────────────
echo "Installing CUDA toolkit and PyTorch..."
conda install -y conda-forge::cudatoolkit-dev="${CUDA_VERSION}.0"
conda install -y pytorch=="${PYTORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" \
    pytorch-cuda="${CUDA_VERSION}" -c pytorch -c nvidia

echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo ""

# ── 3. Clone CF-3DGS with submodules ────────────────────────────────────────
if [ -d "CF-3DGS" ]; then
    echo "CF-3DGS directory already exists. Pulling latest..."
    cd CF-3DGS
    git pull
    git submodule update --init --recursive
else
    echo "Cloning CF-3DGS..."
    git clone --recursive https://github.com/NVlabs/CF-3DGS.git
    cd CF-3DGS
fi

echo ""

# ── 4. Install Python dependencies ──────────────────────────────────────────
echo "Installing Python dependencies..."
pip install -r requirements.txt

# ── 5. Create data directory ─────────────────────────────────────────────────
mkdir -p data/my_room/images

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy your extracted frames to CF-3DGS/data/my_room/images/"
echo "     scp -r data/my_room/images/*.jpg user@this-server:$(pwd)/data/my_room/images/"
echo ""
echo "  2. Run training:"
echo "     conda activate ${ENV_NAME}"
echo "     python run_cf3dgs.py -s ./data/my_room/ --mode train --data_type custom"
echo ""
echo "  3. Results will be saved in:"
echo "     ./output/progressive/my_room/"
echo "       ├── chkpnt/ep00_init.pth   (Gaussian model checkpoint)"
echo "       ├── pose/ep00_init.pth     (estimated camera poses)"
echo "       ├── train/                 (training visualizations)"
echo "       └── eval/                  (evaluation renders)"
