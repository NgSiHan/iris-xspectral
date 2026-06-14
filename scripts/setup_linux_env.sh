#!/usr/bin/env bash
# Setup conda environment for iris-xspectral on Linux (RTX 4090).
# Run from the repo root:  bash scripts/setup_linux_env.sh

set -euo pipefail

ENV_NAME="iris-xspectral"
PYTHON_VERSION="3.10"

echo "=== Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch (CUDA 12.4) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing dependencies ==="
pip install \
    opencv-python-headless \
    numpy \
    Pillow \
    pyyaml \
    scikit-learn \
    pandas \
    tqdm \
    matplotlib \
    pyeer \
    einops

echo "=== Verifying ==="
python -c "
import torch
print(f'torch {torch.__version__}  CUDA {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Done ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Set env var:    export IRIS_ENV=linux"
echo "Verify:         python scripts/check_env.py"
