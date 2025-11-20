#!/usr/bin/env bash
set -e

################################################################################
# 1. Create a Python venv environment
################################################################################
# You may replace python with any other system-installed Python version (python3).

python -m venv efi_env
source efi_env/bin/activate

# Upgrade pip
pip install --upgrade pip


################################################################################
# 2. Install PyTorch (GPU build)
################################################################################
# IMPORTANT:
# Always check the official PyTorch install page if CUDA version differs:
# https://pytorch.org/get-started/locally/

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118


################################################################################
# 3. Install required supporting packages
################################################################################
pip install \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    tqdm \
    cython \
    pandas \
    matplotlib \
    seaborn


################################################################################
# 4. Install PointNeXt (for PointNext model)
################################################################################
pip install pointnext==0.0.5


################################################################################
# 5. Final message
################################################################################
echo ""
echo "======================================="
echo " Environment installation complete!"
echo " Run:  source efi_env/bin/activate"
echo "======================================="
echo ""
