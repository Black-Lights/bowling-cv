#!/bin/bash
# Setup script for GPU-enabled Python environment on Linux
# This script creates a virtual environment and installs all required packages

set -e  # Exit on error

VENV_NAME="venv_gpu_linux"
PROJECT_DIR="/media/black-lights/407A01527A01465E/dev/image_analysis_v2"

echo "========================================="
echo "GPU Environment Setup for Linux"
echo "========================================="
echo ""

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA toolkit found: $(nvcc --version | grep release)"
else
    echo "⚠ WARNING: CUDA toolkit not found. GPU acceleration will not be available."
    echo "  Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment: $VENV_NAME"
cd "$PROJECT_DIR"
python3 -m venv "$VENV_NAME"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages
echo "Installing required packages..."
pip install numpy pandas scipy scikit-learn matplotlib tqdm pytest black flake8

# Install OpenCV (standard version - no GPU)
echo ""
echo "========================================="
echo "OpenCV Installation"
echo "========================================="
echo "Note: pip version of OpenCV does NOT include CUDA support"
echo "Installing standard OpenCV for now..."
pip install opencv-contrib-python

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "To activate this environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Installed packages:"
pip list | grep -E "(numpy|opencv|scipy|scikit-learn|pandas|matplotlib|tqdm)"
echo ""

# Check OpenCV CUDA support
echo "Checking OpenCV CUDA support..."
python -c "import cv2; print('OpenCV version:', cv2.__version__); print('CUDA module available:', hasattr(cv2, 'cuda')); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0)"
echo ""

echo "========================================="
echo "For GPU-Accelerated OpenCV:"
echo "========================================="
echo "The pip version doesn't include CUDA. To get GPU support:"
echo "1. Uninstall current OpenCV: pip uninstall opencv-contrib-python"
echo "2. Build OpenCV from source with CUDA enabled"
echo "3. See docs/GPU_ACCELERATION_GUIDE.md for detailed instructions"
echo ""
echo "OR use the install_opencv_gpu.sh script (coming soon)"
echo "========================================="
