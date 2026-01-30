# GPU Environment Setup for Linux

## Overview
This document describes the GPU-enabled Python environment setup for Linux.

## What's Installed

### Virtual Environment
- **Name**: `venv_gpu_linux`
- **Location**: `/media/black-lights/407A01527A01465E/dev/image_analysis_v2/venv_gpu_linux`
- **Python Version**: Python 3.12

### Installed Packages

#### Core Dependencies
- `numpy` (2.4.1) - Array processing
- `scikit-learn` (1.8.0) - Machine learning library ✓ **Added**
- `scipy` (1.17.0) - Scientific computing
- `pandas` (3.0.0) - Data manipulation

#### Computer Vision
- `opencv-contrib-python` (4.13.0.90) - OpenCV with contrib modules
  - ⚠️ **Note**: Standard pip version does NOT include CUDA support
  - For GPU acceleration, you need to build from source (see below)

#### Visualization
- `matplotlib` (3.10.8) - Plotting library

#### Utilities
- `tqdm` (4.67.1) - Progress bars

#### Development Tools
- `pytest` (9.0.2) - Testing framework
- `black` (26.1.0) - Code formatter
- `flake8` (7.3.0) - Linter

## Activation

To activate this environment:
```bash
source venv_gpu_linux/bin/activate
```

To deactivate:
```bash
deactivate
```

## OpenCV GPU Support

### Current Status
The currently installed OpenCV from pip does **NOT** include CUDA support. This is a limitation of the pre-built pip packages.

### How to Get GPU-Accelerated OpenCV

#### Option 1: Use the Automated Build Script (Recommended)
```bash
# Make sure you have CUDA toolkit installed first
# Then run:
./install_opencv_gpu.sh
```

This script will:
1. Check for CUDA toolkit
2. Install build dependencies
3. Download OpenCV source code
4. Configure with CUDA support
5. Build and install into your virtual environment

**Prerequisites**:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (11.0 or later recommended)
- At least 10GB free disk space
- 30-60 minutes for compilation

#### Option 2: Manual Build
See the detailed instructions in `docs/GPU_ACCELERATION_GUIDE.md`

### Verifying GPU Support

After building OpenCV with CUDA:
```python
import cv2
print("OpenCV version:", cv2.__version__)
print("CUDA enabled:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

# Check available CUDA devices
if hasattr(cv2, 'cuda'):
    print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
```

## Requirements Files

### `requirements_gpu_linux.txt`
Contains all Python package requirements for the GPU environment. Use this for fresh installations:
```bash
pip install -r requirements_gpu_linux.txt
```

### `requirements.txt`
Original requirements file (updated with notes about GPU support)

## Quick Start

### Fresh Setup
```bash
# Run the setup script
./setup_gpu_env_linux.sh

# Activate the environment
source venv_gpu_linux/bin/activate

# Verify installation
python -c "import cv2, sklearn, numpy, pandas; print('All packages OK!')"
```

### For GPU-Accelerated OpenCV
```bash
# Activate environment
source venv_gpu_linux/bin/activate

# Run the OpenCV GPU build script
./install_opencv_gpu.sh

# This will take 30-60 minutes
# After completion, verify CUDA support
python -c "import cv2; print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## Troubleshooting

### "CUDA toolkit not found"
Install CUDA toolkit from NVIDIA:
https://developer.nvidia.com/cuda-downloads

### "No module named 'sklearn'"
The environment should have scikit-learn installed. Verify with:
```bash
source venv_gpu_linux/bin/activate
pip list | grep scikit-learn
```

### OpenCV Import Errors
If you get import errors after building OpenCV from source:
1. Make sure you're in the correct virtual environment
2. Check that the build completed successfully
3. Verify the Python path matches your virtual environment

### Build Failures
Common issues:
- Insufficient disk space (need ~10GB)
- Missing build dependencies (run `sudo apt-get install build-essential cmake`)
- CUDA version mismatch (ensure compatible CUDA version)

## Notes

1. **scikit-learn**: Successfully added to the environment (version 1.8.0)
2. **OpenCV**: Currently using standard version; GPU version requires building from source
3. **CUDA**: Required for GPU acceleration; check compatibility with your GPU
4. **Environment isolation**: This environment is separate from `venv_gpu` (Windows)

## Next Steps

1. ✅ Virtual environment created
2. ✅ All packages installed (including scikit-learn)
3. ⏳ Build OpenCV with CUDA support (optional, for GPU acceleration)
4. ⏳ Test GPU acceleration with your lane detection code

## See Also
- `docs/GPU_ACCELERATION_GUIDE.md` - Detailed GPU setup guide
- `install_opencv_gpu.sh` - Automated OpenCV build script
- `setup_gpu_env_linux.sh` - Environment setup script
