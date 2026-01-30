# GPU Acceleration Guide

## Overview

The lane detection pipeline supports GPU acceleration for intensive operations:
- **HSV color space conversion** (preprocessing)
- **Sobel edge detection** (top boundary detection)

GPU acceleration can provide ~5-10x speedup for these operations.

## Current Status

✅ **GPU infrastructure implemented** with automatic CPU fallback  
⚠️ **CUDA not enabled** in standard OpenCV installation

## How GPU Acceleration Works

### 1. Automatic Detection
The system automatically detects CUDA availability:
```
[-] GPU Acceleration DISABLED: No CUDA-enabled devices found
```

### 2. Seamless Fallback
If GPU is unavailable, processing automatically falls back to CPU with no code changes needed.

### 3. Performance Tracking
When GPU is enabled, performance metrics are displayed after processing.

## Enabling GPU Acceleration

### Option 1: Use Pre-built CUDA OpenCV (Recommended)

**Not yet available for Python 3.13**. Standard `opencv-python` package does not include CUDA support.

### Option 2: Build OpenCV from Source with CUDA

This is complex but provides full GPU acceleration:

1. **Prerequisites**:
   - NVIDIA GPU (check: `nvidia-smi`)
   - CUDA Toolkit (13.1 or compatible)
   - Visual Studio Build Tools
   - CMake

2. **Build Steps** (Windows):
   ```powershell
   # Install CUDA Toolkit from NVIDIA
   # Download OpenCV source
   git clone https://github.com/opencv/opencv.git
   git clone https://github.com/opencv/opencv_contrib.git
   
   cd opencv
   mkdir build
   cd build
   
   # Configure with CUDA
   cmake -DWITH_CUDA=ON `
         -DCUDA_ARCH_BIN=8.6 `  # For RTX 3060 (Ampere)
         -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules `
         -DBUILD_opencv_python3=ON ..
   
   # Build (takes 1-2 hours)
   cmake --build . --config Release
   
   # Install
   cmake --install .
   ```

3. **Verify**:
   ```python
   import cv2
   print(cv2.cuda.getCudaEnabledDeviceCount())  # Should be > 0
   ```

### Option 3: Use Current CPU Implementation

The current implementation works perfectly on CPU:
- **Preprocessing**: ~2 fps (acceptable with frame caching)
- **Sobel detection**: ~52 fps
- **Total pipeline**: ~30 seconds per video

**Frame caching saves ~4 minutes** during development, making CPU processing viable.

## Configuration

```python
# config.py
USE_GPU = True  # Attempts GPU, falls back to CPU if unavailable
GPU_VERBOSE = True  # Print GPU status and performance info
```

## Performance Comparison

### Current (CPU):
- HSV conversion: ~2 fps
- Sobel detection: ~52 fps
- Total preprocessing: ~4 minutes (cached after first run)

### Expected (GPU):
- HSV conversion: ~20-50 fps (10-25x speedup)
- Sobel detection: ~200-500 fps (4-10x speedup)
- Total preprocessing: ~30-60 seconds

## Hardware Requirements for GPU

- **GPU**: NVIDIA with CUDA Compute Capability ≥ 3.0
  - RTX 3060: ✅ Compute 8.6 (Ampere)
  - RTX 20XX: ✅ Compute 7.5 (Turing)
  - GTX 10XX: ✅ Compute 6.1 (Pascal)

- **CUDA**: 11.0 or higher (13.1 installed in current system)

- **Memory**: 2GB+ VRAM recommended

## Troubleshooting

### CUDA Available but Not Detected
```python
import cv2
print(hasattr(cv2, 'cuda'))  # Should be True
print(cv2.cuda.getCudaEnabledDeviceCount())  # Should be > 0
```

If False, OpenCV was not built with CUDA support.

### Build Errors
- Ensure CUDA Toolkit version matches OpenCV requirements
- Check Visual Studio version compatibility
- Verify GPU compute capability

## Recommendation

**For development**: Use current CPU implementation with frame caching. It's fast enough.

**For production**: Consider building CUDA-enabled OpenCV if processing many videos or requiring real-time performance.

## Future Enhancements

Potential GPU optimizations:
- Batch processing multiple frames
- GPU-based connected components (patch removal)
- End-to-end GPU pipeline (no CPU-GPU transfers)

These would require custom CUDA kernels or specialized libraries.
