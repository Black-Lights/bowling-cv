#!/bin/bash
# Build and install OpenCV with CUDA support on Linux
# This script will compile OpenCV from source with GPU acceleration enabled

set -e  # Exit on error

# Configuration
OPENCV_VERSION="4.8.1"
VENV_PATH="/media/black-lights/407A01527A01465E/dev/image_analysis_v2/venv_gpu_linux"
BUILD_DIR="$HOME/opencv_build"
INSTALL_DIR="$VENV_PATH"

echo "========================================="
echo "OpenCV GPU Build Script"
echo "========================================="
echo "This will build OpenCV $OPENCV_VERSION with CUDA support"
echo "Build directory: $BUILD_DIR"
echo "Install directory: $INSTALL_DIR"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA toolkit not found. Please install CUDA first."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Installing cmake..."
    sudo apt-get update
    sudo apt-get install -y cmake
fi

echo "✓ CUDA version: $(nvcc --version | grep release)"
echo "✓ CMake version: $(cmake --version | head -n1)"
echo ""

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev

echo ""
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download OpenCV and OpenCV contrib
if [ ! -d "opencv" ]; then
    echo "Downloading OpenCV $OPENCV_VERSION..."
    git clone --depth 1 --branch $OPENCV_VERSION https://github.com/opencv/opencv.git
else
    echo "OpenCV directory already exists, skipping download..."
fi

if [ ! -d "opencv_contrib" ]; then
    echo "Downloading OpenCV contrib modules..."
    git clone --depth 1 --branch $OPENCV_VERSION https://github.com/opencv/opencv_contrib.git
else
    echo "OpenCV contrib directory already exists, skipping download..."
fi

# Create build directory
cd opencv
mkdir -p build
cd build

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Get Python paths (Python 3.12 compatible)
PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_PACKAGES_PATH=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

echo ""
echo "Python configuration:"
echo "  Executable: $PYTHON_EXECUTABLE"
echo "  Include dir: $PYTHON_INCLUDE_DIR"
echo "  Packages path: $PYTHON_PACKAGES_PATH"
echo ""

# Detect CUDA architecture
echo "Detecting CUDA architecture..."
# RTX 3060 Mobile has compute capability 8.6
CUDA_ARCH_BIN="8.6"
echo "Using CUDA architecture: $CUDA_ARCH_BIN (RTX 3060)"
echo ""

# Configure OpenCV with CMake
echo "Configuring OpenCV build with CMake..."
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
    -D PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
    -D PYTHON_PACKAGES_PATH=$PYTHON_PACKAGES_PATH \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_ARCH_BIN=$CUDA_ARCH_BIN \
    -D WITH_CUBLAS=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    ..

echo ""
echo "Building OpenCV (this will take a while)..."
echo "Using $(nproc) CPU cores for compilation..."
make -j$(nproc)

echo ""
echo "Installing OpenCV..."
make install

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Verifying installation..."
python -c "import cv2; print('OpenCV version:', cv2.__version__); print('CUDA enabled:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

echo ""
echo "To use this OpenCV build, activate the virtual environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Note: You may want to clean up the build directory to save space:"
echo "  rm -rf $BUILD_DIR"
echo ""
