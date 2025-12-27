#!/bin/bash
# 快速构建脚本 / Quick Build Script

set -e  # Exit on error

echo "========================================"
echo "PainlessMG Linux CUDA 12 - Quick Build"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check GPU
echo -e "${YELLOW}[1/5] Checking GPU environment...${NC}"
./check_gpu.sh || {
    echo -e "${RED}GPU check failed. Please fix GPU/CUDA issues first.${NC}"
    exit 1
}

# Clean old build
if [ -d "build" ]; then
    echo -e "${YELLOW}[2/5] Cleaning old build...${NC}"
    rm -rf build
fi

# Create build directory
echo -e "${YELLOW}[3/5] Creating build directory...${NC}"
mkdir -p build
cd build

# Configure
echo -e "${YELLOW}[4/5] Configuring with CMake...${NC}"
cmake .. || {
    echo -e "${RED}CMake configuration failed!${NC}"
    echo "请检查依赖库是否已安装 / Please check if dependencies are installed:"
    echo "  sudo apt-get install build-essential cmake nvidia-cuda-toolkit freeglut3-dev libglew-dev"
    exit 1
}

# Build
echo -e "${YELLOW}[5/5] Building (this may take a while)...${NC}"
make -j$(nproc) || {
    echo -e "${RED}Build failed!${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}========================================"
echo "✅ Build Successful!"
echo "========================================${NC}"
echo ""
echo "运行模拟 / Run simulations:"
echo "  Armadillo: cd armadillo && ./armadillo_sim"
echo "  Cloth:     cd cloth && ./cloth_sim"
echo ""
