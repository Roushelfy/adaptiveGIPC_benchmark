#!/bin/bash
# GPU 检查脚本 / GPU Check Script

echo "========================================"
echo "CUDA GPU 环境检查 / CUDA GPU Environment Check"
echo "========================================"
echo ""

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found!"
    echo "请安装 NVIDIA 驱动 / Please install NVIDIA drivers"
    exit 1
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  WARNING: nvcc (CUDA compiler) not found!"
    echo "CUDA 可能未正确安装或未添加到 PATH / CUDA may not be installed or not in PATH"
    echo ""
    echo "尝试运行: / Try running:"
    echo "  export PATH=/usr/local/cuda-12/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:\$LD_LIBRARY_PATH"
    echo ""
else
    echo "✅ CUDA Compiler found:"
    nvcc --version | head -1
    echo ""
fi

# List GPUs
echo "📊 Available GPUs:"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | while IFS=, read -r idx name cap mem; do
    echo "  GPU $idx: $name"
    echo "    Compute Capability: $cap"
    echo "    Memory: $mem"
done
echo ""

# Recommended architecture
echo "💡 Recommended CMAKE_CUDA_ARCHITECTURES:"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr -d '.' | while read arch; do
    echo "  - $arch"
done
echo ""

# Check OpenGL
echo "🎨 OpenGL Check:"
if command -v glxinfo &> /dev/null; then
    glxinfo | grep "OpenGL version" || echo "  ⚠️  Could not get OpenGL version"
else
    echo "  ⚠️  glxinfo not found (install mesa-utils)"
fi
echo ""

echo "========================================"
echo "检查完成 / Check Complete"
echo "========================================"
