#!/bin/bash
# Setup script for vLLM server with Qwen2.5-VL-7B (no quantization - fallback)

set -e

echo "=========================================="
echo "vLLM Setup for Qwen2.5-VL-7B (No Quantization)"
echo "=========================================="

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected. This system requires CUDA."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install vLLM if not already installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm>=0.6.0
else
    echo "vLLM already installed"
fi

# Install additional dependencies
echo "Installing dependencies..."
pip install transformers>=4.40.0 torch>=2.1.0

# Create model cache directory
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-models/cache}"
mkdir -p "$MODEL_CACHE_DIR"

echo ""
echo "=========================================="
echo "Starting vLLM Server (No Quantization)"
echo "=========================================="
echo "Model: Qwen/Qwen2.5-VL-7B-Instruct"
echo "Quantization: None (FP16/BF16)"
echo "Port: 8000"
echo "Memory Usage: Higher (~15GB VRAM)"
echo "=========================================="

# Start vLLM server without quantization
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype auto \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code

echo ""
echo "vLLM server started successfully!"
echo "Access at: http://localhost:8000"
