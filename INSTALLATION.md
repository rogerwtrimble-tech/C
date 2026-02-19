# Installation Guide

## System Requirements

- **OS**: Windows 11 Pro with WSL 2.0
- **CPU**: Intel Core i9 (for orchestration and SOLAR inference)
- **RAM**: 64 GB (for model, documents, and processing)
- **GPU**: RTX 4090 (12GB VRAM) for OCR acceleration
- **Storage**: 2TB+ SSD (encrypted)

## Installation Steps

### 1. Install WSL 2.0 and Docker

```powershell
# Enable WSL
wsl --install

# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop
# Configure to use WSL 2 backend
```

### 2. Setup Ollama

```bash
# In WSL 2.0 Ubuntu terminal
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull solar:10.7b
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Install OCR Engine

#### Option A: Tesseract (Recommended - CPU-based)
```powershell
# Install Tesseract
winget install UB-Mannheim.Tesseract

# Add to PATH: C:\Program Files\Tesseract-OCR

# Verify installation
tesseract --version
```

#### Option B: EasyOCR (GPU-accelerated)
```bash
pip install easyocr
# Requires CUDA 11.8+ for GPU acceleration
```

#### Option C: PaddleOCR (Fastest GPU)
```bash
pip install paddlepaddle-gpu paddleocr
# Requires CUDA 11.8+ for GPU acceleration
```

### 5. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env (optional - defaults are provided)
# Set OCR_ENGINE if you have a preference
```

### 6. Verify Installation

```bash
# Test Ollama connection
python -c "import aiohttp; import asyncio; asyncio.run(aiohttp.ClientSession().get('http://localhost:11434/api/tags'))"

# Test PDF processing
python test_intelligent_extraction.py
```

## Troubleshooting

### Ollama Issues
- Ensure Docker Desktop is running
- Check WSL 2 integration in Docker settings
- Verify port 11434 is not blocked

### OCR Issues
- Tesseract: Add to system PATH
- EasyOCR/PaddleOCR: Install CUDA toolkit from NVIDIA
- Check GPU visibility with `nvidia-smi`

### Memory Issues
- Close unnecessary applications
- Monitor RAM usage (target: <80%)
- Consider reducing MAX_CONCURRENT_REQUESTS in .env

## Performance Optimization

### GPU Acceleration
- Use EasyOCR or PaddleOCR for scanned documents
- Ensure NVIDIA drivers are up to date
- Monitor GPU memory usage

### Concurrent Processing
- Adjust MAX_CONCURRENT_REQUESTS based on hardware
- Native PDFs: Can handle higher concurrency
- Scanned PDFs: Lower concurrency due to OCR overhead
