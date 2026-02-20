# Medical Data Extraction System

HIPAA-Compliant **Fully Local Processing** with dual-mode support:
- **Grade A (85-90% accuracy)**: Multimodal VLM with Qwen2.5-VL-3B + YOLOv8 signature detection
- **Grade B (85-90% accuracy)**: Legacy text-only with SOLAR 10.7B + OCR
- **Grade C (85-90% accuracy)**: Chunked processing for large documents

**System Requirements**: Windows 11 + WSL 2.0 + Docker Desktop + NVIDIA GPU (12GB+ VRAM for VLM, 12GB for legacy)

## Quick Start

### Choose Your Mode

**For Grade A (Multimodal VLM)**: See [VLM_SETUP_GUIDE.md](VLM_SETUP_GUIDE.md)

**For Grade B (Legacy Text-Only)**:

### 1. Setup Ollama (WSL 2.0)

```bash
# In WSL 2.0 Ubuntu terminal
docker run -d --gpus all -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull solar:10.7b
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install OCR Engine (Legacy Mode) or VLM Server (VLM Mode)

**For Legacy Mode - OCR Engine:**

**Option 1: Tesseract (CPU-based)**
```bash
# Windows
winget install UB-Mannheim.Tesseract
# Add to PATH: C:\Program Files\Tesseract-OCR
```

**For VLM Mode - Setup vLLM:**
```bashl
# Run setup script
chmod +x scripts/setup_vllm.sh
./scripts/setup_vllm.sh

# Download YOLOv8 signature model
python scripts/download_yolo_model.py
```

### 4. Configure Environment

```bash
cp .env.example .env

# For VLM Mode:
PROCESSING_MODE=vlm
VLM_ENABLED=true

# For Legacy Mode:
PROCESSING_MODE=text
VLM_ENABLED=false
```

### 5. Run Extraction

```bash
python main.py
# Or test with specific file
python test_intelligent_extraction.py
```

Results are saved to `results/` as JSON files.

## Extracted Fields

| Field | Required | Description |
|-------|----------|-------------|
| `claim_id` | Yes | Claim identifier (alphanumeric + hyphens) |
| `patient_name` | Yes | Patient name |
| `document_type` | No | Document type (defaults to "Unknown") |
| `date_of_loss` | No | Date of injury/loss (ISO 8601) |
| `diagnosis` | No | Diagnosis or ICD-10 code |
| `dob` | No | Date of birth (ISO 8601) |
| `provider_npi` | No | Provider NPI number |
| `total_billed_amount` | No | Total billed amount |

## Architecture

### Dual-Mode Processing

#### Grade A: Multimodal VLM (95%+ Accuracy)

**4-Stage Workflow:**
1. **Preprocessing**: PDF → 300 DPI images with deskew (~2 sec)
2. **Vision Sweep**: Qwen2.5-VL-72B extracts data + identifies signatures (~25-40 sec)
3. **Signature Validation**: YOLOv8 verifies VLM detections (~1 sec)
4. **Auto-Clipping**: Save signature images with bounding boxes
5. **Confidence Check**: Flag VLM/YOLO discrepancies for review

**Capabilities:**
- ✅ Signature detection and validation
- ✅ Handwritten note transcription (90-95% accuracy)
- ✅ Layout understanding (tables, forms, multi-column)
- ✅ Visual grounding with bounding boxes
- ✅ Stamp/seal detection

**Performance**: 1-2 sec/page on RTX 4070/3060 (12GB VRAM), 4-7 sec/page on RTX 4090 (24GB VRAM)

#### Grade B: Legacy Text-Only (85-90% Accuracy)

**3-Path Workflow:**
1. **Native PDFs** (ratio > 0.30): Direct text extraction
2. **Scanned PDFs** (ratio < 0.10): Full OCR processing
3. **Hybrid PDFs** (0.10-0.30): Per-page selective OCR

**Performance**: 2-12 sec/page depending on PDF type

### Hardware Optimization

**VLM Mode:**
- **CPU**: Orchestration and overflow
- **GPU**: Qwen2.5-VL-3B inference + YOLOv8 detection
- **RAM**: 64GB for model + image staging
- **VRAM**: 12GB+ (RTX 4070/3060 for 3B model, RTX 4090/3090 for 72B model)

**Legacy Mode:**
- **CPU**: SOLAR 10.7B inference
- **GPU**: OCR acceleration (optional)
- **RAM**: 64GB
- **VRAM**: 12GB

## Project Structure

```
poc-ws-ocr/
├── main.py              # Entry point
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── pdfs/               # Input PDF directory
├── results/            # Extraction results (JSON)
├── logs/audit/         # Audit logs (de-identified)
└├── src/
    ├── __init__.py
    ├── config.py           # Configuration management
    ├── models.py           # Pydantic schemas with validation
    ├── pdf_interrogator.py # PDF type detection
    ├── pdf_extractor.py    # PDF text/image extraction
    ├── ocr_extractor.py    # Multi-engine OCR support
    ├── ollama_client.py    # Ollama API client (local SLM)
    ├── audit_logger.py     # Audit logging with de-identification
    ├── secure_handler.py   # Encryption & secure deletion
    └── pipeline.py         # Main processing pipeline
```

## Configuration

Environment variables (set in `.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | No | Ollama host (default: localhost) |
| `OLLAMA_PORT` | No | Ollama port (default: 11434) |
| `OLLAMA_MODEL` | No | Model name (default: solar:10.7b) |
| `OCR_ENGINE` | No | Preferred OCR engine (tesseract/easyocr/paddleocr) |
| `ENCRYPTION_KEY` | No | AES-256 key (base64, 32 bytes) |
| `PDF_INPUT_DIR` | No | Input directory (default: pdfs) |
| `RESULTS_OUTPUT_DIR` | No | Output directory (default: results) |
| `MAX_CONCURRENT_REQUESTS` | No | Concurrent requests (default: 4) |
| `SECURE_DELETION_PASSES` | No | File overwrite passes (default: 3) |

## HIPAA Compliance

This system implements:

- **Fully local inference**: No PHI leaves the system - all processing on local GPU/CPU
- **Zero PHI persistence**: Temp files securely deleted after processing
- **De-identified audit logs**: Only hashes stored, not actual PHI
- **Encryption at rest**: AES-256 for stored data (when configured)
- **Structured extraction**: Pydantic validation prevents invalid data
- **No external API calls**: No BAA required - fully air-gapped processing
- **PDF type tracking**: All processing paths logged for audit trail

## API Usage

```python
from src.pipeline import ExtractionPipeline
import asyncio

async def process():
    pipeline = ExtractionPipeline()
    
    # Process single document
    result = await pipeline.process_document("pdfs/claim.pdf")
    print(result.claim_id)
    
    # Process directory
    results = await pipeline.process_directory()
    
    await pipeline.cleanup()

asyncio.run(process())
```

## Output Format

Results are JSON files with this structure:

```json
{
  "claim_id": "WC-2024-00123",
  "patient_name": "John Doe",
  "document_type": "workers_compensation_claim",
  "date_of_loss": "2024-01-15",
  "diagnosis": "S72.001A",
  "dob": "1985-06-20",
  "provider_npi": "1234567890",
  "total_billed_amount": "$15,750.00",
  "confidence_scores": {
    "claim_id": 0.99,
    "patient_name": 0.95,
    "date_of_loss": 0.87
  }
}
```

## Performance Targets

| Document Type | Processing Path | Expected Accuracy | Latency | OCR Engine |
|---------------|-----------------|-------------------|---------|------------|
| Digital forms | Native Direct | 95%+ | 2-3 sec | N/A |
| Scanned reports | Scanned Full OCR | 85-90% | 10-12 sec | Tesseract |
| Scanned reports | Scanned Full OCR | 90-94% | 6-8 sec | EasyOCR/PaddleOCR |
| Mixed documents | Hybrid Selective | 90%+ | 8 sec | Adaptive |
| Poor quality faxes | Scanned Full OCR | 75-85% | 12 sec | Tesseract |
