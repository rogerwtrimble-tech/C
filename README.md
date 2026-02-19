# Medical Data Extraction System

HIPAA-Compliant **Fully Local Processing** with SOLAR 10.7B SLM via Ollama and **Intelligent PDF Type Detection** for extracting attributes from medical/workers' compensation PDF documents.

**System Requirements**: Windows 11 + WSL 2.0 + Docker Desktop + NVIDIA GPU (12GB+ VRAM)

## Quick Start

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

### 3. Install OCR Engine (Required for Scanned PDFs)

**Option 1: Tesseract (CPU-based)**
```bash
# Windows
winget install UB-Mannheim.Tesseract

# Add to PATH: C:\Program Files\Tesseract-OCR
```

**Option 2: EasyOCR (GPU-accelerated)**
```bash
pip install easyocr
# Requires CUDA for GPU acceleration
```

**Option 3: PaddleOCR (Fastest GPU)**
```bash
pip install paddlepaddle-gpu paddleocr
# Requires CUDA for GPU acceleration
```

### 4. Configure Environment

```bash
cp .env.example .env
# No API key required - all inference is local
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

### Intelligent PDF Processing

The system automatically detects PDF type and selects the optimal processing path:

1. **Native PDFs** (alphanumeric ratio > 0.30)
   - Direct text extraction
   - No OCR required
   - 95%+ accuracy, 2-3 sec latency

2. **Scanned PDFs** (alphanumeric ratio < 0.10)
   - Full OCR processing
   - Tesseract/EasyOCR/PaddleOCR
   - 85-94% accuracy, 6-12 sec latency

3. **Hybrid PDFs** (0.10 ≤ ratio ≤ 0.30)
   - Per-page selective processing
   - Text extraction for native pages
   - OCR for scanned pages
   - 90%+ accuracy, 8 sec latency

### Hardware Optimization

- **CPU**: SOLAR 10.7B inference (text-only model)
- **GPU**: OCR acceleration (EasyOCR/PaddleOCR)
- **RAM**: 64GB for model + document staging

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
