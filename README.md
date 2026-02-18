# Medical Data Extraction System

HIPAA-Compliant **Fully Local Processing** with SOLAR 10.7B SLM via Ollama for extracting attributes from medical/workers' compensation PDF documents.

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

### 3. Configure Environment

```bash
cp .env.example .env
# No API key required - all inference is local
```

### 4. Run Extraction

```bash
python main.py
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

## Project Structure

```
poc-ws-ocr/
├── main.py              # Entry point
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── pdfs/               # Input PDF directory
├── results/            # Extraction results (JSON)
├── logs/audit/         # Audit logs (de-identified)
└── src/
    ├── __init__.py
    ├── config.py       # Configuration management
    ├── models.py       # Pydantic schemas
    ├── ollama_client.py   # Ollama API client (local SLM)
    ├── pdf_extractor.py    # PDF text/image extraction
    ├── audit_logger.py     # Audit logging
    ├── secure_handler.py   # Encryption & secure deletion
    └── pipeline.py     # Main processing pipeline
```

## Configuration

Environment variables (set in `.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | No | Ollama host (default: localhost) |
| `OLLAMA_PORT` | No | Ollama port (default: 11434) |
| `OLLAMA_MODEL` | No | Model name (default: solar:10.7b) |
| `ENCRYPTION_KEY` | No | AES-256 key (base64, 32 bytes) |
| `PDF_INPUT_DIR` | No | Input directory (default: pdfs) |
| `RESULTS_OUTPUT_DIR` | No | Output directory (default: results) |
| `MAX_CONCURRENT_REQUESTS` | No | Concurrent requests (default: 4) |
| `SECURE_DELETION_PASSES` | No | File overwrite passes (default: 3) |

## HIPAA Compliance

This system implements:

- **Fully local inference**: No PHI leaves the system - all processing on local GPU
- **Zero PHI persistence**: Temp files securely deleted after processing
- **De-identified audit logs**: Only hashes stored, not actual PHI
- **Encryption at rest**: AES-256 for stored data (when configured)
- **Structured extraction**: Pydantic validation prevents invalid data
- **No external API calls**: No BAA required - fully air-gapped processing

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

| Metric | Target | Notes |
|--------|--------|-------|
| Field Extraction Rate | 95%+ | |
| Processing Latency | <60 sec/doc | Local GPU inference |
| Alias Recognition | 99%+ | |
| False Positives | <1% | |
