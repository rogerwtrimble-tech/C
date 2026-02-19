# Medical Data Extraction System - UPDATED REQUIREMENTS
## HIPAA-Compliant Local Processing with SOLAR 10.7B + PDF Type Detection

---

## 1. PROJECT OVERVIEW
**Purpose**: Extract 8 defined attributes from medical/workers' compensation PDF documents (both native and scanned) with HIPAA compliance and high retrieval accuracy (target: 90%+ average, 95%+ on native PDFs).

**Processing Model**: Local extraction via SOLAR 10.7B (Ollama) with PDF type detection and conditional OCR preprocessing.

**Compliance**: HIPAA Title II, zero PHI egress, encrypted processing, immutable audit logs.

**Data Residency**: 100% on-premises; no external API calls.

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Hardware Specification
```
CPU:     Intel Core i9 (document orchestration, SOLAR inference on CPU)
RAM:     64 GB (PDF staging, OCR batching, SOLAR model in memory)
GPU:     RTX 4090 (12 GB VRAM)
  └─ Primary use: OCR acceleration (EasyOCR or Paddle-OCR)
  └─ NOT used for SOLAR inference (text-only model runs on CPU)
Storage: ≥2 TB SSD encrypted (temp staging only)
```

### 2.2 Software Stack (REVISED)

#### Core Dependencies
```
Runtime:        Python 3.11+
LLM:            SOLAR 10.7B via Ollama (local, on-premises)
PDF Processing: pdfplumber>=0.11.0, PyMuPDF>=1.23.0
OCR:            Tesseract-OCR + pytesseract (Phase 1)
                EasyOCR>=1.7.0 (Phase 2, GPU-accelerated)
                OR Paddle-OCR>=2.7.0.3 (Phase 2, fastest)
PDF-to-Image:   pdf2image>=1.16.3
Text Cleaning:  regex, unicodedata
Encryption:     cryptography>=41.0.0
Validation:     pydantic>=2.0.0
Async:          asyncio + aiohttp
Logging:        structlog + custom de-identifier
Database:       PostgreSQL 15+ (optional, encrypted)
```

#### Ollama Deployment
```bash
# Local LLM server (no API key, no external calls)
docker run -d \
  --gpus all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama

# Pull SOLAR 10.7B model
ollama pull solar
# Model size: ~6-7GB (fits in 12GB VRAM + RAM spillover)
# Inference: Runs on CPU if GPU VRAM insufficient (auto fallback)
```

---

## 3. CRITICAL INSIGHT: PDF BINARY TEXT SEARCHABILITY

### Why PDFs Require Type Detection

A PDF binary is **only text-searchable if it contains embedded text streams**. Two scenarios:

| Scenario | Root Cause | Detection | Solution |
|----------|-----------|-----------|----------|
| **Native PDF** | Created digitally (Word→PDF, web form, etc.); contains searchable text streams | Text extraction succeeds; alphanumeric_ratio > 0.30 | Send extracted text directly to SOLAR 10.7B |
| **Scanned PDF** | Scanned document; contains only image layers; NO searchable text | Text extraction yields garbage or empty; alphanumeric_ratio < 0.10 | Run OCR first, then send OCR'd text to SOLAR |

**You cannot determine PDF type by inspecting the binary header alone.** You must attempt text extraction and validate the result.

---

## 4. DATA FLOW & PROCESSING PIPELINE (WITH PDF INTERROGATION)

### 4.1 Input Stage
```
1. Receive encrypted PDF (AES-256) or plaintext PDF
2. Validate file integrity (checksum)
3. Decrypt if necessary
4. Log receipt with timestamp, filename hash (no actual filename stored)
```

### 4.2 PDF Interrogation (Type Detection)
```python
def interrogate_pdf(pdf_path: str) -> dict:
    """Determine if PDF is native (text) or scanned (image)."""
    with pdfplumber.open(pdf_path) as pdf:
        # Sample first 3 pages
        sample_pages = min(3, len(pdf.pages))
        extracted_text = ""
        
        for page_num in range(sample_pages):
            page_text = pdf.pages[page_num].extract_text() or ""
            extracted_text += page_text
        
        # Calculate alphanumeric ratio
        if extracted_text:
            alpha_count = sum(1 for c in extracted_text if c.isalnum())
            alpha_ratio = alpha_count / len(extracted_text)
        else:
            alpha_ratio = 0.0
        
        # Classify PDF type
        if alpha_ratio > 0.30:
            pdf_type = "native"
        elif 0.10 <= alpha_ratio <= 0.30:
            pdf_type = "hybrid"
        else:
            pdf_type = "scanned"
        
        return {
            "pdf_type": pdf_type,
            "alphanumeric_ratio": round(alpha_ratio, 3),
            "requires_ocr": pdf_type in ("scanned", "hybrid"),
            "total_pages": len(pdf.pages),
            "sample_text_length": len(extracted_text)
        }
```

**Decision Logic**:
```python
metadata = interrogate_pdf(pdf_path)

if metadata["pdf_type"] == "native":
    path = "NATIVE_PDF"  # Skip OCR
elif metadata["pdf_type"] == "scanned":
    path = "SCANNED_PDF"  # Run OCR
else:  # hybrid
    path = "HYBRID_PDF"  # Selective OCR per page
```

### 4.3 Processing Stage (Conditional on PDF Type)

#### Path A: Native PDF (No OCR Required)
```
1. Extract full text with pdfplumber
2. Clean/normalize:
   ├─ Remove excess whitespace
   ├─ Normalize line breaks
   ├─ Decode Unicode characters
   └─ Remove control characters
3. Chunk if document > 20 pages
4. Send to SOLAR 10.7B with extraction prompt
5. Validate extraction against Pydantic schema
6. De-identify logs (hash document, remove actual PHI)

Performance:
  Accuracy: 95%+
  Latency: 1-3 sec/doc
  Resource: CPU only
```

#### Path B: Scanned PDF (OCR Required)
```
1. Detect image layers (use PyMuPDF to list images or pdfplumber page.images)
2. Convert PDF pages to images:
   └─ Use pdf2image.convert_from_path() for full quality
3. Run OCR on each page:
   
   Option 1 (Phase 1): Tesseract
   ├─ Command-line: tesseract image.png stdout
   ├─ Speed: Fast (1-2 sec/page)
   ├─ Accuracy: 85-90%
   ├─ GPU: No (CPU-only)
   ├─ Cost: $0
   
   Option 2 (Phase 2): EasyOCR (GPU-accelerated)
   ├─ Speed: Medium (2-4 sec/page on GPU)
   ├─ Accuracy: 90-95%
   ├─ GPU: ✅ RTX 4090
   ├─ Cost: $0 (local)
   
   Option 3 (Phase 2): Paddle-OCR (fastest)
   ├─ Speed: Fast (1-2 sec/page on GPU)
   ├─ Accuracy: 92-96%
   ├─ GPU: ✅ RTX 4090
   ├─ Cost: $0 (local)

4. Extract text from OCR output
5. Send to SOLAR 10.7B with extraction prompt
6. Validate extraction against Pydantic schema
7. Flag low-confidence fields (<0.75) for manual review
8. De-identify logs

Performance:
  Accuracy: 85-92% (OCR introduces ~5-10% error vs. native PDF)
  Latency: 8-15 sec/doc (OCR overhead)
  Resource: CPU for orchestration, GPU for OCR (optional, speeds up 2-3x)
```

#### Path C: Hybrid PDF (Selective Processing per Page)
```
1. Iterate through each page
2. For each page:
   ├─ Attempt text extraction
   ├─ If alphanumeric_ratio > 0.30 → treat as native (Path A)
   └─ If alphanumeric_ratio ≤ 0.30 → treat as scanned (Path B)
3. Combine results across pages
4. Send combined text to SOLAR 10.7B

Performance:
  Accuracy: 90%+ (depends on text/image page ratio)
  Latency: 6-12 sec/doc
  Resource: CPU + optional GPU for image pages
```

### 4.4 Field Extraction (SOLAR 10.7B)

**Critical**: SOLAR 10.7B is text-only LLM.
- ✅ Input: Clean text strings (from native PDF or OCR)
- ❌ Input: Image data, binary streams, unprocessed PDFs
- ❌ SOLAR has NO vision capability; cannot perform OCR itself

```python
def extract_fields_with_solar(text: str) -> dict:
    """
    Send cleaned text to SOLAR 10.7B for field extraction.
    SOLAR runs on CPU via Ollama (localhost:11434).
    """
    prompt = f"""
Extract the following 8 fields from the medical document text below.
Return ONLY valid JSON. If a field is missing, set to null.

Expected fields and aliases:
1. claim_id: claim_id, claim_number, case_id, docket_no, control_no, file_no, tpa_ref
2. patient_name: patient_name, insured, claimant, injured_worker, petitioner, ee_name, member
3. document_type: document_type, doc_type, form_name, notice_title, exhibit_type, report_type
4. date_of_loss: date_of_injury, doi, date_of_loss, accident_date, incident_date, injury_dt
5. diagnosis: icd_10, diagnosis_code, dx_code, primary_dx, nature_of_injury
6. dob: date_of_birth, birth_date, dob
7. provider_npi: npi, billing_provider_npi, provider_id
8. total_billed_amount: total_charges, billed_amount, total_due

RULES:
- Do NOT hallucinate data.
- For missing fields, return null.
- For dates, use ISO 8601 (YYYY-MM-DD) or "Unclear".
- For NPI, extract 10-digit number.
- For currency, return as $X.XX.
- Include "confidence_scores": {{ "field_name": 0.0–1.0 }}.

Document text:
{text}

Return JSON only:
"""
    
    # Call SOLAR via Ollama (localhost, no API key)
    response = ollama_client.generate(
        model="solar",
        prompt=prompt,
        stream=False
    )
    
    extracted_json = parse_json_from_response(response)
    return extracted_json
```

### 4.5 Validation & De-identification
```
1. Pydantic schema validation (ExtractionResult model)
2. Check confidence scores:
   ├─ If any field < 0.75 → log as "low_confidence"
   ├─ Flag for manual review if required_field < 0.75
3. De-identify logs:
   ├─ Replace actual PHI (names, dates, IDs) with placeholders
   ├─ Store document hash (SHA-256) for audit trail
   ├─ Log extraction metadata (accuracy, latency, path taken)
4. Delete staging files (PDF, images, OCR temp files)
5. Return JSON extraction result
```

---

## 5. HARDWARE & GPU UTILIZATION

### CPU (Intel Core i9)
- **Used for**: Document orchestration, text extraction, SOLAR inference (LLM), validation, I/O
- **SOLAR 10.7B inference**: Runs on CPU because it's a text-only model
  - ~12-15 GB RAM required (with model in memory)
  - Inference latency: 2-3 sec/doc on i9 (single-threaded)
  - Can be parallelized with async processing

### GPU (RTX 4090, 12GB VRAM)
- **Used for**: OCR acceleration (EasyOCR, Paddle-OCR)
- **NOT used for**: SOLAR inference (it's a CPU-optimized text model)
- **OCR speedup**: 2-3x faster than CPU with EasyOCR/Paddle-OCR
  - Without GPU: 8-15 sec/doc (Tesseract on CPU)
  - With GPU: 4-8 sec/doc (EasyOCR/Paddle-OCR on 4090)

### RAM (64GB)
- **SOLAR model**: ~12-15 GB
- **PDF staging buffer**: ~5-10 GB (for large documents, image batching)
- **OS + system**: ~4 GB
- **Available for processing**: ~30-40 GB

---

## 6. ACCURACY & PERFORMANCE EXPECTATIONS

### By Document Type

| Document Type | PDF Format | Processing Path | Expected Accuracy | Latency | Notes |
|---------------|-----------|-----------------|-------------------|---------|-------|
| Digital claim form | Native | Interrogate → Extract → SOLAR | 95%+ | 2 sec | Best case |
| Scanned medical report | Scanned | Interrogate → OCR → SOLAR | 88-92% | 10-12 sec | Tesseract |
| Scanned report (GPU) | Scanned | Interrogate → EasyOCR → SOLAR | 90-94% | 6-8 sec | With RTX 4090 |
| Mixed doc (text + images) | Hybrid | Per-page routing | 90%+ | 8 sec | Adaptive |
| Fax (poor quality) | Scanned | Interrogate → OCR → SOLAR | 75-85% | 12 sec | Low quality |
| Handwritten notes | Scanned | Interrogate → OCR fails | 20-30% | 10 sec | Flag for manual |

### Field-Level Accuracy

| Field | Native PDF | Scanned PDF (Tesseract) | Scanned PDF (EasyOCR) |
|-------|-----------|------------------------|------------------------|
| claim_id | 98% | 85% | 92% |
| patient_name | 96% | 88% | 94% |
| document_type | 94% | 82% | 90% |
| date_of_loss | 95% | 80% | 88% |
| diagnosis | 92% | 75% | 85% |
| dob | 97% | 82% | 90% |
| provider_npi | 99% | 70% | 88% |
| total_billed_amount | 96% | 78% | 90% |

---

## 7. IMPLEMENTATION PHASES

### Phase 1 (Week 1-2): Native PDF + Basic OCR
**Goal**: Support native PDFs (95%+ accuracy) + scanned PDFs with Tesseract

**Deliverables**:
- PDF interrogation function (`interrogate_pdf()`)
- Native PDF text extraction pipeline
- Tesseract OCR integration (CPU-based)
- SOLAR 10.7B field extraction
- Pydantic validation schema
- Audit logging with de-identification

**Testing**: 20 test PDFs (10 native, 10 scanned)

**Target Accuracy**: 95% on native, 85% on scanned

### Phase 2 (Week 3): GPU-Accelerated OCR (Optional)
**Goal**: Improve scanned PDF accuracy + speed with GPU

**Deliverables**:
- EasyOCR or Paddle-OCR integration
- GPU batching for multiple pages
- Comparative latency benchmarking

**Testing**: Re-test 10 scanned PDFs with GPU vs. CPU Tesseract

**Target Accuracy**: 90% on scanned PDFs

### Phase 3 (Week 4): Production Hardening
**Goal**: Scale to production workload

**Deliverables**:
- Parallel document processing (asyncio)
- Confidence thresholding for manual review
- Performance monitoring
- Incident handling (corrupted PDFs, OCR failures)

**Testing**: 100-doc end-to-end test

---

## 8. HIPAA COMPLIANCE REQUIREMENTS

### 8.1 Administrative
- [ ] Data Processing Agreement with Ollama (if applicable)
- [ ] Risk assessment completed
- [ ] Security officer designated
- [ ] Workforce training on PHI handling
- [ ] Incident response plan documented

### 8.2 Physical
- [ ] Secure facility access (locked server room)
- [ ] No external USB/removable media
- [ ] Screen privacy
- [ ] Temperature monitoring

### 8.3 Technical
- [ ] **Encryption at Rest**: AES-256 for all stored data
- [ ] **Encryption in Transit**: TLS 1.3 (inter-process, if applicable)
- [ ] **Access Controls**: File permissions (700 on sensitive dirs)
- [ ] **Audit Logging**: Immutable, de-identified logs
- [ ] **Secure Deletion**: Overwrite temp files (3-pass shred)
- [ ] **De-identification**: Strip actual PHI from logs; store hashes
- [ ] **Network**: Firewall (no external connections, local only)
- [ ] **Backup**: Encrypted offline backups of audit logs

### 8.4 Key Advantage Over Claude API
✅ **Zero PHI Egress**: All processing is local
- No data sent to external services
- No API keys to Anthropic or any vendor
- Full compliance with data residency requirements
- Auditors approve "data never leaves the facility"

---

## 9. VALIDATION & ERROR HANDLING

### Pydantic Schema
```python
from pydantic import BaseModel, Field, field_validator

class ExtractionResult(BaseModel):
    claim_id: str
    patient_name: str
    document_type: str = "Unknown"
    date_of_loss: Optional[str] = None
    diagnosis: Optional[str] = None
    dob: Optional[str] = None
    provider_npi: Optional[str] = None
    total_billed_amount: Optional[str] = None
    confidence_scores: dict
    processing_path: str  # "native", "scanned", "hybrid"
    
    @field_validator("claim_id")
    def validate_claim_id(cls, v):
        if not v or len(v) > 50:
            raise ValueError("claim_id must be 1-50 characters")
        return v
    
    @field_validator("dob", "date_of_loss", mode="before")
    def validate_dates(cls, v):
        if v and v != "Unclear":
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("dates must be ISO 8601 (YYYY-MM-DD)")
        return v
```

### Fallback Handling
- **Native PDF extraction fails**: Flag document; log error (de-identified); return null fields
- **OCR fails (scanned PDF)**: Flag for manual review; log low confidence
- **SOLAR extraction fails**: Return null fields; log LLM error
- **Schema validation fails**: Log validation error; return partial result with errors

---

## 10. MONITORING & LOGGING

### De-Identified Audit Log Format
```json
{
  "timestamp": "2025-02-18T14:30:00Z",
  "document_hash": "sha256_abc123...",
  "document_type_detected": "workers_compensation_claim",
  "pdf_type": "native",
  "processing_path": "native_direct",
  "fields_attempted": 8,
  "fields_extracted": 7,
  "fields_skipped": 1,
  "confidence_scores": {
    "claim_id": 0.99,
    "patient_name": 0.95,
    "document_type": 0.92,
    "date_of_loss": 0.88,
    "diagnosis": 0.75,
    "dob": 0.91,
    "provider_npi": 0.70,
    "total_billed_amount": 0.85
  },
  "extraction_latency_ms": 2850,
  "solar_model_version": "10.7b",
  "status": "success",
  "notes": "One field below confidence threshold; flagged for manual review"
}
```

### Key Metrics
- Extraction success rate (by document type)
- Average latency per document
- Confidence distribution
- Field-level accuracy (if ground truth available)
- OCR quality metrics (if used)

### Alerts
- Extraction success < 85% (daily)
- Latency spike > 30 sec/doc
- Storage usage > 80%
- Audit log integrity check failure

---

## 11. COST ESTIMATE

| Component | Cost | Notes |
|-----------|------|-------|
| **Hardware** | $8,000–12,000 (one-time) | i9, 64GB RAM, RTX 4090 |
| **Software** | $0 | Ollama, Python, Tesseract: all open-source |
| **Storage** (2TB encrypted SSD) | $200–400 (one-time) | Temporary staging only |
| **Backup/Archive** | $0–100/month | If archiving audit logs long-term |
| **Personnel** | $150K–200K/year | Security, admin, QA |
| **Total Y1** | ~$180K–260K | Includes hardware amortization |
| **Ongoing (Y2+)** | ~$80K–120K | Personnel + maintenance (no API costs) |

---

## 12. DEPLOYMENT CHECKLIST

- [ ] Ollama installed + SOLAR 10.7B model pulled
- [ ] Encryption keys generated (AES-256) + backed up offline
- [ ] Database created (if using) + encryption enabled
- [ ] Audit log directory with restricted permissions (700)
- [ ] Tesseract-OCR installed (Phase 1)
- [ ] EasyOCR/Paddle-OCR tested on GPU (Phase 2)
- [ ] Logging, monitoring configured
- [ ] Security training completed (team)
- [ ] Documentation finalized (SSP, incident response, runbooks)
- [ ] 20-doc staging test (10 native, 10 scanned)
- [ ] Production deployment + 30-day monitoring

---

## 13. RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| **Scanned PDFs fail (low OCR quality)** | Phase in with Tesseract; test on real docs; upgrade to EasyOCR if needed |
| **Handwritten documents** | Mark "unsupported"; flag for manual review; OCR confidence < 0.5 |
| **SOLAR inference too slow** | Parallelization with asyncio; benchmark on CPU; consider hardware upgrade if <1 doc/sec |
| **Data loss (encrypted staging)** | Secure deletion (3-pass shred); no persistent storage of PHI |
| **Audit log tampering** | Immutable log storage; periodic integrity checks (hash verification) |

---

## 14. APPENDIX A: Example Extraction Prompt for SOLAR

```
Extract the following 8 fields from the medical document text. 
Return ONLY valid JSON with the exact field names below.

Field names and possible aliases:
1. claim_id (aliases: claim_number, case_id, docket_no, control_no, file_no, tpa_ref)
2. patient_name (aliases: insured, claimant, injured_worker, petitioner, ee_name, member)
3. document_type (aliases: doc_type, form_name, notice_title, exhibit_type, report_type)
4. date_of_loss (aliases: date_of_injury, doi, accident_date, incident_date, injury_dt)
5. diagnosis (aliases: icd_10, diagnosis_code, dx_code, primary_dx, nature_of_injury)
6. dob (aliases: date_of_birth, birth_date)
7. provider_npi (aliases: npi, billing_provider_npi, provider_id)
8. total_billed_amount (aliases: total_charges, billed_amount, total_due)

STRICT RULES:
- Do NOT hallucinate data not in the document.
- For missing fields, return null.
- For dates, use ISO 8601 format (YYYY-MM-DD) or "Unclear" if ambiguous.
- For NPI, extract 10-digit number if present, else null.
- For currency, return as $X.XX if possible, else raw text.
- Include a "confidence_scores" object with 0–1 confidence for each field.
- For each field, estimate confidence based on:
  - Exact match vs. alias match (0.99 vs. 0.90)
  - Clarity of text (clean OCR = high; garbled text = low)
  - Field presence (found = 0.90+; inferred = 0.60; missing = 0.0)

Document text:
[PDF text here]

Return JSON (and ONLY JSON):
{
  "claim_id": "...",
  "patient_name": "...",
  "document_type": "...",
  "date_of_loss": "...",
  "diagnosis": "...",
  "dob": "...",
  "provider_npi": "...",
  "total_billed_amount": "...",
  "confidence_scores": {
    "claim_id": 0.99,
    "patient_name": 0.95,
    ...
  }
}
```

---

## 15. CONCLUSION

### This Solution **WILL WORK** for:

✅ **Native PDFs** (95%+ accuracy)
- Digital forms, reports, emails saved as PDF
- No OCR required
- Fast (1-3 sec)

✅ **Well-scanned documents** (88-94% accuracy depending on OCR engine)
- Modern scans (sharp, good contrast)
- Tesseract: 85-90% accuracy
- EasyOCR/Paddle-OCR on GPU: 90-94% accuracy

✅ **Hybrid documents** (90%+ accuracy)
- Mixed text + image pages
- Adaptive per-page processing

### This Solution **Will NOT work well** for:

❌ **Handwritten documents** (20-30% accuracy)
- Cursive OCR fails
- Recommend: Flag for manual review

❌ **Poor-quality scans** (faxes, bad photocopies)
- Accuracy drops to 75-85%
- Mitigation: Manual review for low-confidence fields

### Key Advantage

✅ **100% HIPAA-Compliant**
- All processing on-premises
- Zero PHI egress
- No vendor dependency
- Full audit trail with de-identification
- Auditors approve

---

**Document Version**: 2.0 (Updated with PDF Type Detection & OCR)  
**Last Updated**: 2025-02-18  
**Owner**: [Project Lead Name]  
**Classification**: Internal / HIPAA-Regulated
