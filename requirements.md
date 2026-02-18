# Medical Data Extraction System - Requirements
## HIPAA-Compliant Local Processing with SOLAR 10.7B SLM

---

## 1. PROJECT OVERVIEW
**Purpose**: Extract 8 defined attributes from medical/workers' compensation PDF documents with HIPAA compliance and high retrieval accuracy (target: 95%+ fields extracted).

**Processing Model**: **LOCAL extraction via SOLAR 10.7B SLM** running in Ollama Docker container (WSL 2.0). Zero PHI leaves the local system.

**Compliance**: HIPAA Title II, **fully local inference** (no external API calls), encrypted processing, immutable audit logs.

**System Configuration**: Windows 11 + WSL 2.0 (Ubuntu 22.04) + Docker Desktop

---

## 2. SYSTEM ARCHITECTURE

### 2.1 Hardware Specification
```
CPU:     Intel Core i9 (sufficient for local orchestration/validation)
RAM:     64 GB (buffer for document staging, SLM inference, encryption/decryption)
GPU:     RTX 4090 (12 GB VRAM)
  └─ CRITICAL: GPU required for SOLAR 10.7B inference via Ollama
Storage: ≥2 TB SSD (encrypted, temp staging only)
Network: **No external network required** - all inference is local
```

**GPU Usage Notes**:
- **SOLAR 10.7B runs locally on GPU via Ollama**
- GPU acceleration required for acceptable inference latency
- No data leaves the system - fully HIPAA compliant

### 2.2 Software Stack
```
Runtime:      Python 3.11+
LLM Provider: SOLAR 10.7B via Ollama (Docker container, port 11434)
  └─ Model: solar-10.7b-instruct-v1.0
  └─ Runs in WSL 2.0 Ubuntu environment
API Protocol: HTTP localhost:11434 (Ollama REST API)
Encryption:   AES-256 (at-rest), TLS 1.3 (in-transit for local services)
Databases:    None (stateless extraction) OR optional PostgreSQL 15+ with encryption
Async:        asyncio + aiohttp for concurrent local API calls
Logging:      Structured JSON audit logs (encrypted, immutable)
```

### 2.3 WSL 2.0 / Docker Configuration
```
Host OS:      Windows 11 Pro
WSL Version:  WSL 2.0 (Ubuntu 22.04 LTS)
Docker:       Docker Desktop (WSL 2 backend)
Ollama:       Docker container running on port 11434
  └─ docker run -d --gpus all -p 11434:11434 ollama/ollama
  └─ ollama pull solar:10.7b
```

---

## 3. EXTRACTION ATTRIBUTES & ALIASES

| Field | Required | Type | Aliases | Validation |
|-------|----------|------|---------|-----------|
| **claim_id** | Yes | str | claim_id, claim_number, case_id, docket_no, control_no, file_no, tpa_ref | 1–50 chars, alphanumeric + hyphens |
| **patient_name** | Yes | str | patient_name, insured, claimant, injured_worker, petitioner, ee_name, member | Non-empty, ≤150 chars |
| **document_type** | No | str (default: "Unknown") | document_type, doc_type, form_name, notice_title, exhibit_type, report_type | Enum validation against known doc types |
| **date_of_loss** | No | Optional[str] | date_of_injury, doi, date_of_loss, accident_date, incident_date, injury_dt | ISO 8601 (YYYY-MM-DD) or mark as "Unclear" |
| **diagnosis** | No | Optional[str] | icd_10, diagnosis_code, dx_code, primary_dx, nature_of_injury | ICD-10 format or free text, max 500 chars |
| **dob** | No | Optional[str] | date_of_birth, birth_date, dob | ISO 8601 (YYYY-MM-DD) or mark as "Unclear" |
| **provider_npi** | No | Optional[str] | npi, billing_provider_npi, provider_id | 10-digit NPI or alternative ID format |
| **total_billed_amount** | No | Optional[str] | total_charges, billed_amount, total_due | Currency ($X.XX) or numeric, max 15 chars |

---

## 4. DATA FLOW & PROCESSING PIPELINE

### 4.1 Input Stage
```
1. Receive encrypted PDF (AES-256) or plaintext PDF, from the "pdfs" directory 
2. Validate file integrity (checksum)
3. Decrypt if necessary
4. Log receipt with timestamp, filename hash (no filename stored)
```

### 4.2 Processing Stage
```
1. Extract text/images from PDF (local)
   - Use PyPDF2 or pdfplumber for text extraction
   - Optional: Send to Claude's vision API if images detected
2. Chunk if document > 20 pages (split, process independently)
3. Call Claude API with structured extraction prompt
4. Validate extraction against schema (Pydantic)
5. De-identify: replace actual PHI values with placeholders in logs
6. Return JSON extraction result as a stored file in the "results" directory
```

### 4.3 Output Stage
```
1. Return structured JSON (8 fields + confidence scores)
2. Encrypt result in transit (TLS already applied)
3. Delete staging files immediately after processing
4. Log completion (de-identified) to audit trail
5. Optional: Store in encrypted database (PostgreSQL + pgcrypto)
```

### 4.4 Cleanup Stage
```
1. Secure deletion: overwrite temp files (3-pass, e.g., shred utility)
2. Clear buffers containing PHI from RAM
3. Retain only encrypted audit logs with data hashes
```

---

## 5. HIPAA COMPLIANCE REQUIREMENTS

### 5.1 Administrative
- [ ] **BAA Signed** with Anthropic (Claude API usage)
- [ ] Risk assessment completed
- [ ] Security officer designated
- [ ] Workforce training on PHI handling
- [ ] Incident response plan documented

### 5.2 Physical
- [ ] Secure facility access (locked server room)
- [ ] No external USB/removable media
- [ ] Screen privacy (no shoulder surfing)
- [ ] Temperature/environment monitoring

### 5.3 Technical
- [ ] **Encryption at Rest**: AES-256 for all stored data
- [ ] **Encryption in Transit**: TLS 1.3 (Claude API, inter-process)
- [ ] **Access Controls**: API keys stored in secure vaults (e.g., HashiCorp Vault, AWS Secrets Manager)
- [ ] **Audit Logging**: Immutable, de-identified logs of all access/extraction
- [ ] **Secure Deletion**: Immediate overwrite of temp files (no recovery possible)
- [ ] **De-identification**: Strip actual PHI from logs; store only hashes/references
- [ ] **Network**: Firewall rules (whitelist Anthropic API IPs only)
- [ ] **Backup**: Encrypted backups with offline storage (if retaining audit logs)

### 5.4 Documentation
- [ ] System Security Plan (SSP)
- [ ] Data Processing Agreement (DPA)
- [ ] Audit log retention policy (≥6 years)
- [ ] Change log for software updates

---

## 6. PERFORMANCE & ACCURACY TARGETS

| Metric | Target | Notes |
|--------|--------|-------|
| **Field Extraction Rate** | 95%+ | Required fields always; optional fields best-effort |
| **Alias Recognition** | 99%+ | Prompt examples should cover all known variants |
| **Processing Latency** | <30 sec/doc | Document size dependent; single API call per doc |
| **Concurrent Documents** | 10–50 | Rate-limited by Claude API tier (monitor usage) |
| **False Positives** | <1% | Validation schema prevents invalid extractions |
| **Uptime** | 99.5%+ | Claude API availability + local infra redundancy |

---

## 7. CONFIGURATION & DEPLOYMENT

### 7.1 Environment Variables (Secure)
```bash
# .env (encrypted, never in VCS)
# Ollama Configuration (Local SLM)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=solar:10.7b

# Encryption
ENCRYPTION_KEY=<AES-256-key-base64>  # Rotate quarterly
AUDIT_LOG_PATH=/var/log/medical-extraction/audit/  # Encrypted fs
TEMP_DIR=/tmp/medical-processing/  # Ramdisk or encrypted SSD
DATABASE_URL=postgresql://user@localhost/medical_data  # if using DB
SECURE_DELETION_PASSES=3  # Overwrite depth for file deletion
```

### 7.2 Docker Deployment (WSL 2.0)
```dockerfile
# Python application container
FROM python:3.11-slim
RUN apt-get update && apt-get install -y pdfplumber cryptography aiohttp
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/
WORKDIR /app
CMD ["python", "main.py"]

# Run with: docker run --rm -v /encrypted/storage:/data --network host <image>
```

### 7.3 Ollama Container Setup (WSL 2.0)
```bash
# Start Ollama with GPU support
docker run -d --gpus all -p 11434:11434 --name ollama ollama/ollama

# Pull SOLAR 10.7B model
docker exec -it ollama ollama pull solar:10.7b

# Verify model is running
curl http://localhost:11434/api/tags
```

### 7.4 Kubernetes (Enterprise - On-Premises)
- StatefulSet for document processing workers
- **No external network access required** - fully air-gapped deployment
- PersistentVolume: encrypted storage (local SSD)
- Audit logging to centralized syslog (on-prem)
- GPU resource requests for Ollama pods

---

## 8. API INTEGRATION DETAILS

### 8.1 Ollama API Call Pattern (Local SLM)
```python
# Ollama REST API - localhost:11434
import aiohttp

async def extract_with_ollama(document_text: str) -> dict:
    prompt = f"""
Extract the following 8 fields from the medical document below.
Return ONLY valid JSON. If a field is missing, set to null.

Expected fields and aliases:
- claim_id: [alias list]
- patient_name: [alias list]
- document_type: [alias list]
... (all 8 fields)

Document text:
{document_text}

Return JSON:
{{
  "claim_id": "...",
  "patient_name": "...",
  ...
}}
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "solar:10.7b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048
                }
            }
        ) as response:
            result = await response.json()
            return parse_json(result["response"])
```

### 8.2 Vision/Multimodal Support (Future)
```python
# SOLAR 10.7B is text-only; for scanned documents:
# Option 1: Use local OCR (Tesseract) to extract text first
# Option 2: Use a multimodal model like LLaVA via Ollama
# Option 3: Use pdfplumber to extract embedded text

# Current implementation uses pdfplumber + OCR fallback
```

### 8.3 Rate Limiting & Retry Logic
- Implement exponential backoff (1s, 2s, 4s, 8s)
- **No external rate limits** - limited only by local GPU capacity
- Queue documents for async processing
- Log all inference failures for audit trail
- Monitor GPU memory usage to avoid OOM errors

---

## 9. VALIDATION & ERROR HANDLING

### 9.1 Schema Validation (Pydantic)
```python
from pydantic import BaseModel, Field, ValidationError

class ExtractionResult(BaseModel):
    claim_id: str
    patient_name: str
    document_type: str = "Unknown"
    date_of_loss: Optional[str] = None
    diagnosis: Optional[str] = None
    dob: Optional[str] = None
    provider_npi: Optional[str] = None
    total_billed_amount: Optional[str] = None
    confidence_scores: dict  # Field-level confidence (0–1)

# Validate extraction before storing
try:
    result = ExtractionResult(**extracted_data)
except ValidationError as e:
    log_error(e, de_identify=True)  # De-identify in logs
    return {"status": "validation_failed", "fields_recovered": result.dict(exclude_unset=True)}
```

### 9.2 Fallback Handling
- If Claude extraction fails: return null for field
- If PDF unreadable: flag document for manual review (analyst sees de-identified copy only)
- If API timeout: queue for retry (exponential backoff)

---

## 10. MONITORING & LOGGING

### 10.1 Audit Trail (De-identified)
```json
{
  "timestamp": "2025-02-18T14:30:00Z",
  "document_id": "hash-abc123",
  "document_type_detected": "workers_compensation_claim",
  "fields_extracted": 7,
  "fields_successful": 6,
  "extraction_latency_ms": 2850,
  "claude_model_version": "claude-opus-4-5-20251101",
  "confidence_avg": 0.94,
  "status": "success"
}
```

### 10.2 Metrics
- Extraction success rate (daily/weekly)
- Average latency per document
- API error rates
- Field-level accuracy (if ground truth available)
- Storage usage (encrypted data)

### 10.3 Alerts
- API call failures (>5% in 1 hour)
- Latency spike (>60 sec per doc)
- Encryption key rotation due
- Audit log disk usage >80%

---

## 11. SECURITY & THREAT MODEL

### 11.1 Threats Mitigated
| Threat | Mitigation |
|--------|-----------|
| PHI stored locally | Encrypt at rest (AES-256); delete immediately post-extraction |
| Interception in transit | TLS 1.3; no HTTP |
| Unauthorized API access | API key in Vault; rotate quarterly |
| Insider access | Role-based access; audit logging |
| GPU/RAM compromise | Secure wipe of sensitive buffers; no swap on /tmp |
| Database breach (if used) | Database encryption (pgcrypto); separate encryption keys |

### 11.2 Incident Response
1. Detect breach (suspicious logs, failed checks)
2. Isolate system (network disconnect if needed)
3. Preserve evidence (encrypted backups)
4. Notify OCO/Privacy Officer within 24 hours
5. Assess scope (query audit logs for affected dates/documents)
6. Remediate (patch, rotate keys, re-deploy)
7. Document and report (HHS Breach Notification Rule if >500 records)

---

## 12. TESTING & QUALITY ASSURANCE

### 12.1 Unit Tests
- Pydantic schema validation
- Date format parsing (ISO 8601, common variants)
- NPI format validation
- Currency parsing
- De-identification logic

### 12.2 Integration Tests
- Claude API call + mock responses
- PDF extraction (test PDFs with all field types)
- Encryption/decryption round-trip
- Audit logging (verify de-identification)

### 12.3 UAT (User Acceptance Testing)
- 50 representative documents (mix of document types)
- Manual review of 5 extractedfiles to confirm accuracy
- Performance baseline (latency, throughput)

### 12.4 Penetration Testing
- Network scanning (no unauthorized ports exposed)
- API key hardening (no keys in logs/code)
- Encryption validation (verify AES-256 in use)

---

## 13. DEPLOYMENT CHECKLIST

- [ ] Anthropic BAA signed
- [ ] Claude API key provisioned + stored in Vault
- [ ] Encryption keys generated (AES-256) + backed up (offline)
- [ ] Database created (if using) + encryption enabled
- [ ] Audit log directory created with restricted permissions (700)
- [ ] Firewall rules: egress to Anthropic API + no inbound
- [ ] Logging, monitoring, and alerting configured
- [ ] Security training completed (team)
- [ ] Documentation finalized (SSP, DPA, runbooks)
- [ ] Staging environment tested end-to-end
- [ ] Production deployment with rollback plan
- [ ] 30-day monitoring period (validate accuracy, latency, errors)

---

## 14. MAINTENANCE & UPDATES

### 14.1 Dependency Updates
- Python packages: quarterly review + security patches immediately
- Claude model version: test new versions in staging; adopt per roadmap
- TLS/encryption libraries: monthly updates

### 14.2 Encryption Key Rotation
- Primary encryption key: rotate quarterly
- API keys: rotate semi-annually or on leak suspicion
- Database encryption key: rotate annually

### 14.3 Audit Log Retention
- Keep encrypted audit logs for 6 years (HIPAA requirement)
- Archive to offline storage after 1 year
- Periodic integrity checks (hash verification)

---

## 15. COST ESTIMATION

| Component | Est. Cost | Notes |
|-----------|-----------|-------|
| **Hardware** | $8,000–12,000 | One-time (i9, RAM, 4090) |
| **Claude API** | $5–50/month | Depends on document volume; ~$0.003 per 1K tokens (Sonnet) |
| **Encryption/Security Tools** | $0–500/month | Vault, monitoring, log storage |
| **Personnel** | $150K–200K/year | Security officer, admin, QA |
| **Total Y1** | ~$180K–260K | Includes hardware amortization |
| **Ongoing (Y2+)** | ~$80K–120K | API + personnel + maintenance |

---

## 16. REFERENCES & STANDARDS

- **HIPAA Security Rule** (45 CFR §164.300–318): Administrative, physical, technical safeguards
- **NIST Cybersecurity Framework**: Risk assessment, secure configuration
- **HL7 FHIR**: If exchanging data with external systems
- **OWASP Top 10**: Application security best practices
- **OpenAI API Security** (Anthropic follows similar practices): Key management, rate limiting

---

## APPENDIX A: Example Extraction Prompt

```
Extract the following fields from the medical document. 
Return ONLY valid JSON with the exact field names below.

Field names and possible aliases in the document:
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
  "confidence_scores": { "claim_id": 0.99, ... }
}
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-02-18  
**Owner**: [Project Lead Name]  
**Classification**: Internal / HIPAA-Regulated
