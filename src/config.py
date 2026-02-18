"""Configuration management for the extraction system."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()


class Config:
    """Application configuration from environment variables."""
    
    # Ollama SLM settings (local inference via WSL 2.0 Docker)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "solar:10.7b")
    
    # Paths
    PDF_INPUT_DIR: Path = Path(os.getenv("PDF_INPUT_DIR", "pdfs"))
    RESULTS_OUTPUT_DIR: Path = Path(os.getenv("RESULTS_OUTPUT_DIR", "results"))
    AUDIT_LOG_PATH: Path = Path(os.getenv("AUDIT_LOG_PATH", "logs/audit"))
    
    # Security settings
    SECURE_DELETION_PASSES: int = int(os.getenv("SECURE_DELETION_PASSES", "3"))
    ENCRYPTION_KEY: Optional[bytes] = None
    
    # API settings
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))  # Lower for local GPU
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "120"))  # Longer for local inference
    
    # Field aliases for extraction prompt
    FIELD_ALIASES: dict[str, list[str]] = {
        "claim_id": ["claim_id", "claim_number", "case_id", "docket_no", "control_no", "file_no", "tpa_ref"],
        "patient_name": ["patient_name", "insured", "claimant", "injured_worker", "petitioner", "ee_name", "member", "patient name"],
        "document_type": ["document_type", "doc_type", "form_name", "notice_title", "exhibit_type", "report_type"],
        "date_of_loss": ["date_of_injury", "doi", "date_of_loss", "accident_date", "incident_date", "injury_dt"],
        "diagnosis": ["icd_10", "diagnosis_code", "dx_code", "primary_dx", "nature_of_injury"],
        "dob": ["date_of_birth", "birth_date", "dob"],
        "provider_npi": ["npi", "billing_provider_npi", "provider_id"],
        "total_billed_amount": ["total_charges", "billed_amount", "total_due"],
    }
    
    @classmethod
    def get_encryption_key(cls) -> Optional[bytes]:
        """Get and cache the encryption key."""
        if cls.ENCRYPTION_KEY is None:
            key_b64 = os.getenv("ENCRYPTION_KEY", "")
            if key_b64:
                cls.ENCRYPTION_KEY = base64.b64decode(key_b64)
        return cls.ENCRYPTION_KEY
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create required directories if they don't exist."""
        cls.RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.AUDIT_LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        # Ollama runs locally - no API key required
        # Validation happens at runtime via health check
        return True
    
    @classmethod
    def get_ollama_url(cls) -> str:
        """Get the full Ollama API URL."""
        return f"http://{cls.OLLAMA_HOST}:{cls.OLLAMA_PORT}"


# Document types for validation
DOCUMENT_TYPES = [
    "workers_compensation_claim",
    "medical_bill",
    "explanation_of_benefits",
    "first_report_of_injury",
    "medical_report",
    "prescription",
    "lab_report",
    "imaging_report",
    "discharge_summary",
    "progress_note",
    "Unknown",
]
