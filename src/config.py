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
    
    # Processing Mode
    PROCESSING_MODE: str = os.getenv("PROCESSING_MODE", "vlm")  # 'vlm' or 'text'
    
    # VLM Settings (Multimodal Processing)
    VLM_ENABLED: bool = os.getenv("VLM_ENABLED", "true").lower() == "true"
    VLM_MODEL: str = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    VLM_HOST: str = os.getenv("VLM_HOST", "localhost")
    VLM_PORT: int = int(os.getenv("VLM_PORT", "8000"))
    VLM_QUANTIZATION: str = os.getenv("VLM_QUANTIZATION", "awq")
    VLM_MAX_CONCURRENT_PAGES: int = int(os.getenv("VLM_MAX_CONCURRENT_PAGES", "1"))
    VLM_BATCH_SIZE: int = int(os.getenv("VLM_BATCH_SIZE", "1"))
    VLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLM_GPU_MEMORY_UTILIZATION", "0.75"))
    VLM_MAX_MODEL_LEN: int = int(os.getenv("VLM_MAX_MODEL_LEN", "2048"))
    
    # Ollama SLM settings (Legacy text-only mode)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "solar:10.7b")
    
    # Signature Detection
    SIGNATURE_DETECTION_ENABLED: bool = os.getenv("SIGNATURE_DETECTION_ENABLED", "true").lower() == "true"
    SIGNATURE_MODEL_PATH: Path = Path(os.getenv("SIGNATURE_MODEL_PATH", "models/yolov8_signature.pt"))
    SIGNATURE_CONFIDENCE_THRESHOLD: float = float(os.getenv("SIGNATURE_CONFIDENCE_THRESHOLD", "0.75"))
    SIGNATURE_MIN_SIZE: int = int(os.getenv("SIGNATURE_MIN_SIZE", "50"))
    SIGNATURE_MAX_SIZE: int = int(os.getenv("SIGNATURE_MAX_SIZE", "500"))
    
    # Image Processing
    PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))
    IMAGE_PREPROCESSING: bool = os.getenv("IMAGE_PREPROCESSING", "true").lower() == "true"
    DESKEW_ENABLED: bool = os.getenv("DESKEW_ENABLED", "true").lower() == "true"
    IMAGE_FORMAT: str = os.getenv("IMAGE_FORMAT", "PNG")
    IMAGE_QUALITY: int = int(os.getenv("IMAGE_QUALITY", "95"))
    
    # OCR Settings (Fallback)
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract")
    
    # Paths
    PDF_INPUT_DIR: Path = Path(os.getenv("PDF_INPUT_DIR", "pdfs"))
    RESULTS_OUTPUT_DIR: Path = Path(os.getenv("RESULTS_OUTPUT_DIR", "results"))
    AUDIT_LOG_PATH: Path = Path(os.getenv("AUDIT_LOG_PATH", "logs/audit"))
    SIGNATURE_OUTPUT_DIR: Path = Path(os.getenv("SIGNATURE_OUTPUT_DIR", "results/signatures"))
    TEMP_IMAGE_DIR: Path = Path(os.getenv("TEMP_IMAGE_DIR", "temp/images"))
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "models/cache"))
    
    # Security settings
    SECURE_DELETION_PASSES: int = int(os.getenv("SECURE_DELETION_PASSES", "3"))
    ENCRYPTION_KEY: Optional[bytes] = None
    ENCRYPTION_ENABLED: bool = os.getenv("ENCRYPTION_ENABLED", "false").lower() == "true"
    
    # Confidence & Quality Gates
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.75"))
    SIGNATURE_VALIDATION_REQUIRED: bool = os.getenv("SIGNATURE_VALIDATION_REQUIRED", "true").lower() == "true"
    HUMAN_REVIEW_THRESHOLD: float = float(os.getenv("HUMAN_REVIEW_THRESHOLD", "0.70"))
    AUTO_FLAG_DISCREPANCIES: bool = os.getenv("AUTO_FLAG_DISCREPANCIES", "true").lower() == "true"
    
    # API settings
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "120"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_BACKOFF: float = float(os.getenv("RETRY_BACKOFF", "2.0"))
    
    # Performance Tuning
    ENABLE_BATCH_PROCESSING: bool = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
    PARALLEL_PAGE_PROCESSING: bool = os.getenv("PARALLEL_PAGE_PROCESSING", "true").lower() == "true"
    CACHE_PREPROCESSED_IMAGES: bool = os.getenv("CACHE_PREPROCESSED_IMAGES", "false").lower() == "true"
    CLEANUP_TEMP_FILES: bool = os.getenv("CLEANUP_TEMP_FILES", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PERFORMANCE_METRICS: bool = os.getenv("ENABLE_PERFORMANCE_METRICS", "true").lower() == "true"
    ENABLE_VISUAL_ELEMENT_LOGGING: bool = os.getenv("ENABLE_VISUAL_ELEMENT_LOGGING", "true").lower() == "true"
    
    # Feature Flags
    ENABLE_HANDWRITING_DETECTION: bool = os.getenv("ENABLE_HANDWRITING_DETECTION", "true").lower() == "true"
    ENABLE_LAYOUT_ANALYSIS: bool = os.getenv("ENABLE_LAYOUT_ANALYSIS", "true").lower() == "true"
    ENABLE_TABLE_EXTRACTION: bool = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
    ENABLE_FORM_FIELD_DETECTION: bool = os.getenv("ENABLE_FORM_FIELD_DETECTION", "true").lower() == "true"
    
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
        cls.SIGNATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    @classmethod
    def get_vlm_url(cls) -> str:
        """Get the full vLLM API URL."""
        return f"http://{cls.VLM_HOST}:{cls.VLM_PORT}"


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
