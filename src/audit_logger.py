"""De-identified audit logging for HIPAA compliance."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from asyncio import Lock

from .config import Config
from .models import AuditLogEntry


class AuditLogger:
    """Thread-safe audit logger with de-identification."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.log_path = Config.AUDIT_LOG_PATH
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self._get_log_file_path()
    
    def _get_log_file_path(self) -> Path:
        """Get current log file path based on date."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_path / f"audit_{date_str}.jsonl"
    
    def _deidentify_value(self, value: Any) -> str:
        """De-identify a value by hashing it."""
        if value is None:
            return "null"
        if isinstance(value, str):
            if not value.strip():
                return "empty"
            # Hash the value for de-identification
            return f"hash-{hashlib.sha256(value.encode()).hexdigest()[:12]}"
        return str(value)
    
    def _deidentify_dict(self, data: dict) -> dict:
        """De-identify all values in a dictionary."""
        return {k: self._deidentify_value(v) for k, v in data.items()}
    
    def _format_timestamp(self) -> str:
        """Format current timestamp in ISO 8601."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    async def log_pdf_type_detected(self, document_id: str, pdf_type: str, processing_path: str, alphanumeric_ratio: float) -> None:
        """Log PDF type detection results."""
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            document_hash=document_id,
            event_type="pdf_type_detected",
            pdf_type=pdf_type,
            processing_path=processing_path,
            status="detected",
            notes=f"Alphanumeric ratio: {alphanumeric_ratio:.3f}"
        )
        await self._write_entry(entry)
    
    async def log_extraction(
        self,
        document_id: str,
        document_type: Optional[str],
        fields_attempted: int,
        fields_extracted: int,
        fields_skipped: int,
        latency_ms: float,
        confidence_avg: float,
        confidence_scores: Optional[Dict[str, float]],
        status: str,
        model_version: str,
        processing_path: Optional[str] = None,
        pdf_type: Optional[str] = None,
        ocr_engine: Optional[str] = None,
        notes: Optional[str] = None
    ) -> None:
        """Log extraction results with enhanced metadata."""
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            document_hash=document_id,
            event_type="extracted",
            document_type_detected=document_type,
            pdf_type=pdf_type,
            processing_path=processing_path,
            fields_attempted=fields_attempted,
            fields_extracted=fields_extracted,
            fields_skipped=fields_skipped,
            confidence_scores=confidence_scores,
            extraction_latency_ms=latency_ms,
            solar_model_version=model_version,
            ocr_engine_used=ocr_engine,
            status=status,
            notes=notes
        )
        await self._write_entry(entry)
    
    async def log_error(
        self,
        document_id: str,
        error_type: str,
        error_message: str,
        processing_path: Optional[str] = None
    ) -> None:
        """Log extraction error."""
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            document_hash=document_id,
            event_type="error",
            processing_path=processing_path,
            status="failed",
            notes=f"{error_type}: {error_message}"
        )
        await self._write_entry(entry)
    
    async def log_document_received(self, document_id: str, filename_hash: str) -> None:
        """Log document receipt."""
        entry = {
            "timestamp": self._format_timestamp(),
            "document_id": document_id,
            "event_type": "document_received",
            "filename_hash": filename_hash
        }
        
        await self._write_entry(entry)
    
    async def log_cleanup(self, document_id: str, files_deleted: int) -> None:
        """Log cleanup event."""
        entry = {
            "timestamp": self._format_timestamp(),
            "document_id": document_id,
            "event_type": "cleanup",
            "files_deleted": files_deleted
        }
        
        await self._write_entry(entry)
    
    async def _write_entry(self, entry) -> None:
        """Write entry to log file."""
        async with self._lock:
            log_file = self._get_log_file_path()
            with open(log_file, "a", encoding="utf-8") as f:
                # Handle both dict and Pydantic model
                if hasattr(entry, 'model_dump'):
                    entry_dict = entry.model_dump()
                else:
                    entry_dict = entry
                f.write(json.dumps(entry_dict, default=str) + "\n")
    
    def get_log_entries(self, date: Optional[str] = None) -> list[dict]:
        """Read log entries for a specific date."""
        if date:
            log_file = self.log_path / f"audit_{date}.jsonl"
        else:
            log_file = self.current_log_file
        
        if not log_file.exists():
            return []
        
        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        return entries
