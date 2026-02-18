"""Audit logging system with de-identification."""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
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
    
    async def log_extraction(
        self,
        document_id: str,
        document_type: Optional[str],
        fields_extracted: int,
        fields_successful: int,
        latency_ms: float,
        confidence_avg: float,
        status: str,
        model_version: str
    ) -> None:
        """Log an extraction event."""
        entry = AuditLogEntry(
            timestamp=self._format_timestamp(),
            document_id=document_id,
            document_type_detected=document_type,
            fields_extracted=fields_extracted,
            fields_successful=fields_successful,
            extraction_latency_ms=round(latency_ms, 2),
            claude_model_version=model_version,
            confidence_avg=round(confidence_avg, 4),
            status=status
        )
        
        await self._write_entry(entry.model_dump())
    
    async def log_error(
        self,
        document_id: str,
        error_type: str,
        error_message: str,
        fields_recovered: Optional[dict] = None
    ) -> None:
        """Log an error event."""
        entry = {
            "timestamp": self._format_timestamp(),
            "document_id": document_id,
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "fields_recovered": self._deidentify_dict(fields_recovered or {})
        }
        
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
    
    async def _write_entry(self, entry: dict) -> None:
        """Write entry to log file."""
        async with self._lock:
            # Check if we need to rotate to new day's file
            new_log_file = self._get_log_file_path()
            if new_log_file != self.current_log_file:
                self.current_log_file = new_log_file
            
            # Append to log file
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    
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
