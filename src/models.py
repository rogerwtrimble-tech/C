"""Pydantic models for medical data extraction."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

from .visual_models import VisualElements, MultimodalExtractionMetadata


class ExtractionResult(BaseModel):
    """Result of medical data extraction from a PDF document."""
    
    # Required fields
    claim_id: Optional[str] = Field(None, description="Claim identifier")
    patient_name: Optional[str] = Field(None, description="Patient name")
    
    # Optional fields
    document_type: str = Field("Unknown", description="Document type")
    date_of_loss: Optional[str] = Field(None, description="Date of loss/injury")
    diagnosis: Optional[str] = Field(None, description="Diagnosis code")
    dob: Optional[str] = Field(None, description="Date of birth")
    provider_npi: Optional[str] = Field(None, description="Provider NPI")
    total_billed_amount: Optional[str] = Field(None, description="Total billed amount")
    
    # Metadata
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores per field")
    processing_path: str = Field(description="Processing path taken (native, scanned, hybrid, multimodal_vlm)")
    pdf_metadata: Optional[Dict[str, Any]] = Field(None, description="PDF interrogation metadata")
    extraction_latency_ms: Optional[float] = Field(None, description="Extraction latency in milliseconds")
    model_version: str = Field(description="Model version used for extraction")
    
    # Visual elements (multimodal only)
    visual_elements: Optional[VisualElements] = Field(None, description="Visual elements detected by VLM")
    multimodal_metadata: Optional[MultimodalExtractionMetadata] = Field(None, description="Multimodal processing metadata")
    
    @field_validator("claim_id")
    def validate_claim_id(cls, v):
        if v and len(v) > 50:
            raise ValueError("claim_id must be 1-50 characters")
        return v
    
    @field_validator("dob", "date_of_loss", mode="before")
    def validate_dates(cls, v):
        if v and v != "Unclear" and v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("dates must be ISO 8601 (YYYY-MM-DD)")
        return v
    
    def get_low_confidence_fields(self, threshold: float = 0.75) -> list[str]:
        """Get fields with confidence below threshold."""
        low_confidence = []
        for field, confidence in self.confidence_scores.items():
            if confidence < threshold:
                low_confidence.append(field)
        return low_confidence
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across all fields."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
    
    def has_required_fields(self) -> bool:
        """Check if required fields are present."""
        return bool(self.claim_id and self.patient_name)


class ExtractionError(BaseModel):
    """Error information for failed extractions."""
    
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Error message")
    document_id: Optional[str] = Field(None, description="Document ID")
    processing_path: Optional[str] = Field(None, description="Processing path attempted")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AuditLogEntry(BaseModel):
    """De-identified audit log entry."""
    
    timestamp: datetime = Field(description="Event timestamp")
    document_hash: str = Field(description="SHA-256 hash of document")
    event_type: str = Field(description="Type of event (received, extracted, error, cleaned)")
    document_type_detected: Optional[str] = Field(None, description="Detected document type")
    pdf_type: Optional[str] = Field(None, description="PDF type (native, scanned, hybrid)")
    processing_path: Optional[str] = Field(None, description="Processing path taken")
    fields_attempted: Optional[int] = Field(None, description="Number of fields attempted")
    fields_extracted: Optional[int] = Field(None, description="Number of fields successfully extracted")
    fields_skipped: Optional[int] = Field(None, description="Number of fields skipped")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores")
    extraction_latency_ms: Optional[float] = Field(None, description="Extraction latency")
    solar_model_version: Optional[str] = Field(None, description="SOLAR model version")
    ocr_engine_used: Optional[str] = Field(None, description="OCR engine used")
    status: str = Field(description="Status (success, partial, failed)")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
