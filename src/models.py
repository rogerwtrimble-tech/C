"""Pydantic models for extraction schema validation."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class ExtractionResult(BaseModel):
    """Schema for extracted medical document fields."""
    
    claim_id: str = Field(..., min_length=1, max_length=50, description="Claim identifier")
    patient_name: str = Field(..., min_length=1, max_length=150, description="Patient name")
    document_type: str = Field(default="Unknown", description="Document type")
    date_of_loss: Optional[str] = Field(default=None, description="Date of loss/injury")
    diagnosis: Optional[str] = Field(default=None, max_length=500, description="Diagnosis or ICD-10 code")
    dob: Optional[str] = Field(default=None, description="Date of birth")
    provider_npi: Optional[str] = Field(default=None, description="Provider NPI number")
    total_billed_amount: Optional[str] = Field(default=None, max_length=15, description="Total billed amount")
    confidence_scores: dict[str, float] = Field(default_factory=dict, description="Field-level confidence scores")
    
    @field_validator('claim_id')
    @classmethod
    def validate_claim_id(cls, v: str) -> str:
        """Validate claim_id is alphanumeric with hyphens."""
        if not re.match(r'^[a-zA-Z0-9\-]+$', v):
            raise ValueError('claim_id must be alphanumeric with hyphens only')
        return v
    
    @field_validator('date_of_loss', 'dob')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate date is ISO 8601 or 'Unclear'."""
        if v is None:
            return v
        if v == "Unclear":
            return v
        # Check ISO 8601 format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(f'Date must be ISO 8601 (YYYY-MM-DD) or "Unclear", got: {v}')
        return v
    
    @field_validator('provider_npi')
    @classmethod
    def validate_npi(cls, v: Optional[str]) -> Optional[str]:
        """Validate NPI is 10 digits or alternative format."""
        if v is None:
            return v
        # Allow 10-digit NPI or alternative ID formats
        if re.match(r'^\d{10}$', v):
            return v
        # Allow alternative formats (alphanumeric with hyphens)
        if re.match(r'^[a-zA-Z0-9\-]+$', v):
            return v
        raise ValueError('provider_npi must be 10-digit NPI or valid alternative format')
    
    @field_validator('total_billed_amount')
    @classmethod
    def validate_amount(cls, v: Optional[str]) -> Optional[str]:
        """Validate currency format."""
        if v is None:
            return v
        # Allow $X.XX format or numeric
        if re.match(r'^\$?\d+\.?\d*$', v.replace(',', '')):
            return v
        # Allow raw text if not matching standard format
        return v
    
    @field_validator('confidence_scores')
    @classmethod
    def validate_confidence_scores(cls, v: dict) -> dict:
        """Validate confidence scores are between 0 and 1."""
        for field_name, score in v.items():
            if not 0 <= score <= 1:
                raise ValueError(f'Confidence score for {field_name} must be between 0 and 1')
        return v


class ExtractionError(BaseModel):
    """Schema for extraction error details."""
    
    document_id: str
    error_type: str
    error_message: str
    timestamp: str
    fields_recovered: dict[str, str] = Field(default_factory=dict)


class AuditLogEntry(BaseModel):
    """Schema for audit log entries (de-identified)."""
    
    timestamp: str
    document_id: str
    document_type_detected: Optional[str] = None
    fields_extracted: int
    fields_successful: int
    extraction_latency_ms: float
    claude_model_version: str
    confidence_avg: float
    status: str
