"""Pydantic models for visual elements in multimodal extraction."""

from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: int = Field(description="Left x coordinate")
    y1: int = Field(description="Top y coordinate")
    x2: int = Field(description="Right x coordinate")
    y2: int = Field(description="Bottom y coordinate")
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple format."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def area(self) -> int:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> Tuple[int, int]:
        """Calculate center point."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class SignatureElement(BaseModel):
    """Detected signature element."""
    signature_id: str = Field(description="Unique signature identifier")
    bbox: BoundingBox = Field(description="Bounding box coordinates")
    confidence: float = Field(description="Detection confidence (0.0-1.0)")
    page_number: int = Field(description="Page number (1-indexed)")
    label: Optional[str] = Field(None, description="Signature label (e.g., 'patient signature')")
    image_path: Optional[str] = Field(None, description="Path to extracted signature image")
    validated: bool = Field(False, description="Whether signature was validated by YOLO")
    vlm_detected: bool = Field(True, description="Whether detected by VLM")
    yolo_detected: bool = Field(False, description="Whether detected by YOLO")


class StampElement(BaseModel):
    """Detected stamp or seal element."""
    bbox: BoundingBox = Field(description="Bounding box coordinates")
    confidence: float = Field(description="Detection confidence (0.0-1.0)")
    page_number: int = Field(description="Page number (1-indexed)")
    label: Optional[str] = Field(None, description="Stamp label or text")


class HandwrittenNote(BaseModel):
    """Transcribed handwritten note."""
    text: str = Field(description="Transcribed text")
    bbox: Optional[BoundingBox] = Field(None, description="Location of handwritten text")
    confidence: float = Field(description="Transcription confidence (0.0-1.0)")
    page_number: int = Field(description="Page number (1-indexed)")


class TableElement(BaseModel):
    """Extracted table data."""
    bbox: BoundingBox = Field(description="Table bounding box")
    headers: List[str] = Field(default_factory=list, description="Table headers")
    rows: List[List[str]] = Field(default_factory=list, description="Table rows")
    page_number: int = Field(description="Page number (1-indexed)")
    confidence: float = Field(description="Extraction confidence (0.0-1.0)")


class FormField(BaseModel):
    """Detected form field."""
    label: str = Field(description="Field label")
    value: Optional[str] = Field(None, description="Field value")
    bbox: BoundingBox = Field(description="Field bounding box")
    page_number: int = Field(description="Page number (1-indexed)")
    confidence: float = Field(description="Extraction confidence (0.0-1.0)")


class VisualElements(BaseModel):
    """Collection of visual elements detected in document."""
    signatures: List[SignatureElement] = Field(default_factory=list, description="Detected signatures")
    stamps: List[StampElement] = Field(default_factory=list, description="Detected stamps/seals")
    handwritten_notes: List[HandwrittenNote] = Field(default_factory=list, description="Handwritten annotations")
    tables: List[TableElement] = Field(default_factory=list, description="Extracted tables")
    form_fields: List[FormField] = Field(default_factory=list, description="Form fields")
    
    def get_signature_count(self) -> int:
        """Get total number of signatures."""
        return len(self.signatures)
    
    def get_validated_signature_count(self) -> int:
        """Get number of validated signatures."""
        return sum(1 for sig in self.signatures if sig.validated)
    
    def get_signature_pages(self) -> List[int]:
        """Get list of pages with signatures."""
        return sorted(list(set(sig.page_number for sig in self.signatures)))
    
    def has_discrepancies(self) -> bool:
        """Check if there are VLM/YOLO signature discrepancies."""
        for sig in self.signatures:
            if sig.vlm_detected and not sig.yolo_detected:
                return True
        return False
    
    def get_discrepancy_summary(self) -> Dict[str, int]:
        """Get summary of signature detection discrepancies."""
        vlm_only = sum(1 for sig in self.signatures if sig.vlm_detected and not sig.yolo_detected)
        yolo_only = sum(1 for sig in self.signatures if sig.yolo_detected and not sig.vlm_detected)
        both = sum(1 for sig in self.signatures if sig.vlm_detected and sig.yolo_detected)
        
        return {
            "vlm_only": vlm_only,
            "yolo_only": yolo_only,
            "both_detected": both,
            "total_signatures": len(self.signatures)
        }


class MultimodalExtractionMetadata(BaseModel):
    """Metadata for multimodal extraction."""
    total_pages: int = Field(description="Total number of pages processed")
    pages_with_signatures: List[int] = Field(default_factory=list, description="Pages containing signatures")
    pages_with_handwriting: List[int] = Field(default_factory=list, description="Pages with handwritten notes")
    pages_with_tables: List[int] = Field(default_factory=list, description="Pages with tables")
    image_preprocessing_time_ms: float = Field(description="Time spent on image preprocessing")
    vlm_inference_time_ms: float = Field(description="Time spent on VLM inference")
    signature_detection_time_ms: float = Field(description="Time spent on signature detection")
    total_processing_time_ms: float = Field(description="Total processing time")
    vlm_model_used: str = Field(description="VLM model identifier")
    yolo_model_used: Optional[str] = Field(None, description="YOLO model identifier")
    requires_human_review: bool = Field(False, description="Whether document requires human review")
    review_reasons: List[str] = Field(default_factory=list, description="Reasons for human review")
    
    def add_review_reason(self, reason: str) -> None:
        """Add a reason for human review."""
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)
            self.requires_human_review = True
