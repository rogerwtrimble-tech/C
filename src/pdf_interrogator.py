"""PDF type detection and interrogation module."""

from pathlib import Path
from typing import Dict
import pdfplumber


class PDFInterrogator:
    """Interrogates PDFs to determine type and processing requirements."""
    
    def __init__(self):
        self.sample_pages = 3  # Number of pages to sample for type detection
        self.native_threshold = 0.30  # Alphanumeric ratio for native PDFs
        self.hybrid_min_threshold = 0.10  # Minimum ratio for hybrid PDFs
    
    def interrogate_pdf(self, pdf_path: Path) -> Dict:
        """
        Determine if PDF is native (text), scanned (image), or hybrid.
        
        Returns:
            Dict with PDF metadata and processing recommendations.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                sample_pages = min(self.sample_pages, total_pages)
                
                # Extract text from sample pages
                extracted_text = ""
                page_details = []
                
                for page_num in range(sample_pages):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text() or ""
                    
                    # Calculate page-level metrics
                    alpha_count = sum(1 for c in page_text if c.isalnum())
                    total_chars = len(page_text)
                    alpha_ratio = alpha_count / total_chars if total_chars > 0 else 0.0
                    
                    # Check for images
                    has_images = bool(page.images)
                    
                    page_details.append({
                        "page_number": page_num + 1,
                        "text_length": total_chars,
                        "alphanumeric_ratio": round(alpha_ratio, 3),
                        "has_images": has_images,
                        "sample_text": page_text[:200] + "..." if len(page_text) > 200 else page_text
                    })
                    
                    extracted_text += page_text
                
                # Calculate overall alphanumeric ratio
                if extracted_text:
                    alpha_count = sum(1 for c in extracted_text if c.isalnum())
                    alpha_ratio = alpha_count / len(extracted_text)
                else:
                    alpha_ratio = 0.0
                
                # Classify PDF type
                if alpha_ratio > self.native_threshold:
                    pdf_type = "native"
                    requires_ocr = False
                    processing_path = "native_direct"
                elif alpha_ratio >= self.hybrid_min_threshold:
                    pdf_type = "hybrid"
                    requires_ocr = True
                    processing_path = "hybrid_selective"
                else:
                    pdf_type = "scanned"
                    requires_ocr = True
                    processing_path = "scanned_full_ocr"
                
                return {
                    "pdf_type": pdf_type,
                    "alphanumeric_ratio": round(alpha_ratio, 3),
                    "requires_ocr": requires_ocr,
                    "processing_path": processing_path,
                    "total_pages": total_pages,
                    "sample_pages_analyzed": sample_pages,
                    "sample_text_length": len(extracted_text),
                    "page_details": page_details,
                    "confidence": self._calculate_detection_confidence(alpha_ratio, total_pages)
                }
                
        except Exception as e:
            # Fallback to scanned if interrogation fails
            return {
                "pdf_type": "unknown",
                "alphanumeric_ratio": 0.0,
                "requires_ocr": True,
                "processing_path": "scanned_full_ocr",
                "total_pages": 0,
                "sample_pages_analyzed": 0,
                "sample_text_length": 0,
                "page_details": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_detection_confidence(self, alpha_ratio: float, total_pages: int) -> float:
        """Calculate confidence in PDF type detection."""
        # High confidence for clear cases
        if alpha_ratio > 0.50 or alpha_ratio < 0.05:
            return 0.95
        elif alpha_ratio > 0.35 or alpha_ratio < 0.15:
            return 0.85
        else:
            # Lower confidence for borderline cases
            return 0.70
    
    def analyze_page_types(self, pdf_path: Path) -> Dict:
        """
        Analyze each page individually for hybrid documents.
        
        Returns:
            Dict with page-by-page classification.
        """
        page_types = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    
                    # Page-level classification
                    if page_text:
                        alpha_count = sum(1 for c in page_text if c.isalnum())
                        alpha_ratio = alpha_count / len(page_text)
                        
                        if alpha_ratio > self.native_threshold:
                            page_type = "native"
                        elif alpha_ratio >= self.hybrid_min_threshold:
                            page_type = "hybrid"
                        else:
                            page_type = "scanned"
                    else:
                        page_type = "scanned"
                    
                    page_types.append({
                        "page_number": page_num + 1,
                        "type": page_type,
                        "alphanumeric_ratio": round(alpha_ratio, 3) if page_text else 0.0,
                        "has_images": bool(page.images),
                        "text_length": len(page_text)
                    })
        except Exception as e:
            return {"error": str(e), "page_types": []}
        
        return {"page_types": page_types}
    
    def should_use_gpu_ocr(self, pdf_path: Path) -> bool:
        """
        Determine if GPU-accelerated OCR should be used.
        
        Returns True if:
        - PDF is scanned or hybrid
        - Has more than 5 pages (worth GPU overhead)
        """
        metadata = self.interrogate_pdf(pdf_path)
        
        if not metadata.get("requires_ocr", False):
            return False
        
        return metadata.get("total_pages", 0) > 5
