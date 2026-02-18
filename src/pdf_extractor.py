"""PDF extraction module for text and image extraction."""

import base64
import hashlib
from io import BytesIO
from pathlib import Path
import pdfplumber
from .pdf_interrogator import PDFInterrogator


class PDFExtractor:
    """Extract text and images from PDF documents."""
    
    def __init__(self):
        self.max_pages_for_single_call = 20
        self.interrogator = PDFInterrogator()
    
    def get_document_id(self, pdf_path: Path) -> str:
        """Generate a unique document ID from file hash."""
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        return f"doc-{file_hash}"
    
    def extract_text(self, pdf_path: Path) -> tuple[str, bool]:
        """
        Extract text from PDF using pdfplumber.
        
        Returns:
            Tuple of (extracted_text, has_images)
        """
        text_parts = []
        has_images = False
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                # Check for images
                if page.images:
                    has_images = True
        
        full_text = "\n\n".join(text_parts)
        return full_text, has_images
    
    def extract_images(self, pdf_path: Path) -> list[tuple[str, str, bytes]]:
        """
        Extract images from PDF for vision API processing.
        
        Returns:
            List of tuples: (image_id, media_type, image_bytes)
        """
        images = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Convert page to image for vision processing
                im = page.to_image(resolution=150)
                img_bytes = BytesIO()
                im.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                
                image_id = f"page_{page_num + 1}"
                images.append((image_id, "image/png", img_bytes.read()))
        
        return images
    
    def extract_images_base64(self, pdf_path: Path) -> list[tuple[str, str, str]]:
        """
        Extract images as base64 for Claude Vision API.
        
        Returns:
            List of tuples: (image_id, media_type, base64_string)
        """
        images = self.extract_images(pdf_path)
        result = []
        
        for image_id, media_type, img_bytes in images:
            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            result.append((image_id, media_type, b64))
        
        return result
    
    def interrogate_pdf(self, pdf_path: Path) -> dict:
        """Interrogate PDF to determine type and processing requirements."""
        return self.interrogator.interrogate_pdf(pdf_path)
    
    def extract_text_clean(self, pdf_path: Path) -> str:
        """Extract and clean text from native PDFs."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    # Clean and normalize text
                    cleaned = self._clean_text(page_text)
                    text_parts.append(f"--- Page {page_num + 1} ---\n{cleaned}")
        
        return "\n\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excess whitespace
        text = ' '.join(text.split())
        
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common OCR confusion
        
        return text.strip()
    
    def chunk_large_document(self, pdf_path: Path) -> list[tuple[int, int]]:
        """
        Determine chunk boundaries for large documents.
        
        Returns:
            List of (start_page, end_page) tuples
        """
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        if total_pages <= self.max_pages_for_single_call:
            return [(0, total_pages)]
        
        chunks = []
        for i in range(0, total_pages, self.max_pages_for_single_call):
            end = min(i + self.max_pages_for_single_call, total_pages)
            chunks.append((i, end))
        
        return chunks
    
    def get_page_types(self, pdf_path: Path) -> list:
        """Get page-by-page type classification for hybrid documents."""
        analysis = self.interrogator.analyze_page_types(pdf_path)
        return analysis.get('page_types', [])
    
    def get_page_count(self, pdf_path: Path) -> int:
        """Get total page count of PDF."""
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
