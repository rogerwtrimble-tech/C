"""OCR fallback for scanned PDFs with multiple engine support."""

from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class OCRExtractor:
    """Extract text from images using multiple OCR engines."""
    
    def __init__(self, preferred_engine: str = "tesseract"):
        self.preferred_engine = preferred_engine
        self.easyocr_reader = None
        self.paddleocr_reader = None
        
        if not PYTESSERACT_AVAILABLE and preferred_engine == "tesseract":
            print("Warning: pytesseract not installed. Tesseract OCR not available.")
    
    def extract_from_pdf(self, pdf_path: Path, dpi: int = 300, engine: Optional[str] = None) -> str:
        """Extract text from PDF using OCR.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF to image conversion
            engine: OCR engine to use (tesseract, easyocr, paddleocr)
            
        Returns:
            Extracted text string
        """
        engine = engine or self.preferred_engine
        
        if engine == "tesseract" and PYTESSERACT_AVAILABLE:
            return self._extract_with_tesseract(pdf_path, dpi)
        elif engine == "easyocr" and EASYOCR_AVAILABLE:
            return self._extract_with_easyocr(pdf_path, dpi)
        elif engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            return self._extract_with_paddleocr(pdf_path, dpi)
        else:
            # Fallback to available engine
            if PYTESSERACT_AVAILABLE:
                return self._extract_with_tesseract(pdf_path, dpi)
            else:
                print("No OCR engine available")
                return ""
    
    def _extract_with_tesseract(self, pdf_path: Path, dpi: int) -> str:
        """Extract text using Tesseract OCR."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Convert page to image
                im = page.to_image(resolution=dpi)
                img_bytes = BytesIO()
                im.save(img_bytes, format='PNG')
                
                # OCR the image
                image = Image.open(img_bytes)
                page_text = pytesseract.image_to_string(image)
                
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_easyocr(self, pdf_path: Path, dpi: int) -> str:
        """Extract text using EasyOCR (GPU-accelerated)."""
        if self.easyocr_reader is None:
            self.easyocr_reader = easyocr.Reader(['en'])
        
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Convert page to image
                im = page.to_image(resolution=dpi)
                img_bytes = BytesIO()
                im.save(img_bytes, format='PNG')
                
                # OCR the image
                result = self.easyocr_reader.readtext(img_bytes.getvalue())
                page_text = '\n'.join([text for _, text, _ in result])
                
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_paddleocr(self, pdf_path: Path, dpi: int) -> str:
        """Extract text using PaddleOCR (fastest GPU option)."""
        if self.paddleocr_reader is None:
            self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
        
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Convert page to image
                im = page.to_image(resolution=dpi)
                img_bytes = BytesIO()
                im.save(img_bytes, format='PNG')
                
                # OCR the image
                result = self.paddleocr_reader.ocr(img_bytes.getvalue(), cls=True)
                page_text = '\n'.join([line[1][0] for line in result[0]]) if result[0] else ""
                
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        engines = []
        if PYTESSERACT_AVAILABLE:
            engines.append("tesseract")
        if EASYOCR_AVAILABLE:
            engines.append("easyocr")
        if PADDLEOCR_AVAILABLE:
            engines.append("paddleocr")
        return engines
    
    def benchmark_engines(self, pdf_path: Path) -> Dict[str, Dict]:
        """Benchmark all available OCR engines on a PDF."""
        results = {}
        
        for engine in self.get_available_engines():
            import time
            start_time = time.time()
            text = self.extract_from_pdf(pdf_path, engine=engine)
            end_time = time.time()
            
            results[engine] = {
                "text_length": len(text),
                "processing_time": end_time - start_time,
                "sample_text": text[:200] + "..." if len(text) > 200 else text
            }
        
        return results
