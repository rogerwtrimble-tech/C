"""Debug OCR extraction step."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ocr_extractor import OCRExtractor

def debug_ocr():
    """Debug OCR extraction."""
    print("=== Debug OCR Extraction ===")
    
    pdf_path = Path("pdfs/sample_discharge_summary.pdf")
    print(f"Processing: {pdf_path}")
    
    # Initialize OCR extractor
    ocr = OCRExtractor()
    
    # Check available engines
    print(f"\nAvailable engines: {ocr.get_available_engines()}")
    
    # Select engine (use tesseract as default)
    engine = "tesseract"
    print(f"Selected engine: {engine}")
    
    # Extract text
    print(f"\nExtracting text with {engine}...")
    try:
        text = ocr.extract_from_pdf(pdf_path, engine=engine)
        print(f"Extracted text length: {len(text)}")
        print(f"First 500 chars:\n{text[:500]}")
        
        if not text.strip():
            print("ERROR: No text extracted!")
        else:
            print("SUCCESS: Text extracted")
            
    except Exception as e:
        print(f"Error in OCR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ocr()
