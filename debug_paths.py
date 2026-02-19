"""Debug script to check PDF directory and Tesseract installation."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config

def check_pdf_directory():
    """Check if PDF directory exists and contains files."""
    print("=== PDF Directory Check ===")
    
    pdf_dir = Config.PDF_INPUT_DIR
    print(f"Configured PDF directory: {pdf_dir}")
    print(f"Absolute path: {pdf_dir.absolute()}")
    print(f"Directory exists: {pdf_dir.exists()}")
    
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"PDF files found: {len(pdf_files)}")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file.name} ({pdf_file.stat().st_size} bytes)")
    else:
        print("ERROR: PDF directory does not exist!")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Parent directory contents: {list(Path.cwd().parent.iterdir())}")

def check_tesseract():
    """Check Tesseract installation."""
    print("\n=== Tesseract Check ===")
    
    # Check if pytesseract is installed
    try:
        import pytesseract
        print("✓ pytesseract Python package is installed")
    except ImportError:
        print("✗ pytesseract Python package is NOT installed")
        print("  Install with: pip install pytesseract")
        return False
    
    # Check if Tesseract executable is in PATH
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract executable found: version {version}")
        
        # Test Tesseract on a simple image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image with text
            img = Image.new('RGB', (100, 30), color='white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            draw.text((5, 5), "TEST", fill='black')
            
            # OCR the test image
            text = pytesseract.image_to_string(img)
            if "TEST" in text:
                print("✓ Tesseract OCR test passed")
            else:
                print(f"✗ Tesseract OCR test failed. Got: '{text.strip()}'")
        except Exception as e:
            print(f"✗ Tesseract OCR test failed: {e}")
            
    except pytesseract.TesseractNotFoundError:
        print("✗ Tesseract executable NOT found in PATH")
        print("  Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Add to PATH: C:\\Program Files\\Tesseract-OCR")
        return False
    except Exception as e:
        print(f"✗ Error checking Tesseract: {e}")
        return False
    
    return True

def check_environment():
    """Check environment variables."""
    print("\n=== Environment Variables ===")
    
    env_vars = [
        "OLLAMA_HOST",
        "OLLAMA_PORT", 
        "OLLAMA_MODEL",
        "PDF_INPUT_DIR",
        "RESULTS_OUTPUT_DIR",
        "OCR_ENGINE"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: (not set)")

if __name__ == "__main__":
    check_pdf_directory()
    check_tesseract()
    check_environment()
