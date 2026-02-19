"""Debug main.py to see why PDFs aren't being processed."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import ExtractionPipeline
from config import Config

async def debug_main():
    """Debug the main pipeline execution."""
    print("=== Debug Main Pipeline ===")
    
    # Check PDF directory
    pdf_dir = Config.PDF_INPUT_DIR
    print(f"PDF directory: {pdf_dir}")
    print(f"Directory exists: {pdf_dir.exists()}")
    
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"PDF files found: {len(pdf_files)}")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
    
    # Initialize pipeline
    print("\n=== Initializing Pipeline ===")
    pipeline = ExtractionPipeline()
    print("Pipeline initialized")
    
    # Try to process directory
    print("\n=== Processing Directory ===")
    input_dir = Config.PDF_INPUT_DIR
    print(f"Input dir: {input_dir}")
    print(f"Input dir type: {type(input_dir)}")
    
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"PDF files in process_directory: {len(pdf_files)}")
    
    # Try processing one file directly
    if pdf_files:
        print(f"\n=== Processing Single File ===")
        first_pdf = pdf_files[0]
        print(f"Processing: {first_pdf}")
        
        try:
            result = await pipeline.process_document(first_pdf)
            print(f"Result: {result}")
            if result:
                print(f"  Processing Path: {result.processing_path}")
                print(f"  PDF Type: {result.pdf_metadata.get('pdf_type') if result.pdf_metadata else 'None'}")
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No PDF files to process")
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_main())
