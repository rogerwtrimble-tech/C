#!/usr/bin/env python3
"""Test PDF detection directly"""

from pathlib import Path
import asyncio
from src.multimodal_pipeline import MultimodalPipeline

async def test_pdf_detection():
    """Test PDF detection with manual path"""
    
    # Use explicit Windows path
    pdf_dir = Path("pdfs")
    
    print(f"Testing PDF detection...")
    print(f"Directory: {pdf_dir.absolute()}")
    print(f"Exists: {pdf_dir.exists()}")
    print(f"Is directory: {pdf_dir.is_dir()}")
    
    # Find PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"PDF files found: {len(pdf_files)}")
    
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    if pdf_files:
        print("\nTesting with MultimodalPipeline...")
        pipeline = MultimodalPipeline()
        
        # Test processing one file
        result = await pipeline.process_document(pdf_files[0])
        
        if result:
            print(f"✅ Successfully processed: {pdf_files[0].name}")
            print(f"   Processing path: {result.processing_path}")
            print(f"   Document type: {result.document_type}")
            if result.claim_id:
                print(f"   Claim ID: {result.claim_id}")
            if result.patient_name:
                print(f"   Patient: {result.patient_name}")
        else:
            print(f"❌ Failed to process: {pdf_files[0].name}")
        
        await pipeline.cleanup()
    else:
        print("❌ No PDF files found to test")

if __name__ == "__main__":
    asyncio.run(test_pdf_detection())
