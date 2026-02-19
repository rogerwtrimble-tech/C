"""Test the intelligent PDF extraction system."""

import asyncio
from pathlib import Path
from src.pipeline import ExtractionPipeline

async def test_pdf_intelligence():
    """Test PDF type detection and intelligent processing."""
    pipeline = ExtractionPipeline()
    
    # Test files
    test_files = [
        "pdfs/sample_discharge_summary.pdf",  # Scanned
        "pdfs/sample_lab_result.pdf",         # Unknown type
        "pdfs/sample_medical_history.pdf"     # Unknown type
    ]
    
    for pdf_file in test_files:
        pdf_path = Path(pdf_file)
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing: {pdf_path.name}")
        print(f"{'='*60}")
        
        # Interrogate PDF first
        metadata = pipeline.pdf_extractor.interrogate_pdf(pdf_path)
        print(f"📄 PDF Type: {metadata['pdf_type']}")
        print(f"🔤 Alphanumeric Ratio: {metadata['alphanumeric_ratio']:.3f}")
        print(f"⚙️  Processing Path: {metadata['processing_path']}")
        print(f"📄 Total Pages: {metadata['total_pages']}")
        print(f"🔍 Confidence: {metadata['confidence']:.2f}")
        
        # Process the document
        result = await pipeline.process_document(pdf_path)
        
        if result:
            print(f"\n✅ Extraction Successful!")
            print(f"  Claim ID: {result.claim_id}")
            print(f"  Patient: {result.patient_name}")
            print(f"  Document Type: {result.document_type}")
            print(f"  DOB: {result.dob}")
            print(f"  Date of Loss: {result.date_of_loss}")
            print(f"  Diagnosis: {result.diagnosis}")
            print(f"  Provider NPI: {result.provider_npi}")
            print(f"  Total Billed: {result.total_billed_amount}")
            print(f"  Processing Path: {result.processing_path}")
            print(f"  Latency: {result.extraction_latency_ms:.0f}ms")
            print(f"  Avg Confidence: {result.get_average_confidence():.2%}")
            
            # Check low confidence fields
            low_conf = result.get_low_confidence_fields(0.75)
            if low_conf:
                print(f"  ⚠️  Low Confidence Fields: {', '.join(low_conf)}")
        else:
            print(f"\n❌ Extraction Failed")
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(test_pdf_intelligence())
