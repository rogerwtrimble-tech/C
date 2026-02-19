"""Debug the process_document method step by step."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import ExtractionPipeline
from src.config import Config

async def debug_process_document():
    """Debug process_document step by step."""
    print("=== Debug Process Document ===")
    
    pdf_path = Path("pdfs/sample_discharge_summary.pdf")
    print(f"Processing: {pdf_path}")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Step 1: Generate document ID
    print("\n=== Step 1: Document ID ===")
    document_id = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:12]
    print(f"Document ID: {document_id}")
    
    # Step 2: Interrogate PDF
    print("\n=== Step 2: PDF Interrogation ===")
    try:
        pdf_metadata = pipeline.pdf_extractor.interrogate_pdf(pdf_path)
        print(f"PDF Metadata: {pdf_metadata}")
        processing_path = pdf_metadata["processing_path"]
        print(f"Processing Path: {processing_path}")
    except Exception as e:
        print(f"Error in interrogation: {e}")
        import traceback
        traceback.print_exc()
        await pipeline.cleanup()
        return
    
    # Step 3: Process based on type
    print(f"\n=== Step 3: Processing ({processing_path}) ===")
    try:
        if processing_path == "native_direct":
            print("Using native direct processing")
            result, latency = await pipeline._process_native_pdf(pdf_path, document_id)
            ocr_engine = None
        elif processing_path == "scanned_full_ocr":
            print("Using scanned full OCR processing")
            result, latency, ocr_engine = await pipeline._process_scanned_pdf(pdf_path, document_id)
        elif processing_path == "hybrid_selective":
            print("Using hybrid selective processing")
            result, latency, ocr_engine = await pipeline._process_hybrid_pdf(pdf_path, document_id)
        else:
            print("Using fallback OCR processing")
            result, latency, ocr_engine = await pipeline._process_scanned_pdf(pdf_path, document_id)
        
        print(f"Result: {result}")
        print(f"Latency: {latency}ms")
        print(f"OCR Engine: {ocr_engine}")
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        await pipeline.cleanup()
        return
    
    # Step 4: Save result
    print(f"\n=== Step 4: Save Result ===")
    if result:
        print("Result is not None, saving...")
        await pipeline._save_result(result, document_id)
    else:
        print("Result is None, creating empty result...")
        empty_result = ExtractionResult(
            claim_id=None,
            patient_name=None,
            document_type="Unknown",
            date_of_loss=None,
            diagnosis=None,
            dob=None,
            provider_npi=None,
            total_billed_amount=None,
            confidence_scores={},
            processing_path=processing_path,
            pdf_metadata=pdf_metadata,
            extraction_latency_ms=0.0,
            model_version=Config.OLLAMA_MODEL
        )
        await pipeline._save_result(empty_result, document_id)
    
    await pipeline.cleanup()

if __name__ == "__main__":
    import hashlib
    from src.models import ExtractionResult
    asyncio.run(debug_process_document())
