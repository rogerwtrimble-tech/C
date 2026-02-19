"""Test empty result creation for failed extractions."""

import asyncio
from pathlib import Path
from src.pipeline import ExtractionPipeline
from src.models import ExtractionResult

async def test_empty_results():
    """Test that empty results are created and saved for failed extractions."""
    pipeline = ExtractionPipeline()
    
    # Create a test empty result
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
        processing_path="scanned_full_ocr",
        pdf_metadata={"pdf_type": "scanned", "alphanumeric_ratio": 0.0},
        extraction_latency_ms=0.0,
        model_version="solar:10.7b"
    )
    
    # Save the empty result
    document_id = "test-empty-doc"
    output_path = await pipeline._save_result(empty_result, document_id)
    
    print(f"Empty result saved to: {output_path}")
    
    # Read and display the saved file
    with open(output_path, 'r') as f:
        content = f.read()
        print("\nSaved JSON content:")
        print(content)
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(test_empty_results())
