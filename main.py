"""Main entry point for the Medical Data Extraction System."""

import asyncio
import sys

from src.pipeline import ExtractionPipeline


async def run():
    """Run the extraction pipeline."""
    print("Medical Data Extraction System")
    print("=" * 40)
    
    pipeline = ExtractionPipeline()
    
    try:
        results = await pipeline.process_directory()
        
        if not results:
            print("No documents processed. Check the 'pdfs' directory for PDF files.")
            return 0
        
        print(f"\nProcessed {len(results)} documents successfully\n")
        
        for i, result in enumerate(results, 1):
            print(f"Document {i}:")
            print(f"  Claim ID: {result.claim_id}")
            print(f"  Patient: {result.patient_name}")
            print(f"  Document Type: {result.document_type}")
            print(f"  Date of Loss: {result.date_of_loss or 'N/A'}")
            print(f"  Diagnosis: {result.diagnosis or 'N/A'}")
            print(f"  DOB: {result.dob or 'N/A'}")
            print(f"  Provider NPI: {result.provider_npi or 'N/A'}")
            print(f"  Total Billed: {result.total_billed_amount or 'N/A'}")
            
            if result.confidence_scores:
                avg_conf = sum(result.confidence_scores.values()) / len(result.confidence_scores)
                print(f"  Avg Confidence: {avg_conf:.1%}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    finally:
        await pipeline.cleanup()


def main():
    """Main entry point."""
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
