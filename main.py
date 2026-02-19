"""Main entry point for the Medical Data Extraction System."""

import asyncio
import sys

from src.config import Config
from src.pipeline import ExtractionPipeline
from src.multimodal_pipeline import MultimodalPipeline


async def run():
    """Run the extraction pipeline."""
    print("Medical Data Extraction System")
    print("=" * 40)
    
    # Determine processing mode
    processing_mode = Config.PROCESSING_MODE.lower()
    
    if processing_mode == "vlm" and Config.VLM_ENABLED:
        print(f"Mode: Multimodal VLM (Grade A - 95%+ accuracy)")
        print(f"Model: {Config.VLM_MODEL}")
        pipeline = MultimodalPipeline()
    else:
        print(f"Mode: Legacy Text-Only (Grade B - 85-90% accuracy)")
        print(f"Model: {Config.OLLAMA_MODEL}")
        pipeline = ExtractionPipeline()
    
    print("=" * 40)
    print()
    
    try:
        results = await pipeline.process_directory()
        
        if not results:
            print("No documents processed. Check the 'pdfs' directory for PDF files.")
            return 0
        
        print(f"\n{'='*60}")
        print(f"Processed {len(results)} documents successfully")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(results, 1):
            print(f"Document {i}:")
            print(f"  Processing Path: {result.processing_path}")
            
            # Show PDF metadata if available
            if result.pdf_metadata:
                print(f"  PDF Type: {result.pdf_metadata.get('pdf_type', 'Unknown')}")
                print(f"  Alphanumeric Ratio: {result.pdf_metadata.get('alphanumeric_ratio', 0):.3f}")
            
            # Show extracted fields
            print(f"  Claim ID: {result.claim_id or 'Not found'}")
            print(f"  Patient: {result.patient_name or 'Not found'}")
            print(f"  Document Type: {result.document_type}")
            print(f"  Date of Loss: {result.date_of_loss or 'N/A'}")
            print(f"  Diagnosis: {result.diagnosis or 'N/A'}")
            print(f"  DOB: {result.dob or 'N/A'}")
            print(f"  Provider NPI: {result.provider_npi or 'N/A'}")
            print(f"  Total Billed: {result.total_billed_amount or 'N/A'}")
            
            # Show confidence metrics
            if result.confidence_scores:
                avg_conf = sum(result.confidence_scores.values()) / len(result.confidence_scores)
                print(f"  Avg Confidence: {avg_conf:.1%}")
                print(f"  Fields Extracted: {len([v for v in result.confidence_scores.values() if v > 0])}/8")
            else:
                print("  Avg Confidence: 0%")
                print("  Fields Extracted: 0/8")
            
            # Show visual elements if available (VLM mode)
            if result.visual_elements:
                sig_count = result.visual_elements.get_signature_count()
                validated_count = result.visual_elements.get_validated_signature_count()
                print(f"  Signatures Detected: {sig_count} (validated: {validated_count})")
                
                if result.visual_elements.has_discrepancies():
                    print("  ⚠️  Signature discrepancies detected")
            
            # Show multimodal metadata if available
            if result.multimodal_metadata:
                print(f"  VLM Inference: {result.multimodal_metadata.vlm_inference_time_ms:.0f}ms")
                print(f"  Signature Detection: {result.multimodal_metadata.signature_detection_time_ms:.0f}ms")
                
                if result.multimodal_metadata.requires_human_review:
                    print(f"  🔍 Requires Review: {', '.join(result.multimodal_metadata.review_reasons)}")
            
            # Show latency
            if result.extraction_latency_ms:
                print(f"  Total Latency: {result.extraction_latency_ms:.0f}ms")
            
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        await pipeline.cleanup()


def main():
    """Main entry point."""
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
