import asyncio
from src.pipeline import ExtractionPipeline
from pathlib import Path

async def test():
    pipeline = ExtractionPipeline()
    
    # Test with empty text (scanned PDF)
    pdf_path = Path('pdfs/sample_discharge_summary.pdf')
    
    # Get document ID
    document_id = pipeline.pdf_extractor.get_document_id(pdf_path)
    print(f'Document ID: {document_id}')
    
    # Extract text
    text, has_images = pipeline.pdf_extractor.extract_text(pdf_path)
    print(f'Has images: {has_images}')
    print(f'Text length: {len(text)}')
    
    # Check if vision needed
    use_vision = pipeline.pdf_extractor.should_use_vision(pdf_path, text)
    print(f'Use vision: {use_vision}')
    
    if use_vision:
        # Try enhanced extraction
        enhanced_text = pipeline._enhance_text_extraction(pdf_path, text)
        print(f'Enhanced text length: {len(enhanced_text)}')
        
        # Try extraction with enhanced text
        result, latency = await pipeline._process_with_text(enhanced_text, document_id)
        
        if result:
            print('Success with enhanced text!')
            print(f'Claim ID: {result.claim_id}')
            print(f'Patient: {result.patient_name}')
            print(f'DOB: {result.dob}')
        else:
            print('Still failed - need OCR')
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(test())
