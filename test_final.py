import asyncio
from src.pipeline import ExtractionPipeline
from pathlib import Path

async def test():
    pipeline = ExtractionPipeline()
    
    # Test with scanned PDF
    pdf_path = Path('pdfs/sample_discharge_summary.pdf')
    
    print(f'Processing {pdf_path}...')
    
    result = await pipeline.process_document(pdf_path)
    
    if result:
        print('Success! Result:')
        print(f'  Claim ID: {result.claim_id}')
        print(f'  Patient: {result.patient_name}')
        print(f'  DOB: {result.dob}')
        print(f'  Document Type: {result.document_type}')
        print(f'  Date of Loss: {result.date_of_loss}')
        print(f'  Diagnosis: {result.diagnosis}')
        print(f'  Provider NPI: {result.provider_npi}')
        print(f'  Total Billed: {result.total_billed_amount}')
    else:
        print('Failed to extract data')
    
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(test())
