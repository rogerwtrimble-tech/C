"""Debug ExtractionResult validation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import ExtractionResult

def debug_validation():
    """Debug ExtractionResult validation."""
    print("=== Debug ExtractionResult Validation ===")
    
    # Sample extracted data (missing required fields)
    extracted_data = {
        'patient_name': 'Robert Smith',
        'dob': '1955-12-10',
        'document_type': 'HOSPITAL DISCHARGE SUMMARY',
        'diagnosis': 'Primary: Pneumonia (J18.9)\nsecondary: COPD',
        'confidence_scores': {
            'patient_name': 0.99,
            'dob': 0.95,
            'document_type': 0.7,
            'diagnosis': 0.8
        }
    }
    
    print(f"Extracted data keys: {extracted_data.keys()}")
    
    # Try to create ExtractionResult
    print("\n=== Creating ExtractionResult ===")
    try:
        result = ExtractionResult(**extracted_data)
        print("SUCCESS: ExtractionResult created")
        print(f"Claim ID: {result.claim_id}")
        print(f"Patient: {result.patient_name}")
        print(f"Document Type: {result.document_type}")
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e)}")
        
        # Try with all fields
        print("\n=== Trying with all fields ===")
        full_data = extracted_data.copy()
        full_data.update({
            'claim_id': None,
            'date_of_loss': None,
            'provider_npi': None,
            'total_billed_amount': None
        })
        
        try:
            result = ExtractionResult(**full_data)
            print("SUCCESS: ExtractionResult created with all fields")
            print(f"Claim ID: {result.claim_id}")
            print(f"Patient: {result.patient_name}")
        except Exception as e2:
            print(f"Still failed: {e2}")

if __name__ == "__main__":
    debug_validation()
