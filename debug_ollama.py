"""Debug Ollama extraction step."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ollama_client import OllamaClient

async def debug_ollama():
    """Debug Ollama extraction."""
    print("=== Debug Ollama Extraction ===")
    
    # Sample OCR text
    ocr_text = """--- Page 1 ---
HOSPITAL DISCHARGE SUMMARY

Patient Name: Robert Smith

DOB: 12/10/1955

Admission Date: 11/10/2023
Discharge Date: 11/15/2023

Facility: General Hospital

Attending Physician: Dr. Carol White

Diagnosis:
Primary: Pneumonia (J18.9)
secondary: COPD

Hospital Course:
Patient admitted with fever and cough. Started on IV antibiotics.
Improved significantly over 3 days. Switched to oral antibiotics.

Discharge Instructions:
- Complete course of antibiotics.
- Follow up with PCP in 1 week"""
    
    print(f"Text length: {len(ocr_text)}")
    print(f"First 200 chars:\n{ocr_text[:200]}")
    
    # Initialize Ollama client
    client = OllamaClient()
    
    # Check health
    print("\n=== Checking Ollama Health ===")
    try:
        healthy = await client.check_health()
        print(f"Ollama healthy: {healthy}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Extract
    print("\n=== Extracting with Ollama ===")
    document_id = "test-doc"
    try:
        result, latency = await client.extract(ocr_text, document_id)
        print(f"Result: {result}")
        print(f"Latency: {latency}ms")
        
        if result:
            print(f"Claim ID: {result.claim_id}")
            print(f"Patient: {result.patient_name}")
            print(f"DOB: {result.dob}")
            print(f"Diagnosis: {result.diagnosis}")
        else:
            print("ERROR: No result returned!")
            
    except Exception as e:
        print(f"Error in extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ollama())
