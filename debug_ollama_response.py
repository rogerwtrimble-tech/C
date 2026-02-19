"""Debug Ollama raw response."""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ollama_client import OllamaClient

async def debug_ollama_response():
    """Debug Ollama raw response."""
    print("=== Debug Ollama Raw Response ===")
    
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
    
    # Initialize Ollama client
    client = OllamaClient()
    
    # Build prompt
    prompt = client._build_extraction_prompt(ocr_text)
    print(f"Prompt length: {len(prompt)}")
    print(f"First 500 chars of prompt:\n{prompt[:500]}")
    
    # Call Ollama API directly
    print("\n=== Calling Ollama API ===")
    try:
        import aiohttp
        import time
        from aiohttp import ClientTimeout, ClientError
        
        timeout = ClientTimeout(total=120)
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{client.base_url}/api/generate",
                json={
                    "model": client.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                        "top_p": 0.9,
                    }
                }
            ) as response:
                print(f"Response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    return
                
                result = await response.json()
                latency_ms = (time.time() - start_time) * 1000
                
                print(f"Latency: {latency_ms}ms")
                print(f"Response keys: {result.keys()}")
                
                response_text = result.get("response", "")
                print(f"Response text length: {len(response_text)}")
                print(f"Raw response:\n{response_text}")
                
                # Try to parse JSON
                print("\n=== Parsing JSON ===")
                extracted_data = client._parse_json_response(response_text)
                print(f"Parsed data: {extracted_data}")
                
                if extracted_data:
                    print("JSON parsed successfully!")
                    print(f"Keys: {extracted_data.keys()}")
                else:
                    print("Failed to parse JSON from response")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ollama_response())
