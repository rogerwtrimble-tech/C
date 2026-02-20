#!/usr/bin/env python3
"""Test anti-fabrication rules for VLM extraction."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.vllm_client import VLLMClient

def test_validation():
    """Test the validation function with various inputs."""
    print("🔧 Testing Anti-Fabrication Validation")
    print("=" * 50)
    
    client = VLLMClient()
    
    # Test cases with fabricated data
    test_cases = [
        {
            "name": "John Doe",
            "expected": "Not found"
        },
        {
            "name": "Jane Smith", 
            "expected": "Not found"
        },
        {
            "name": "patient",
            "expected": "Not found"
        },
        {
            "name": "Unknown",
            "expected": "Not found"
        },
        {
            "name": "N/A",
            "expected": "Not found"
        },
        {
            "name": "Robert Smith",
            "expected": "Robert Smith"  # Should pass
        },
        {
            "name": "ABC123",
            "expected": "ABC123"  # Should pass
        },
        {
            "name": "",
            "expected": "Not found"
        },
        {
            "name": "aaaa",
            "expected": "Not found"  # Repeated characters
        }
    ]
    
    print("Testing fabricated pattern detection:")
    for i, test_case in enumerate(test_cases, 1):
        data = {"patient_name": test_case["name"]}
        validated = client._validate_extracted_data(data)
        result = validated["patient_name"]
        status = "✅" if result == test_case["expected"] else "❌"
        print(f"  {i}. '{test_case['name']}' -> '{result}' {status}")
        if result != test_case["expected"]:
            print(f"     Expected: '{test_case['expected']}'")
    
    print()
    
    # Test date validation
    date_test_cases = [
        {"date": "2024-01-15", "expected": "2024-01-15"},  # Valid
        {"date": "01/15/2024", "expected": "2024-01-15"},  # Valid (will be converted)
        {"date": "Not found", "expected": "Not found"},  # Valid "Not found"
        {"date": "invalid-date", "expected": "Not found"},  # Invalid format
        {"date": "01-15-2024", "expected": "2024-01-15"},  # Valid (will be converted)
        {"date": "", "expected": "Not found"},  # Empty
    ]
    
    print("Testing date validation:")
    for i, test_case in enumerate(date_test_cases, 1):
        data = {"date_of_loss": test_case["date"]}
        validated = client._validate_extracted_data(data)
        # Convert date format if needed
        if validated["date_of_loss"] != "Not found":
            validated["date_of_loss"] = client._convert_date_format(validated["date_of_loss"])
        result = validated["date_of_loss"]
        status = "✅" if result == test_case["expected"] else "❌"
        print(f"  {i}. '{test_case['date']}' -> '{result}' {status}")
        if result != test_case["expected"]:
            print(f"     Expected: '{test_case['expected']}'")
    
    print()
    print("✅ Anti-fabrication validation test completed!")

async def test_vlm_extraction():
    """Test VLM extraction with anti-fabrication rules."""
    print("\n🔧 Testing VLM Extraction with Anti-Fabrication")
    print("=" * 50)
    
    client = VLLMClient()
    
    # Test prompt to ensure it includes anti-fabrication instructions
    prompt = client._build_multimodal_prompt({})
    print("VLM Prompt Preview:")
    print("-" * 30)
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print()
    
    # Check if prompt contains anti-fabrication instructions
    anti_fabrication_keywords = ["never fabricate", "not found", "explicitly visible"]
    has_anti_fabrication = any(keyword.lower() in prompt.lower() for keyword in anti_fabrication_keywords)
    
    if has_anti_fabrication:
        print("✅ Anti-fabrication instructions found in prompt")
    else:
        print("❌ Anti-fabrication instructions missing from prompt")
    
    print("\n✅ VLM extraction test completed!")

if __name__ == "__main__":
    test_validation()
    asyncio.run(test_vlm_extraction())
