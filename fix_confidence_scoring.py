#!/usr/bin/env python3
"""Fix confidence scoring issues in VLM extraction."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.vllm_client import VLLMClient
from src.config import Config

def analyze_confidence_issue():
    """Analyze the confidence scoring problem."""
    print("🔍 Confidence Scoring Issue Analysis")
    print("=" * 50)
    
    client = VLLMClient()
    
    # Current prompt
    current_prompt = client._build_multimodal_prompt({})
    print("Current Prompt Template:")
    print("-" * 30)
    print(current_prompt)
    print()
    
    # Check if confidence scores are hardcoded to 0
    if '"claim_id":0' in current_prompt:
        print("❌ ISSUE FOUND: Confidence scores are hardcoded to 0 in prompt template")
        print("   The VLM is literally copying the template instead of calculating confidence")
    else:
        print("✅ Confidence scores appear to be dynamic")
    
    print()

def provide_solutions():
    """Provide multiple solutions for the confidence issue."""
    print("🛠️  Available Solutions")
    print("=" * 50)
    
    print("""
OPTION 1: Remove Confidence Scoring (Quick Fix)
- Remove confidence_scores from the prompt entirely
- Focus on data extraction accuracy
- Set default confidence of 0.8 for found fields, 0.0 for "Not found"
- Fastest solution, maintains functionality

OPTION 2: Improve Confidence Instructions (Medium Fix)  
- Update prompt to include confidence calculation instructions
- Add examples of proper confidence scoring
- May require more tokens (context window concern)

OPTION 3: Post-Processing Confidence Calculation (Advanced Fix)
- Calculate confidence based on data characteristics
- Use heuristics like field length, format validation, etc.
- Most accurate but requires more development

OPTION 4: Model Retraining/Prompt Engineering (Long-term Fix)
- Use a larger model with better confidence calibration
- Implement few-shot prompting with confidence examples
- Best quality but requires more resources
""")

def create_option1_fix():
    """Create the quick fix for Option 1."""
    print("🔧 Creating Option 1 Fix (Remove Confidence Scoring)")
    print("-" * 50)
    
    # Read current vllm_client.py
    vllm_file = Path("src/vllm_client.py")
    content = vllm_file.read_text()
    
    # New prompt without confidence scores
    new_prompt = '''"""Extract medical data from document. CRITICAL: Only extract data explicitly visible in the document. Never fabricate or guess values. If information is not found, use "Not found". JSON only:

{"claim_id":"Not found","patient_name":"Not found","document_type":"Not found","date_of_loss":"Not found","diagnosis":"Not found","dob":"Not found","provider_npi":"Not found","total_billed_amount":"Not found"}

"""'''
    
    # Replace the old prompt
    old_prompt_start = 'prompt = """Extract medical data from document. CRITICAL: Only extract data explicitly visible in the document. Never fabricate or guess values. If information is not found, use "Not found". JSON only:'
    old_prompt_end = '\n"""'
    
    # Find the old prompt section
    start_idx = content.find(old_prompt_start)
    if start_idx != -1:
        end_idx = content.find(old_prompt_end, start_idx) + len(old_prompt_end)
        old_section = content[start_idx:end_idx]
        new_content = content.replace(old_section, new_prompt)
        
        # Write the fixed version
        backup_file = Path("src/vllm_client.py.backup")
        vllm_file.rename(backup_file)
        
        new_vllm_file = Path("src/vllm_client.py")
        new_vllm_file.write_text(new_content)
        
        print(f"✅ Backup created: {backup_file}")
        print(f"✅ Fixed version written: {new_vllm_file}")
        print("\n📝 Changes made:")
        print("   - Removed confidence_scores from prompt template")
        print("   - VLM will focus on data extraction only")
        print("   - Confidence will be calculated in post-processing")
    else:
        print("❌ Could not find prompt section to replace")

def create_option2_fix():
    """Create Option 2 fix with improved confidence instructions."""
    print("\n🔧 Creating Option 2 Fix (Improved Confidence Instructions)")
    print("-" * 50)
    
    improved_prompt = '''"""Extract medical data from document. CRITICAL: Only extract data explicitly visible in the document. Never fabricate or guess values. If information is not found, use "Not found". 

Rate confidence for each field (0.0-1.0):
- 0.9-1.0: Clear, unambiguous data
- 0.7-0.8: Likely correct but some uncertainty  
- 0.5-0.6: Possible but unclear
- 0.0-0.4: Not found or very uncertain

JSON only:
{"claim_id":"Not found","patient_name":"Not found","document_type":"Not found","date_of_loss":"Not found","diagnosis":"Not found","dob":"Not found","provider_npi":"Not found","total_billed_amount":"Not found","confidence_scores":{"claim_id":0.0,"patient_name":0.0,"document_type":0.0,"date_of_loss":0.0,"diagnosis":0.0,"dob":0.0,"provider_npi":0.0,"total_billed_amount":0.0}}

"""'''
    
    print("Improved prompt with confidence instructions:")
    print(improved_prompt)
    print("\n⚠️  This option increases prompt length significantly")
    print("   May cause context window issues with 3B model")

def main():
    """Main function to analyze and provide solutions."""
    analyze_confidence_issue()
    provide_solutions()
    
    print("\n" + "=" * 50)
    print("🚀 Recommended Action Plan:")
    print("=" * 50)
    print("1. Try Option 1 (Remove Confidence Scoring) - Quick fix")
    print("2. Test with a PDF document")
    print("3. If needed, try Option 2 (Improved Instructions)")
    print("4. Consider Option 3 (Post-Processing) for long-term")
    
    choice = input("\nApply Option 1 fix now? (y/n): ").lower().strip()
    if choice == 'y':
        create_option1_fix()
        print("\n✅ Option 1 applied! Test with: python main.py")
    else:
        print("\n📋 Manual fixes available in the options above")
        create_option2_fix()

if __name__ == "__main__":
    main()
