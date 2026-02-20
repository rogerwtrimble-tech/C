#!/usr/bin/env python3
"""Test and demonstrate the chunking solution for large PDFs."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.chunking_processor import ChunkingProcessor
from src.config import Config

def analyze_token_impact():
    """Analyze the impact of different token limits."""
    print("🔍 Token Size Impact Analysis")
    print("=" * 50)
    
    processor = ChunkingProcessor()
    
    print(f"Current Configuration:")
    print(f"  Model: {Config.VLM_MODEL}")
    print(f"  Max Model Length: {Config.VLM_MAX_MODEL_LEN} tokens")
    print(f"  GPU Memory Utilization: {Config.VLM_GPU_MEMORY_UTILIZATION}")
    print(f"  Max Pages Per Chunk: {processor.max_pages_per_chunk}")
    print()
    
    # Simulate different scenarios
    scenarios = [
        {"pages": 1, "description": "Single page document"},
        {"pages": 3, "description": "Small document (3 pages)"},
        {"pages": 8, "description": "Medium document (8 pages)"},
        {"pages": 23, "description": "Large document (23 pages)"},
        {"pages": 200, "description": "Very large document (200 pages)"},
    ]
    
    # Estimate tokens per page (conservative)
    tokens_per_page = 600
    prompt_tokens = 100
    response_tokens = 200
    
    print("Document Size Analysis:")
    print("-" * 40)
    for scenario in scenarios:
        pages = scenario["pages"]
        total_tokens = (pages * tokens_per_page) + prompt_tokens + response_tokens
        chunks_needed = (pages + processor.max_pages_per_chunk - 1) // processor.max_pages_per_chunk
        
        status = "✅" if total_tokens <= Config.VLM_MAX_MODEL_LEN else "❌"
        if chunks_needed > 1:
            status = f"🔄 {chunks_needed} chunks"
        
        print(f"  {scenario['description']:25} | {pages:3} pages | {total_tokens:5} tokens | {status}")
    
    print()

def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("🛠️  Chunking Strategies")
    print("=" * 50)
    
    strategies = [
        {
            "name": "Current (Conservative)",
            "max_model_len": 2048,
            "gpu_util": 0.75,
            "description": "Safe for 12GB VRAM, 2-3 pages per chunk"
        },
        {
            "name": "Moderate",
            "max_model_len": 4096,
            "gpu_util": 0.80,
            "description": "4-5 pages per chunk, moderate VRAM usage"
        },
        {
            "name": "Aggressive",
            "max_model_len": 8192,
            "gpu_util": 0.90,
            "description": "8-10 pages per chunk, high VRAM usage"
        }
    ]
    
    for strategy in strategies:
        # Calculate max pages for this strategy
        available_tokens = strategy["max_model_len"] - 300  # Reserve for prompt/response
        max_pages = max(1, available_tokens // 600)  # 600 tokens per page estimate
        
        print(f"  {strategy['name']:20} | {strategy['max_model_len']:4} tokens | {max_pages:2} pages/chunk | {strategy['description']}")
    
    print()

async def test_chunking_processor():
    """Test the chunking processor with simulated data."""
    print("🧪 Testing Chunking Processor")
    print("=" * 50)
    
    processor = ChunkingProcessor()
    
    # Test with different document sizes
    test_cases = [
        {"pages": 2, "expected_chunks": 1},
        {"pages": 5, "expected_chunks": 2},
        {"pages": 10, "expected_chunks": 4},
        {"pages": 23, "expected_chunks": 8},
    ]
    
    for test_case in test_cases:
        pages = test_case["pages"]
        expected_chunks = test_case["expected_chunks"]
        
        # Simulate images (dummy data)
        images = [f"image_{i}" for i in range(pages)]
        
        # Calculate chunks
        chunks = processor._create_chunks(images)
        
        status = "✅" if len(chunks) == expected_chunks else "❌"
        print(f"  {pages:2} pages → {len(chunks)} chunks {status} (expected {expected_chunks})")
        
        # Show chunk details for first test case
        if pages == 5:
            print(f"    Chunk 1: {len(chunks[0])} pages")
            print(f"    Chunk 2: {len(chunks[1])} pages")
    
    print()

def show_merging_strategy():
    """Show the result merging strategy."""
    print("🔄 Result Merging Strategy")
    print("=" * 50)
    
    print("When processing multiple chunks, we use confidence-weighted merging:")
    print()
    print("Example - Patient Name from 3 chunks:")
    print("  Chunk 1 (pages 1-2): 'John Smith'    confidence: 0.95")
    print("  Chunk 2 (pages 3-4): 'J. Smith'      confidence: 0.70")
    print("  Chunk 3 (pages 5-6): 'Not found'     confidence: 0.00")
    print("  → Selected: 'John Smith' (highest confidence)")
    print()
    print("Benefits:")
    print("  ✅ Prioritizes high-confidence extractions")
    print("  ✅ Handles conflicting data intelligently")
    print("  ✅ Maintains traceability of source chunks")
    print("  ✅ Graceful fallback when chunks fail")
    print()

def provide_implementation_options():
    """Provide implementation options for the user."""
    print("🚀 Implementation Options")
    print("=" * 50)
    
    print("OPTION 1: Increase Context Length (Quick)")
    print("  - Update VLM_MAX_MODEL_LEN to 4096")
    print("  - Restart vLLM server with --max-model-len 4096")
    print("  - Handles 4-5 pages per chunk")
    print("  - Risk: Higher memory usage")
    print()
    
    print("OPTION 2: Implement Chunking (Recommended)")
    print("  - Already implemented in chunking_processor.py")
    print("  - Automatic detection of large documents")
    print("  - Intelligent merging of chunk results")
    print("  - Works with current 2048 token limit")
    print()
    
    print("OPTION 3: Hybrid Approach (Best)")
    print("  - Increase to 4096 tokens + implement chunking")
    print("  - Handles both medium and very large documents")
    print("  - Maximum flexibility and performance")
    print()
    
    print("OPTION 4: Page Limit (Simplest)")
    print("  - Limit processing to first N pages (e.g., 5 pages)")
    print("  - Fastest but incomplete extraction")
    print("  - Good for summaries only")

def main():
    """Main function to run all analyses."""
    analyze_token_impact()
    demonstrate_chunking_strategies()
    asyncio.run(test_chunking_processor())
    show_merging_strategy()
    provide_implementation_options()
    
    print("=" * 50)
    print("🎯 RECOMMENDATION")
    print("=" * 50)
    print("1. Implement chunking (already done)")
    print("2. Test with current large PDFs")
    print("3. If needed, increase context length to 4096")
    print("4. Monitor memory usage and performance")
    print()
    print("Current chunking processor handles:")
    print(f"  - Max pages per chunk: {ChunkingProcessor().max_pages_per_chunk}")
    print(f"  - Automatic large document detection")
    print(f"  - Confidence-weighted result merging")

if __name__ == "__main__":
    main()
