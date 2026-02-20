#!/usr/bin/env python3
"""Test performance monitoring in WSL environment."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from performance_monitor import performance_monitor
from multimodal_pipeline import MultimodalPipeline

async def test_performance_monitoring():
    """Test performance monitoring without full pipeline."""
    print("🔧 Testing Performance Monitoring in WSL")
    print("=" * 50)
    
    # Test basic performance monitoring
    print("Testing basic performance monitoring...")
    start_time = performance_monitor.start_operation("test_operation")
    
    # Simulate some work
    await asyncio.sleep(0.1)
    
    metrics = performance_monitor.end_operation("test_operation", start_time)
    
    print(f"Operation: {metrics.operation_name}")
    print(f"Duration: {metrics.duration_ms:.2f}ms")
    print(f"Memory Before: {metrics.memory_before_mb:.2f}MB")
    print(f"Memory After: {metrics.memory_after_mb:.2f}MB")
    print(f"Memory Peak: {metrics.memory_peak_mb:.2f}MB")
    print(f"GPU Memory Before: {metrics.gpu_memory_before_mb}")
    print(f"GPU Memory After: {metrics.gpu_memory_after_mb}")
    print()
    
    print("✅ Performance monitoring test completed successfully!")
    
    # Test with a simple PDF processing if available
    pdf_dir = Path("pdfs")
    if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
        print("\n🔧 Testing VLM with performance monitoring...")
        try:
            pipeline = MultimodalPipeline()
            
            # Get first PDF
            pdf_file = list(pdf_dir.glob("*.pdf"))[0]
            print(f"Testing with: {pdf_file.name}")
            
            # Process just the VLM part
            result = await pipeline.process_document(pdf_file)
            
            if result:
                print(f"✅ VLM processing successful!")
                print(f"   Patient: {result.patient_name}")
                print(f"   Document Type: {result.document_type}")
            else:
                print("❌ VLM processing failed")
            
            await pipeline.cleanup()
            
        except Exception as e:
            print(f"❌ VLM test failed: {e}")
    else:
        print("ℹ️  No PDF files found for VLM testing")
    
    print("\n🌍 Environment Info:")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

if __name__ == "__main__":
    asyncio.run(test_performance_monitoring())
