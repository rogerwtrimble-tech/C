#!/usr/bin/env python3
"""
Performance monitoring utility to detect torch-c-dlpack-ext benefits.

This script analyzes performance metrics and provides specific recommendations
for when to install torch-c-dlpack-ext for performance improvements.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from performance_analyzer import analyze_current_performance, log_performance_recommendations
    from config import Config
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def main():
    """Main performance analysis function."""
    print("🔍 VLM Performance Analysis")
    print("=" * 50)
    print()
    
    # Check if performance metrics are enabled
    if not Config.ENABLE_PERFORMANCE_METRICS:
        print("⚠️  Performance metrics are disabled in config")
        print("Set ENABLE_PERFORMANCE_METRICS=true in .env to enable monitoring")
        return
    
    # Generate performance report
    report = analyze_current_performance()
    print(report)
    
    # Log recommendations to system log
    log_performance_recommendations()
    
    print()
    print("📋 Next Steps:")
    print("1. If bottlenecks are detected, install: pip install torch-c-dlpack-ext")
    print("2. Restart the VLM server to apply optimizations")
    print("3. Run this script again to verify improvements")


if __name__ == "__main__":
    main()
