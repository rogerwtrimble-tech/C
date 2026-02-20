#!/usr/bin/env python3
"""Test cross-platform path functionality."""

import sys
from pathlib import Path
from src.path_utils import path_manager
from src.config import Config

def test_path_normalization():
    """Test path normalization across platforms."""
    print("🔧 Testing Cross-Platform Path Normalization")
    print("=" * 50)
    
    # Test project root detection
    project_root = path_manager.get_project_root()
    print(f"Project Root: {project_root}")
    print(f"Project Root Type: {type(project_root)}")
    print(f"Exists: {project_root.exists()}")
    print()
    
    # Test PDF input directory
    pdf_dir = Config.PDF_INPUT_DIR
    print(f"PDF Input Dir: {pdf_dir}")
    print(f"PDF Input Dir (absolute): {pdf_dir.absolute()}")
    print(f"PDF Input Dir Exists: {pdf_dir.exists()}")
    print(f"PDF Input Dir is Directory: {pdf_dir.is_dir()}")
    print()
    
    # Test audit log directory
    audit_dir = Config.AUDIT_LOG_PATH
    print(f"Audit Log Dir: {audit_dir}")
    print(f"Audit Log Dir (absolute): {audit_dir.absolute()}")
    print(f"Audit Log Dir Exists: {audit_dir.exists()}")
    print(f"Audit Log Dir is Directory: {audit_dir.is_dir()}")
    print()
    
    # Test PDF file detection
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"PDF Files Found: {len(pdf_files)}")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
    else:
        print("❌ PDF directory not found")
    
    print()
    
    # Test path manager utilities
    print("🛠️  Testing Path Manager Utilities")
    print("-" * 30)
    
    # Test ensure_directory
    test_dir = path_manager.ensure_directory("temp/test_cross_platform")
    print(f"Created Test Dir: {test_dir}")
    print(f"Test Dir Exists: {test_dir.exists()}")
    
    # Test safe_join
    joined_path = path_manager.safe_join(project_root, "src", "config.py")
    print(f"Safe Join Result: {joined_path}")
    print(f"Safe Join Exists: {joined_path.exists()}")
    
    print()
    
    # Test environment detection
    print("🌍 Environment Detection")
    print("-" * 25)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current Working Directory: {Path.cwd()}")
    
    # Test WSL detection
    try:
        import os
        uname_info = os.uname()
        print(f"Uname Release: {uname_info.release}")
        print(f"Is WSL: {'microsoft' in uname_info.release.lower()}")
    except (AttributeError, OSError):
        print("WSL detection not available")
    
    print()
    print("✅ Cross-Platform Path Test Complete!")

if __name__ == "__main__":
    test_path_normalization()
