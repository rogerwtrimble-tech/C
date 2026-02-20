#!/usr/bin/env python3
"""Increase VLM context length to handle larger documents."""

import subprocess
import sys
from pathlib import Path

def update_config():
    """Update configuration to increase context length."""
    print("🔧 Updating Configuration for Increased Context Length")
    print("=" * 60)
    
    # Update config.py
    config_file = Path("src/config.py")
    if config_file.exists():
        content = config_file.read_text()
        
        # Update VLM_MAX_MODEL_LEN
        if "VLM_MAX_MODEL_LEN: int = int(os.getenv(\"VLM_MAX_MODEL_LEN\", \"2048\"))" in content:
            content = content.replace(
                'VLM_MAX_MODEL_LEN: int = int(os.getenv("VLM_MAX_MODEL_LEN", "2048"))',
                'VLM_MAX_MODEL_LEN: int = int(os.getenv("VLM_MAX_MODEL_LEN", "4096"))'
            )
            config_file.write_text(content)
            print("✅ Updated config.py: VLM_MAX_MODEL_LEN = 4096")
        
        # Update GPU memory utilization
        if "VLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv(\"VLM_GPU_MEMORY_UTILIZATION\", \"0.75\"))" in content:
            content = content.replace(
                'VLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLM_GPU_MEMORY_UTILIZATION", "0.75"))',
                'VLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLM_GPU_MEMORY_UTILIZATION", "0.85"))'
            )
            config_file.write_text(content)
            print("✅ Updated config.py: VLM_GPU_MEMORY_UTILIZATION = 0.85")
    
    # Update setup script
    setup_file = Path("scripts/setup_vllm.sh")
    if setup_file.exists():
        content = setup_file.read_text()
        
        # Update max-model-len
        if "--max-model-len 2048" in content:
            content = content.replace("--max-model-len 2048", "--max-model-len 4096")
            setup_file.write_text(content)
            print("✅ Updated setup_vllm.sh: --max-model-len 4096")
        
        # Update gpu-memory-utilization
        if "--gpu-memory-utilization 0.75" in content:
            content = content.replace("--gpu-memory-utilization 0.75", "--gpu-memory-utilization 0.85")
            setup_file.write_text(content)
            print("✅ Updated setup_vllm.sh: --gpu-memory-utilization 0.85")

def show_impact():
    """Show the impact of increased context length."""
    print("\n📊 Impact of Context Length Increase")
    print("=" * 60)
    
    scenarios = [
        {"tokens": 2048, "pages": 4, "description": "Current (Conservative)"},
        {"tokens": 4096, "pages": 9, "description": "Proposed (Moderate)"},
        {"tokens": 8192, "pages": 19, "description": "Future (Aggressive)"},
    ]
    
    for scenario in scenarios:
        tokens = scenario["tokens"]
        pages = scenario["pages"]
        desc = scenario["description"]
        
        # Calculate document sizes
        small_doc = 3 * pages  # Small documents
        large_doc = 23 * pages  # Large documents
        
        print(f"{desc:25} | {tokens:4} tokens | {pages:2} pages/chunk")
        print(f"{'':25} | Small docs: {small_doc:3} pages | Large docs: {large_doc:3} pages")
        print()

def check_vllm_server():
    """Check if VLM server is running and needs restart."""
    print("🔍 Checking VLM Server Status")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8000/v1/models"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ VLM server is running")
            print("⚠️  NOTE: Server needs restart to apply new context length")
            print("   Run: bash scripts/setup_vllm.sh")
        else:
            print("❌ VLM server is not running")
            print("   Start with: bash scripts/setup_vllm.sh")
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Cannot connect to VLM server")
        print("   Make sure server is running: bash scripts/setup_vllm.sh")

def provide_restart_instructions():
    """Provide instructions for restarting VLM server."""
    print("\n🚀 VLM Server Restart Instructions")
    print("=" * 60)
    
    print("To apply the new context length:")
    print()
    print("1. Stop current VLM server (if running):")
    print("   pkill -f 'vllm.entrypoints.openai.api_server'")
    print()
    print("2. Start VLM server with new settings:")
    print("   bash scripts/setup_vllm.sh")
    print()
    print("3. Verify new context length:")
    print("   curl http://localhost:8000/v1/models")
    print()
    print("4. Test with large PDF:")
    print("   python main.py")
    print()

def main():
    """Main function to increase context length."""
    print("🔧 Increasing VLM Context Length for Large PDFs")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("This will update configuration files. Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Operation cancelled")
        return
    
    # Update configuration
    update_config()
    
    # Show impact
    show_impact()
    
    # Check server status
    check_vllm_server()
    
    # Provide restart instructions
    provide_restart_instructions()
    
    print("\n✅ Configuration updated successfully!")
    print("📋 Next steps:")
    print("   1. Restart VLM server: bash scripts/setup_vllm.sh")
    print("   2. Test with large PDFs: python main.py")
    print("   3. Monitor memory usage")

if __name__ == "__main__":
    main()
