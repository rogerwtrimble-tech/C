#!/bin/bash
# Quick fix for Triton C compiler issue

echo "🔧 Fixing Triton C compiler issue..."

# Install build tools
echo "Installing build-essential..."
sudo apt-get update
sudo apt-get install -y build-essential gcc g++

# Set environment variables
echo "Setting CC and CXX environment variables..."
export CC=gcc
export CXX=g++

# Add to .bashrc for persistence
echo "Adding CC/CXX to .bashrc..."
echo 'export CC=gcc' >> ~/.bashrc
echo 'export CXX=g++' >> ~/.bashrc

# Verify installation
echo "Verifying GCC installation..."
gcc --version
g++ --version

echo "✅ Fix complete! Please restart your terminal or run:"
echo "source ~/.bashrc"
echo ""
echo "Then try starting the vLLM server again."
