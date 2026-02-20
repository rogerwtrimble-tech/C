#!/bin/bash
# Quick fix for missing Poppler utilities

echo "🔧 Installing Poppler utilities for PDF processing..."

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Poppler utilities
echo "Installing poppler-utils..."
sudo apt-get install -y poppler-utils

# Verify installation
echo "Verifying Poppler installation..."
which pdftoppm
which pdfinfo

# Test Poppler
echo "Testing Poppler functionality..."
pdftoppm --version
pdfinfo --version

echo "✅ Poppler installation complete!"
echo ""
echo "Please restart your terminal and try the multimodal pipeline again."
