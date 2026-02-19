"""Download YOLOv8 signature detection model."""

import os
from pathlib import Path
from ultralytics import YOLO

def download_signature_model():
    """Download pre-trained YOLOv8 signature detection model."""
    
    print("=" * 60)
    print("YOLOv8 Signature Detection Model Setup")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "yolov8_signature.pt"
    
    if model_path.exists():
        print(f"Model already exists at: {model_path}")
        print("Skipping download.")
        return
    
    print("\nDownloading YOLOv8 signature detection model...")
    print("Note: Using YOLOv8n as base model for signature detection")
    print("For production, train on signature dataset or use pre-trained signature model")
    
    try:
        # Download YOLOv8n model (lightweight, ~6MB)
        # In production, replace with actual signature-trained model
        model = YOLO('yolov8n.pt')
        
        # Save to models directory
        model.save(str(model_path))
        
        print(f"\n✓ Model downloaded successfully to: {model_path}")
        print("\nIMPORTANT: This is a base YOLOv8 model.")
        print("For production use, train on signature dataset or download")
        print("a pre-trained signature detection model from:")
        print("  - https://github.com/ultralytics/ultralytics")
        print("  - https://universe.roboflow.com/ (signature datasets)")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nManual download instructions:")
        print("1. Download a signature detection model")
        print("2. Save as: models/yolov8_signature.pt")
        return 1
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(download_signature_model())
