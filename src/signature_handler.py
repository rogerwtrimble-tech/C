"""Signature detection and validation using YOLOv8."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .config import Config
from .image_preprocessor import ImagePreprocessor


@dataclass
class SignatureDetection:
    """Signature detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    page_number: int
    signature_id: str
    image_path: Optional[Path] = None


class SignatureHandler:
    """Detects and validates signatures using YOLOv8."""
    
    def __init__(self):
        self.enabled = Config.SIGNATURE_DETECTION_ENABLED
        self.model_path = Config.SIGNATURE_MODEL_PATH
        self.confidence_threshold = Config.SIGNATURE_CONFIDENCE_THRESHOLD
        self.min_size = Config.SIGNATURE_MIN_SIZE
        self.max_size = Config.SIGNATURE_MAX_SIZE
        self.output_dir = Config.SIGNATURE_OUTPUT_DIR
        self.preprocessor = ImagePreprocessor()
        
        # Load YOLO model if available
        self.model = None
        if self.enabled and YOLO_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLOv8 signature detection model."""
        if self.model_path.exists():
            try:
                self.model = YOLO(str(self.model_path))
                print(f"Loaded YOLOv8 signature model from {self.model_path}")
            except Exception as e:
                print(f"Failed to load YOLOv8 model: {e}")
                self.enabled = False
        else:
            print(f"YOLOv8 model not found at {self.model_path}")
            print("Signature detection will be disabled")
            self.enabled = False
    
    def detect_signatures(
        self,
        images: List[np.ndarray],
        document_id: str
    ) -> List[SignatureDetection]:
        """
        Detect signatures in a list of page images.
        
        Args:
            images: List of page images (numpy arrays)
            document_id: Document identifier
            
        Returns:
            List of signature detections
        """
        if not self.enabled or self.model is None:
            return []
        
        all_detections = []
        
        for page_num, image in enumerate(images):
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Convert to integers
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Validate size
                    width = x2 - x1
                    height = y2 - y1
                    
                    if self.min_size <= width <= self.max_size and \
                       self.min_size <= height <= self.max_size:
                        
                        # Generate signature ID
                        sig_id = self._generate_signature_id(
                            document_id, page_num, bbox
                        )
                        
                        detection = SignatureDetection(
                            bbox=bbox,
                            confidence=confidence,
                            page_number=page_num + 1,  # 1-indexed
                            signature_id=sig_id
                        )
                        
                        all_detections.append(detection)
        
        return all_detections
    
    def validate_signature_regions(
        self,
        vlm_regions: List[Dict],
        yolo_detections: List[SignatureDetection]
    ) -> Dict[str, bool]:
        """
        Validate VLM-detected signature regions against YOLO detections.
        
        Args:
            vlm_regions: Signature regions from VLM (with bounding boxes)
            yolo_detections: YOLO signature detections
            
        Returns:
            Dictionary mapping region IDs to validation status
        """
        validation_results = {}
        
        for vlm_region in vlm_regions:
            region_id = vlm_region.get("id", "unknown")
            vlm_bbox = vlm_region.get("bbox")
            page_num = vlm_region.get("page_number", 1)
            
            if not vlm_bbox:
                validation_results[region_id] = False
                continue
            
            # Find matching YOLO detections on same page
            page_detections = [
                d for d in yolo_detections
                if d.page_number == page_num
            ]
            
            # Check for overlap with any YOLO detection
            validated = False
            for detection in page_detections:
                iou = self._calculate_iou(vlm_bbox, detection.bbox)
                
                # Consider validated if IoU > 0.5
                if iou > 0.5:
                    validated = True
                    break
            
            validation_results[region_id] = validated
        
        return validation_results
    
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def extract_signature_images(
        self,
        images: List[np.ndarray],
        detections: List[SignatureDetection],
        document_id: str
    ) -> List[SignatureDetection]:
        """
        Extract and save signature image crops.
        
        Args:
            images: List of page images
            detections: List of signature detections
            document_id: Document identifier
            
        Returns:
            Updated detections with image_path set
        """
        updated_detections = []
        
        for detection in detections:
            page_idx = detection.page_number - 1  # Convert to 0-indexed
            
            if page_idx >= len(images):
                updated_detections.append(detection)
                continue
            
            # Get page image
            page_image = images[page_idx]
            
            # Crop signature region
            sig_crop = self.preprocessor.crop_region(
                page_image,
                detection.bbox
            )
            
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{document_id}_sig_{detection.signature_id}_{timestamp}.png"
            output_path = self.output_dir / filename
            
            # Save cropped signature
            self.preprocessor.save_image(sig_crop, output_path)
            
            # Update detection with image path
            detection.image_path = output_path
            updated_detections.append(detection)
        
        return updated_detections
    
    def _generate_signature_id(
        self,
        document_id: str,
        page_num: int,
        bbox: Tuple[int, int, int, int]
    ) -> str:
        """
        Generate unique signature ID.
        
        Args:
            document_id: Document identifier
            page_num: Page number (0-indexed)
            bbox: Bounding box coordinates
            
        Returns:
            Unique signature ID (hash)
        """
        # Create unique string from document, page, and bbox
        unique_str = f"{document_id}_p{page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        
        # Generate hash
        sig_hash = hashlib.sha256(unique_str.encode()).hexdigest()[:12]
        
        return sig_hash
    
    def get_signature_summary(
        self,
        detections: List[SignatureDetection]
    ) -> Dict[str, any]:
        """
        Generate summary statistics for signature detections.
        
        Args:
            detections: List of signature detections
            
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                "total_signatures": 0,
                "pages_with_signatures": [],
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }
        
        pages_with_sigs = list(set(d.page_number for d in detections))
        confidences = [d.confidence for d in detections]
        
        return {
            "total_signatures": len(detections),
            "pages_with_signatures": sorted(pages_with_sigs),
            "average_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "signature_ids": [d.signature_id for d in detections]
        }
