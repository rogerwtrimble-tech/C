"""Image preprocessing for multimodal VLM processing."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import pdf2image

from .config import Config


class ImagePreprocessor:
    """Converts PDFs to high-quality images for VLM processing."""
    
    def __init__(self):
        self.dpi = Config.PDF_DPI
        self.enable_preprocessing = Config.IMAGE_PREPROCESSING
        self.enable_deskew = Config.DESKEW_ENABLED
        self.image_format = Config.IMAGE_FORMAT
        self.image_quality = Config.IMAGE_QUALITY
        self.temp_dir = Config.TEMP_IMAGE_DIR
        
    def pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """
        Convert PDF to list of images (one per page).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of numpy arrays (images) in RGB format
        """
        # Convert PDF to PIL images
        pil_images = pdf2image.convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            fmt=self.image_format.lower()
        )
        
        # Convert to numpy arrays
        images = []
        for pil_img in pil_images:
            img_array = np.array(pil_img)
            
            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                img_array = self._preprocess_image(img_array)
            
            images.append(img_array)
        
        return images
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to improve image quality.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Deskew if enabled
        if self.enable_deskew:
            image = self._deskew_image(image)
        
        # Enhance contrast
        image = self._enhance_contrast(image)
        
        # Denoise
        image = self._denoise_image(image)
        
        return image
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in scanned documents.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all non-zero points
        coords = np.column_stack(np.where(binary > 0))
        
        # Calculate minimum area rectangle
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Normalize angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only deskew if angle is significant (> 0.5 degrees)
            if abs(angle) > 0.5:
                # Get image center
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                
                # Rotate image
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to reduce scan artifacts.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Use fastNlMeansDenoisingColored for RGB images
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        """
        Save preprocessed image to disk.
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save with specified quality
        pil_image.save(
            output_path,
            format=self.image_format,
            quality=self.image_quality,
            optimize=True
        )
    
    def get_image_dimensions(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (width, height)
        """
        h, w = image.shape[:2]
        return (w, h)
    
    def crop_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop a region from the image using bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped image region
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def cleanup_temp_images(self) -> None:
        """Remove temporary image files if cleanup is enabled."""
        if Config.CLEANUP_TEMP_FILES and self.temp_dir.exists():
            for img_file in self.temp_dir.glob("*"):
                if img_file.is_file():
                    img_file.unlink()
