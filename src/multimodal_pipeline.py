"""Multimodal pipeline for Grade A PDF extraction using VLM + YOLOv8."""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

from .config import Config
from .models import ExtractionResult
from .visual_models import (
    VisualElements, MultimodalExtractionMetadata,
    SignatureElement, BoundingBox
)
from .vllm_client import VLLMClient
from .image_preprocessor import ImagePreprocessor
from .signature_handler import SignatureHandler, SignatureDetection
from .audit_logger import AuditLogger
from .secure_handler import SecureFileHandler
from .performance_monitor import performance_monitor
from .performance_analyzer import log_performance_recommendations
from .path_utils import path_manager


class MultimodalPipeline:
    """
    Grade A multimodal extraction pipeline.
    
    Workflow:
    1. PDF → Images (300 DPI, preprocessed)
    2. VLM Vision Sweep (extract data + identify signature zones)
    3. YOLOv8 Signature Validation (verify VLM detections)
    4. Auto-Clipping (save signature images)
    5. Confidence Check (flag discrepancies for review)
    """
    
    def __init__(self):
        Config.ensure_directories()
        Config.validate()
        
        self.vlm_client = VLLMClient()
        self.image_preprocessor = ImagePreprocessor()
        self.signature_handler = SignatureHandler()
        self.audit_logger = AuditLogger()
        self.secure_handler = SecureFileHandler()
    
    async def process_document(self, pdf_path: Path) -> Optional[ExtractionResult]:
        """
        Process a single PDF document with multimodal VLM workflow.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractionResult with visual elements or None if failed
        """
        document_id = self._get_document_id(pdf_path)
        start_time = time.time()
        
        # Log document received
        filename_hash = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:12]
        await self.audit_logger.log_document_received(document_id, filename_hash)
        
        try:
            # Stage 1: Preprocessing - PDF to Images
            print(f"Stage 1: Converting PDF to images...")
            preprocess_start = time.time()
            images = self.image_preprocessor.pdf_to_images(pdf_path)
            preprocess_time = (time.time() - preprocess_start) * 1000
            print(f"  Converted {len(images)} pages in {preprocess_time:.0f}ms")
            
            # Stage 2: Vision Sweep - VLM Extraction
            print(f"Stage 2: VLM vision sweep...")
            vlm_start = time.time()
            result, vlm_latency, visual_elements_dict = await self.vlm_client.extract_multimodal(
                images, document_id
            )
            vlm_time = (time.time() - vlm_start) * 1000
            print(f"  VLM extraction completed in {vlm_time:.0f}ms")
            
            if not result:
                print("  VLM extraction failed")
                return None
            
            # Parse visual elements from VLM response
            visual_elements = self._parse_visual_elements(visual_elements_dict) if visual_elements_dict else VisualElements()
            
            # Stage 3: Signature Validation - YOLOv8
            print(f"Stage 3: YOLOv8 signature validation...")
            yolo_start = time.time()
            yolo_detections = self.signature_handler.detect_signatures(images, document_id)
            yolo_time = (time.time() - yolo_start) * 1000
            print(f"  Detected {len(yolo_detections)} signatures in {yolo_time:.0f}ms")
            
            # Stage 4: Auto-Clipping - Extract signature images
            print(f"Stage 4: Auto-clipping signatures...")
            if yolo_detections:
                yolo_detections = self.signature_handler.extract_signature_images(
                    images, yolo_detections, document_id
                )
            
            # Merge VLM and YOLO signature detections
            visual_elements = self._merge_signature_detections(
                visual_elements, yolo_detections
            )
            
            # Stage 5: Confidence Check - Validate and flag discrepancies
            print(f"Stage 5: Confidence validation...")
            multimodal_metadata = self._create_metadata(
                len(images),
                preprocess_time,
                vlm_time,
                yolo_time,
                visual_elements
            )
            
            # Check for discrepancies
            if visual_elements.has_discrepancies() and Config.AUTO_FLAG_DISCREPANCIES:
                multimodal_metadata.add_review_reason("VLM/YOLO signature discrepancy detected")
                print("  ⚠️  Signature discrepancy detected - flagged for review")
            
            # Check confidence thresholds
            low_confidence_fields = result.get_low_confidence_fields(Config.MIN_CONFIDENCE_THRESHOLD)
            if low_confidence_fields:
                multimodal_metadata.add_review_reason(f"Low confidence fields: {', '.join(low_confidence_fields)}")
                print(f"  ⚠️  Low confidence fields: {', '.join(low_confidence_fields)}")
            
            # Update result with visual elements and metadata
            result.visual_elements = visual_elements
            result.multimodal_metadata = multimodal_metadata
            result.processing_path = "multimodal_vlm"
            result.extraction_latency_ms = (time.time() - start_time) * 1000
            
            # Save result
            await self._save_result(result, document_id)
            
            # Log success
            await self._log_extraction(result, document_id)
            
            # Cleanup temp images if configured
            self.image_preprocessor.cleanup_temp_images()
            
            print(f"✅ Multimodal extraction completed in {result.extraction_latency_ms:.0f}ms")
            
            # Log performance recommendations
            log_performance_recommendations()
            
            return result
            
        except Exception as e:
            print(f"❌ Error in multimodal processing: {e}")
            await self.audit_logger.log_error(
                document_id=document_id,
                error_type=type(e).__name__,
                error_message=str(e),
                processing_path="multimodal_vlm"
            )
            return None
    
    def _parse_visual_elements(self, visual_dict: Dict) -> VisualElements:
        """
        Parse visual elements dictionary from VLM response.
        
        Args:
            visual_dict: Dictionary of visual elements from VLM
            
        Returns:
            VisualElements object
        """
        visual_elements = VisualElements()
        
        # Parse signatures
        if "signatures" in visual_dict:
            for sig_data in visual_dict["signatures"]:
                bbox_data = sig_data.get("bbox", [0, 0, 0, 0])
                signature = SignatureElement(
                    signature_id=f"vlm_{len(visual_elements.signatures)}",
                    bbox=BoundingBox(
                        x1=int(bbox_data[0]),
                        y1=int(bbox_data[1]),
                        x2=int(bbox_data[2]),
                        y2=int(bbox_data[3])
                    ),
                    confidence=sig_data.get("confidence", 0.0),
                    page_number=sig_data.get("page_number", 1),
                    label=sig_data.get("label"),
                    vlm_detected=True,
                    yolo_detected=False
                )
                visual_elements.signatures.append(signature)
        
        # Parse other visual elements (stamps, handwritten notes, tables, form fields)
        # Implementation can be extended as needed
        
        return visual_elements
    
    def _merge_signature_detections(
        self,
        visual_elements: VisualElements,
        yolo_detections: List[SignatureDetection]
    ) -> VisualElements:
        """
        Merge VLM and YOLO signature detections.
        
        Args:
            visual_elements: Visual elements from VLM
            yolo_detections: YOLO signature detections
            
        Returns:
            Updated VisualElements with merged signatures
        """
        # Mark VLM signatures that match YOLO detections
        for vlm_sig in visual_elements.signatures:
            for yolo_det in yolo_detections:
                if vlm_sig.page_number == yolo_det.page_number:
                    # Calculate IoU
                    iou = self._calculate_iou(
                        vlm_sig.bbox.to_tuple(),
                        yolo_det.bbox
                    )
                    if iou > 0.5:
                        vlm_sig.yolo_detected = True
                        vlm_sig.validated = True
                        vlm_sig.image_path = str(yolo_det.image_path) if yolo_det.image_path else None
                        break
        
        # Add YOLO-only detections (not detected by VLM)
        for yolo_det in yolo_detections:
            # Check if already matched
            matched = False
            for vlm_sig in visual_elements.signatures:
                if vlm_sig.page_number == yolo_det.page_number:
                    iou = self._calculate_iou(
                        vlm_sig.bbox.to_tuple(),
                        yolo_det.bbox
                    )
                    if iou > 0.5:
                        matched = True
                        break
            
            if not matched:
                # Add as YOLO-only detection
                signature = SignatureElement(
                    signature_id=yolo_det.signature_id,
                    bbox=BoundingBox(
                        x1=yolo_det.bbox[0],
                        y1=yolo_det.bbox[1],
                        x2=yolo_det.bbox[2],
                        y2=yolo_det.bbox[3]
                    ),
                    confidence=yolo_det.confidence,
                    page_number=yolo_det.page_number,
                    image_path=str(yolo_det.image_path) if yolo_det.image_path else None,
                    vlm_detected=False,
                    yolo_detected=True,
                    validated=True
                )
                visual_elements.signatures.append(signature)
        
        return visual_elements
    
    def _calculate_iou(
        self,
        bbox1: tuple,
        bbox2: tuple
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_metadata(
        self,
        total_pages: int,
        preprocess_time: float,
        vlm_time: float,
        yolo_time: float,
        visual_elements: VisualElements
    ) -> MultimodalExtractionMetadata:
        """Create multimodal extraction metadata."""
        return MultimodalExtractionMetadata(
            total_pages=total_pages,
            pages_with_signatures=visual_elements.get_signature_pages(),
            pages_with_handwriting=[],  # Can be populated from visual_elements
            pages_with_tables=[],       # Can be populated from visual_elements
            image_preprocessing_time_ms=preprocess_time,
            vlm_inference_time_ms=vlm_time,
            signature_detection_time_ms=yolo_time,
            total_processing_time_ms=preprocess_time + vlm_time + yolo_time,
            vlm_model_used=Config.VLM_MODEL,
            yolo_model_used=str(Config.SIGNATURE_MODEL_PATH) if Config.SIGNATURE_DETECTION_ENABLED else None
        )
    
    async def _save_result(self, result: ExtractionResult, document_id: str) -> Path:
        """Save extraction result to JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_filename = f"{document_id}_{timestamp}.json"
        output_path = Config.RESULTS_OUTPUT_DIR / output_filename
        
        result_json = result.model_dump_json(indent=2)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        
        return output_path
    
    async def _log_extraction(self, result: ExtractionResult, document_id: str) -> None:
        """Log extraction to audit log."""
        fields_extracted = sum(1 for field in [
            result.claim_id, result.patient_name, result.document_type,
            result.date_of_loss, result.diagnosis, result.dob,
            result.provider_npi, result.total_billed_amount
        ] if field and field != "Unknown")
        
        await self.audit_logger.log_extraction(
            document_id=document_id,
            document_type=result.document_type,
            fields_attempted=8,
            fields_extracted=fields_extracted,
            fields_skipped=8 - fields_extracted,
            latency_ms=result.extraction_latency_ms or 0.0,
            confidence_avg=result.get_average_confidence(),
            confidence_scores=result.confidence_scores,
            status="success" if fields_extracted > 0 else "failed",
            model_version=result.model_version,
            processing_path=result.processing_path,
            pdf_type="multimodal",
            ocr_engine="VLM",
            notes=f"Signatures detected: {result.visual_elements.get_signature_count() if result.visual_elements else 0}"
        )
    
    def _get_document_id(self, pdf_path: Path) -> str:
        """Generate document ID from PDF filename."""
        return hashlib.sha256(pdf_path.name.encode()).hexdigest()[:12]
    
    async def process_directory(
        self,
        input_dir: Optional[Path] = None,
        max_concurrent: int = 2  # Lower for VLM due to memory
    ) -> List[ExtractionResult]:
        """
        Process all PDFs in a directory with multimodal pipeline.
        
        Args:
            input_dir: Directory containing PDFs
            max_concurrent: Maximum concurrent processes
            
        Returns:
            List of extraction results
        """
        if input_dir is None:
            input_dir = Config.PDF_INPUT_DIR
        
        # Ensure input directory is properly normalized
        input_dir = path_manager.normalize_path(input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            return []
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(pdf_path: Path) -> Optional[ExtractionResult]:
            async with semaphore:
                return await self.process_document(pdf_path)
        
        tasks = [process_with_limit(pdf) for pdf in pdf_files]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        return [r for r in results if r is not None]
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.vlm_client.close()
        self.image_preprocessor.cleanup_temp_images()
