"""Main processing pipeline for medical document extraction with PDF type detection."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import hashlib
import pdfplumber

from .config import Config
from .models import ExtractionResult
from .ollama_client import OllamaClient
from .pdf_extractor import PDFExtractor
from .ocr_extractor import OCRExtractor
from .audit_logger import AuditLogger
from .secure_handler import SecureFileHandler


class ExtractionPipeline:
    """Main pipeline for processing PDF documents."""
    
    def __init__(self):
        Config.ensure_directories()
        Config.validate()
        
        self.ollama_client = OllamaClient()
        self.pdf_extractor = PDFExtractor()
        self.ocr_extractor = OCRExtractor()
        self.audit_logger = AuditLogger()
        self.secure_handler = SecureFileHandler()
    
    async def process_document(self, pdf_path: Path) -> Optional[ExtractionResult]:
        """
        Process a single PDF document with intelligent type detection.
        
        Returns:
            ExtractionResult or None if failed
        """
        document_id = self.pdf_extractor.get_document_id(pdf_path)
        
        # Log document received
        filename_hash = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:12]
        await self.audit_logger.log_document_received(document_id, filename_hash)
        
        try:
            # Interrogate PDF to determine type
            pdf_metadata = self.pdf_extractor.interrogate_pdf(pdf_path)
            processing_path = pdf_metadata["processing_path"]
            
            # Log PDF type detection
            await self.audit_logger.log_pdf_type_detected(
                document_id=document_id,
                pdf_type=pdf_metadata["pdf_type"],
                processing_path=processing_path,
                alphanumeric_ratio=pdf_metadata["alphanumeric_ratio"]
            )
            
            # Process based on PDF type
            if processing_path == "native_direct":
                result, latency = await self._process_native_pdf(pdf_path, document_id)
                ocr_engine = None
            elif processing_path == "scanned_full_ocr":
                result, latency, ocr_engine = await self._process_scanned_pdf(pdf_path, document_id)
            elif processing_path == "hybrid_selective":
                result, latency, ocr_engine = await self._process_hybrid_pdf(pdf_path, document_id)
            else:
                # Fallback to OCR for unknown
                result, latency, ocr_engine = await self._process_scanned_pdf(pdf_path, document_id)
            
            if result:
                # Add metadata to result
                result.processing_path = processing_path
                result.pdf_metadata = pdf_metadata
                result.extraction_latency_ms = latency
                result.model_version = Config.OLLAMA_MODEL
                
                # Check confidence thresholds
                low_confidence_fields = result.get_low_confidence_fields(0.75)
                status = "success" if not low_confidence_fields else "partial"
                
                # Save result
                await self._save_result(result, document_id)
                
                # Log success
                await self.audit_logger.log_extraction(
                    document_id=document_id,
                    document_type=result.document_type,
                    fields_attempted=8,
                    fields_extracted=self._count_successful_fields(result),
                    fields_skipped=len(low_confidence_fields),
                    latency_ms=latency,
                    confidence_avg=result.get_average_confidence(),
                    confidence_scores=result.confidence_scores,
                    status=status,
                    model_version=Config.OLLAMA_MODEL,
                    processing_path=processing_path,
                    pdf_type=pdf_metadata["pdf_type"],
                    ocr_engine=ocr_engine,
                    notes=f"Low confidence fields: {', '.join(low_confidence_fields)}" if low_confidence_fields else None
                )
            
            return result
            
        except Exception as e:
            # Log error
            await self.audit_logger.log_error(
                document_id=document_id,
                error_type=type(e).__name__,
                error_message=str(e),
                processing_path=processing_path if 'processing_path' in locals() else "unknown"
            )
            return None
    
    async def _process_native_pdf(self, pdf_path: Path, document_id: str) -> tuple[Optional[ExtractionResult], float]:
        """Process native PDF (text-based) directly."""
        # Extract clean text
        text = self.pdf_extractor.extract_text_clean(pdf_path)
        
        # Send to SOLAR
        return await self.ollama_client.extract(text, document_id)
    
    async def _process_scanned_pdf(self, pdf_path: Path, document_id: str) -> tuple[Optional[ExtractionResult], float, Optional[str]]:
        """Process scanned PDF with OCR."""
        # Check Ollama health on first document
        if not hasattr(self, '_health_checked'):
            self._health_checked = True
            if not await self.ollama_client.check_health():
                raise RuntimeError(
                    f"Ollama server not running or model not available. "
                    f"Ensure Ollama is running on {Config.get_ollama_url()} "
                    f"with model '{Config.OLLAMA_MODEL}'"
                )
        
        # Determine OCR engine
        ocr_engine = self._select_ocr_engine(pdf_path)
        
        # Extract text with OCR
        ocr_text = self.ocr_extractor.extract_from_pdf(pdf_path, engine=ocr_engine)
        
        if not ocr_text.strip():
            return None, 0.0, ocr_engine
        
        # Send to SOLAR
        result, latency = await self.ollama_client.extract(ocr_text, document_id)
        return result, latency, ocr_engine
    
    async def _process_hybrid_pdf(self, pdf_path: Path, document_id: str) -> tuple[Optional[ExtractionResult], float, Optional[str]]:
        """Process hybrid PDF with per-page selective OCR."""
        # Get page types
        page_types = self.pdf_extractor.get_page_types(pdf_path)
        
        text_parts = []
        ocr_engine = None
        
        for page_info in page_types:
            page_num = page_info["page_number"] - 1  # Convert to 0-indexed
            
            if page_info["type"] == "native":
                # Extract text directly
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_num]
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            else:
                # Use OCR for this page
                if not ocr_engine:
                    ocr_engine = self._select_ocr_engine(pdf_path)
                
                # Extract single page with OCR
                page_text = self._extract_page_ocr(pdf_path, page_num, ocr_engine)
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        combined_text = "\n\n".join(text_parts)
        
        if not combined_text.strip():
            return None, 0.0, ocr_engine
        
        # Send to SOLAR
        result, latency = await self.ollama_client.extract(combined_text, document_id)
        return result, latency, ocr_engine
    
    async def _save_result(self, result: ExtractionResult, document_id: str) -> Path:
        """Save extraction result to JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_filename = f"{document_id}_{timestamp}.json"
        output_path = Config.RESULTS_OUTPUT_DIR / output_filename
        
        result_json = result.model_dump_json(indent=2)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        
        return output_path
    
    def _select_ocr_engine(self, pdf_path: Path) -> str:
        """Select appropriate OCR engine based on document characteristics."""
        available = self.ocr_extractor.get_available_engines()
        
        # Prefer GPU-accelerated engines for larger documents
        if self.pdf_extractor.get_page_count(pdf_path) > 5:
            if "paddleocr" in available:
                return "paddleocr"
            elif "easyocr" in available:
                return "easyocr"
        
        # Default to Tesseract for small documents or if others unavailable
        return "tesseract" if "tesseract" in available else available[0] if available else "tesseract"
    
    def _extract_page_ocr(self, pdf_path: Path, page_num: int, engine: str) -> str:
        """Extract text from a single page using OCR."""
        # Create temporary single-page PDF (simplified)
        # For now, extract from full PDF and filter by page
        full_text = self.ocr_extractor.extract_from_pdf(pdf_path, engine=engine)
        
        # Split by page markers and return the requested page
        pages = full_text.split("--- Page")
        for page in pages:
            if f"Page {page_num + 1}" in page:
                return page.replace(f" {page_num + 1} ---", "").strip()
        
        return ""
    
    def _count_successful_fields(self, result: ExtractionResult) -> int:
        """Count number of successfully extracted fields."""
        count = 0
        for field in [
            result.claim_id,
            result.patient_name,
            result.document_type,
            result.date_of_loss,
            result.diagnosis,
            result.dob,
            result.provider_npi,
            result.total_billed_amount,
        ]:
            if field is not None and field != "Unknown" and field != "":
                count += 1
        return count
    
    async def process_directory(
        self, 
        input_dir: Optional[Path] = None,
        max_concurrent: int = 4  # Lower for local GPU inference
    ) -> list[ExtractionResult]:
        """
        Process all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDFs (defaults to config)
            max_concurrent: Maximum concurrent processes
            
        Returns:
            List of extraction results
        """
        if input_dir is None:
            input_dir = Config.PDF_INPUT_DIR
        
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
        await self.ollama_client.close()


async def main():
    """Main entry point for processing."""
    pipeline = ExtractionPipeline()
    
    try:
        results = await pipeline.process_directory()
        
        print(f"Processed {len(results)} documents successfully")
        
        for result in results:
            print(f"\n--- {result.claim_id} ---")
            print(f"Patient: {result.patient_name}")
            print(f"Document Type: {result.document_type}")
            print(f"Processing Path: {result.processing_path}")
            print(f"Date of Loss: {result.date_of_loss}")
            print(f"Avg Confidence: {result.get_average_confidence():.2%}")
            
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
