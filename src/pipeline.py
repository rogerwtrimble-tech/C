"""Main processing pipeline for medical document extraction."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import hashlib

from .config import Config
from .models import ExtractionResult
from .ollama_client import OllamaClient
from .pdf_extractor import PDFExtractor
from .audit_logger import AuditLogger
from .secure_handler import SecureFileHandler


class ExtractionPipeline:
    """Main pipeline for processing PDF documents."""
    
    def __init__(self):
        Config.ensure_directories()
        Config.validate()
        
        self.ollama_client = OllamaClient()
        self.pdf_extractor = PDFExtractor()
        self.audit_logger = AuditLogger()
        self.secure_handler = SecureFileHandler()
    
    async def process_document(self, pdf_path: Path) -> Optional[ExtractionResult]:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractionResult or None if failed
        """
        document_id = self.pdf_extractor.get_document_id(pdf_path)
        
        # Log document received
        filename_hash = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:12]
        await self.audit_logger.log_document_received(document_id, filename_hash)
        
        try:
            # Check Ollama health on first document
            if not hasattr(self, '_health_checked'):
                self._health_checked = True
                if not await self.ollama_client.check_health():
                    raise RuntimeError(
                        f"Ollama server not running or model not available. "
                        f"Ensure Ollama is running on {Config.get_ollama_url()} "
                        f"with model '{Config.OLLAMA_MODEL}'"
                    )
            
            
            # Extract text from PDF
            text, has_images = self.pdf_extractor.extract_text(pdf_path)
            
            # SOLAR 10.7B is text-only; use OCR fallback for scanned docs
            if self.pdf_extractor.should_use_vision(pdf_path, text):
                # Try to extract more text with pdfplumber page rendering
                text = self._enhance_text_extraction(pdf_path, text)
            
            # Process with local SLM
            result, latency = await self._process_with_text(text, document_id)
            
            if result:
                # Save result
                await self._save_result(result, document_id)
                
                # Log success
                await self.audit_logger.log_extraction(
                    document_id=document_id,
                    document_type=result.document_type,
                    fields_extracted=8,
                    fields_successful=self._count_successful_fields(result),
                    latency_ms=latency,
                    confidence_avg=self._calculate_avg_confidence(result),
                    status="success",
                    model_version=Config.OLLAMA_MODEL
                )
            
            return result
            
        except Exception as e:
            # Log error
            await self.audit_logger.log_error(
                document_id=document_id,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return None
    
    async def _process_with_text(
        self, 
        text: str, 
        document_id: str
    ) -> tuple[Optional[ExtractionResult], float]:
        """Process document using local SLM."""
        return await self.ollama_client.extract(text, document_id)
    
    def _enhance_text_extraction(self, pdf_path: Path, original_text: str) -> str:
        """Enhance text extraction for scanned documents."""
        # Try extracting text from page images with higher resolution
        text_parts = [original_text] if original_text.strip() else []
        
        # Add page-by-page extraction
        page_texts = self.pdf_extractor.extract_page_range_text(pdf_path, 0, 100)
        if page_texts:
            text_parts.append(page_texts)
        
        return "\n\n".join(text_parts) if text_parts else original_text
    
    async def _save_result(self, result: ExtractionResult, document_id: str) -> Path:
        """Save extraction result to JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_filename = f"{document_id}_{timestamp}.json"
        output_path = Config.RESULTS_OUTPUT_DIR / output_filename
        
        result_json = result.model_dump_json(indent=2)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        
        return output_path
    
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
    
    def _calculate_avg_confidence(self, result: ExtractionResult) -> float:
        """Calculate average confidence score."""
        if not result.confidence_scores:
            return 0.0
        
        scores = list(result.confidence_scores.values())
        return sum(scores) / len(scores) if scores else 0.0
    
    async def process_directory(
        self, 
        input_dir: Optional[Path] = None,
        max_concurrent: int = 10
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
            print(f"Date of Loss: {result.date_of_loss}")
            print(f"Avg Confidence: {pipeline._calculate_avg_confidence(result):.2%}")
            
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
