"""Intelligent chunking processor for large PDF documents."""

import asyncio
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import logging

from .config import Config
from .vllm_client import VLLMClient
from .models import ExtractionResult


@dataclass
class ChunkResult:
    """Result from processing a chunk of pages."""
    chunk_id: int
    page_range: Tuple[int, int]  # (start_page, end_page)
    extraction_result: Optional[ExtractionResult]
    confidence_scores: Dict[str, float]
    processing_time_ms: float


class ChunkingProcessor:
    """
    Intelligent chunking processor for handling large PDF documents.
    
    Strategies:
    1. Increase context length if possible (up to model limits)
    2. Chunk documents by page count when context is exceeded
    3. Merge results from multiple chunks with confidence weighting
    """
    
    def __init__(self):
        self.vlm_client = VLLMClient()
        self.logger = logging.getLogger(__name__)
        self.max_pages_per_chunk = self._calculate_max_pages_per_chunk()
    
    def _calculate_max_pages_per_chunk(self) -> int:
        """
        Calculate maximum pages per chunk based on model constraints.
        
        Returns:
            Maximum number of pages that can fit in context window
        """
        # Base calculation: 2048 tokens total
        # Reserve tokens for prompt (~100) and response (~200)
        available_tokens = Config.VLM_MAX_MODEL_LEN - 300
        
        # Estimate image tokens per page (realistic estimate at 100 DPI)
        # Vision models typically use 300-500 tokens per image at 100 DPI
        tokens_per_page = 400
        
        max_pages = max(1, available_tokens // tokens_per_page)
        
        # Less aggressive safety margin
        max_pages = max(1, max_pages)
        
        self.logger.info(f"Calculated max pages per chunk: {max_pages} (available tokens: {available_tokens})")
        return max_pages
    
    async def process_large_document(
        self, 
        images: List[np.ndarray], 
        document_id: str,
        max_pages_per_chunk: Optional[int] = None
    ) -> Tuple[Optional[ExtractionResult], List[ChunkResult]]:
        """
        Process a large document by chunking it into manageable pieces.
        
        Args:
            images: List of page images
            document_id: Document identifier
            max_pages_per_chunk: Override for max pages per chunk
            
        Returns:
            Merged extraction result and list of chunk results
        """
        if max_pages_per_chunk:
            self.max_pages_per_chunk = max_pages_per_chunk
        
        total_pages = len(images)
        
        # If document fits in one chunk, process normally
        if total_pages <= self.max_pages_per_chunk:
            self.logger.info(f"Document fits in single chunk: {total_pages} pages")
            result, latency, visual_elements = await self.vlm_client.extract_multimodal(
                images, document_id
            )
            return result, [ChunkResult(
                chunk_id=0,
                page_range=(0, total_pages - 1),
                extraction_result=result,
                confidence_scores=result.confidence_scores if result else {},
                processing_time_ms=latency or 0
            )]
        
        # Chunk the document
        self.logger.info(f"Chunking document: {total_pages} pages into chunks of {self.max_pages_per_chunk}")
        chunks = self._create_chunks(images)
        
        # Process each chunk
        chunk_results = []
        for i, chunk_images in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            start_page = i * self.max_pages_per_chunk
            end_page = min(start_page + len(chunk_images) - 1, total_pages - 1)
            
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}: pages {start_page}-{end_page}")
            
            try:
                result, latency, visual_elements = await self.vlm_client.extract_multimodal(
                    chunk_images, chunk_id
                )
                
                chunk_result = ChunkResult(
                    chunk_id=i,
                    page_range=(start_page, end_page),
                    extraction_result=result,
                    confidence_scores=result.confidence_scores if result else {},
                    processing_time_ms=latency or 0
                )
                
                chunk_results.append(chunk_result)
                
            except Exception as e:
                self.logger.error(f"Failed to process chunk {i}: {e}")
                # Add empty result for failed chunk
                chunk_results.append(ChunkResult(
                    chunk_id=i,
                    page_range=(start_page, end_page),
                    extraction_result=None,
                    confidence_scores={},
                    processing_time_ms=0
                ))
        
        # Merge results from all chunks
        merged_result = self._merge_chunk_results(chunk_results, document_id)
        
        return merged_result, chunk_results
    
    def _create_chunks(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Create chunks of images based on max pages per chunk.
        
        Args:
            images: List of all page images
            
        Returns:
            List of image chunks
        """
        chunks = []
        for i in range(0, len(images), self.max_pages_per_chunk):
            chunk = images[i:i + self.max_pages_per_chunk]
            chunks.append(chunk)
        
        return chunks
    
    def _merge_chunk_results(
        self, 
        chunk_results: List[ChunkResult], 
        document_id: str
    ) -> Optional[ExtractionResult]:
        """
        Merge results from multiple chunks using confidence weighting.
        
        Args:
            chunk_results: Results from each chunk
            document_id: Original document ID
            
        Returns:
            Merged extraction result
        """
        if not chunk_results:
            return None
        
        # Collect all non-null results
        valid_results = [cr for cr in chunk_results if cr.extraction_result is not None]
        
        if not valid_results:
            return None
        
        # Initialize merged data
        merged_data = {
            'processing_path': 'multimodal_vlm_chunked',
            'pdf_metadata': None,
            'extraction_latency_ms': sum(cr.processing_time_ms for cr in chunk_results),
            'model_version': self.vlm_client.model,
            'confidence_scores': {}
        }
        
        # Field merging strategy
        fields = ['claim_id', 'patient_name', 'document_type', 'date_of_loss', 'diagnosis', 'dob', 'provider_npi', 'total_billed_amount']
        
        for field in fields:
            field_values = []
            field_confidences = []
            
            # Collect values and confidences from all chunks
            for cr in valid_results:
                value = getattr(cr.extraction_result, field, None)
                confidence = cr.confidence_scores.get(field, 0.0)
                
                if value and value != "Not found" and confidence > 0:
                    field_values.append(value)
                    field_confidences.append(confidence)
            
            # Select best value based on confidence
            if field_values:
                # Use value with highest confidence
                best_idx = max(range(len(field_confidences)), key=lambda i: field_confidences[i])
                merged_data[field] = field_values[best_idx]
                merged_data['confidence_scores'][field] = field_confidences[best_idx]
            else:
                merged_data[field] = "Not found"
                merged_data['confidence_scores'][field] = 0.0
        
        # Calculate average confidence
        if merged_data['confidence_scores']:
            avg_confidence = sum(merged_data['confidence_scores'].values()) / len(merged_data['confidence_scores'])
        else:
            avg_confidence = 0.0
        
        # Create merged result
        try:
            merged_result = ExtractionResult(**merged_data)
            self.logger.info(f"Successfully merged {len(valid_results)} chunks with avg confidence: {avg_confidence:.2%}")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"Failed to create merged result: {e}")
            return None
    
    async def try_increase_context_length(self) -> bool:
        """
        Attempt to increase the model context length.
        
        Returns:
            True if context length was successfully increased
        """
        try:
            # Check current model limits
            health = await self.vlm_client.check_health()
            if not health:
                return False
            
            # For Qwen2.5-VL-3B, the theoretical max is higher than 2048
            # Try increasing to 4096 if we have GPU memory
            if Config.VLM_GPU_MEMORY_UTILIZATION < 0.9:
                new_max_len = min(4096, Config.VLM_MAX_MODEL_LEN * 2)
                
                # Update config (this would require server restart in production)
                old_max_len = Config.VLM_MAX_MODEL_LEN
                Config.VLM_MAX_MODEL_LEN = new_max_len
                
                self.logger.info(f"Attempting to increase context length from {old_max_len} to {new_max_len}")
                
                # Recalculate max pages per chunk
                self.max_pages_per_chunk = self._calculate_max_pages_per_chunk()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to increase context length: {e}")
        
        return False
