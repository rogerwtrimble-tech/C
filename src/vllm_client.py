"""VLM client for multimodal document extraction using Qwen2.5-VL."""

import asyncio
import aiohttp
import base64
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from datetime import datetime
from PIL import Image
from io import BytesIO

from .config import Config
from .models import ExtractionResult, VisualElements
from .performance_monitor import performance_monitor

class VLLMClient:
    """Client for Vision-Language Model inference via vLLM."""
    
    def __init__(self):
        self.host = Config.VLM_HOST
        self.port = Config.VLM_PORT
        self.model = Config.VLM_MODEL
        self.base_url = Config.get_vlm_url()
        self.timeout = aiohttp.ClientTimeout(total=Config.API_TIMEOUT)
        self._semaphore = asyncio.Semaphore(Config.VLM_MAX_CONCURRENT_PAGES)
        self.max_retries = Config.RETRY_ATTEMPTS
        self.retry_backoff = Config.RETRY_BACKOFF
    
    def _build_multimodal_prompt(self, field_aliases: Dict[str, List[str]]) -> str:
        """
        Build extraction prompt for multimodal VLM.
        
        Args:
            field_aliases: Dictionary of field names to aliases
            
        Returns:
            Formatted prompt string
        """
        prompt = """Extract medical data from document. CRITICAL: Only extract data explicitly visible in the document. Never fabricate or guess values. If information is not found, use "Not found". JSON only:

{"claim_id":"Not found","patient_name":"Not found","document_type":"Not found","date_of_loss":"Not found","diagnosis":"Not found","dob":"Not found","provider_npi":"Not found","total_billed_amount":"Not found","confidence_scores":{"claim_id":0,"patient_name":0,"document_type":0,"date_of_loss":0,"diagnosis":0,"dob":0,"provider_npi":0,"total_billed_amount":0}}

"""
        return prompt
    
    async def extract_multimodal(
        self,
        images: List[np.ndarray],
        document_id: str
    ) -> Tuple[Optional[ExtractionResult], float, Optional[Dict]]:
        """
        Extract data from document images using VLM.
        
        Args:
            images: List of page images (numpy arrays)
            document_id: Document identifier
            
        Returns:
            Tuple of (ExtractionResult, latency_ms, visual_elements)
        """
        # Start performance monitoring
        perf_start = performance_monitor.start_operation("vlm_multimodal_extraction")
        
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_multimodal_prompt(Config.FIELD_ALIASES)
        
        # Convert images to base64 (monitor tensor operations)
        convert_start = performance_monitor.start_operation("numpy_to_base64_conversion")
        image_data = []
        for img in images:
            img_b64 = self._numpy_to_base64(img)
            image_data.append(img_b64)
        performance_monitor.end_operation("numpy_to_base64_conversion", convert_start)
        
        # Retry logic
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._semaphore:
                    async with aiohttp.ClientSession(timeout=self.timeout) as session:
                        # Build request payload
                        payload = {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt}
                                    ] + [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{img_b64}"
                                            }
                                        }
                                        for img_b64 in image_data
                                    ]
                                }
                            ],
                            "temperature": 0.1,
                            "max_tokens": 512,
                            "top_p": 0.9
                        }
                        
                        # Make API call
                        async with session.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"VLM API error {response.status}: {error_text}")
                            
                            result = await response.json()
                            latency_ms = (time.time() - start_time) * 1000
                            
                            # Parse response
                            response_text = result["choices"][0]["message"]["content"]
                            extracted_data = self._parse_json_response(response_text)
                            
                            if extracted_data:
                                # Separate visual elements from extraction data
                                visual_elements = extracted_data.pop("visual_elements", None)
                                
                                # Add required fields
                                extracted_data.update({
                                    'processing_path': 'multimodal_vlm',
                                    'pdf_metadata': None,
                                    'extraction_latency_ms': None,
                                    'model_version': self.model
                                })
                                
                                try:
                                    result_obj = ExtractionResult(**extracted_data)
                                    return result_obj, latency_ms, visual_elements
                                except Exception as e:
                                    print(f"Validation error: {e}")
                                    return None, latency_ms, visual_elements
                            
                            return None, latency_ms, None
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries:
                    print(f"VLM extraction failed after {self.max_retries} attempts: {e}")
                    return None, (time.time() - start_time) * 1000, None
                
                wait_time = (self.retry_backoff ** attempt)
                await asyncio.sleep(wait_time)
                continue
            
            except Exception as e:
                print(f"Unexpected error in VLM extraction: {e}")
                performance_monitor.end_operation("vlm_multimodal_extraction", perf_start)
                return None, (time.time() - start_time) * 1000, None
        
        # End performance monitoring
        performance_monitor.end_operation("vlm_multimodal_extraction", perf_start)
        return None, (time.time() - start_time) * 1000, None
    
    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded string.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Save to bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode to base64
        img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_b64
    
    def _validate_extracted_data(self, data: Dict) -> Dict:
        """
        Validate extracted data to prevent fabricated values.
        
        Args:
            data: Raw extracted data from VLM
            
        Returns:
            Validated data with "Not found" for missing/fabricated values
        """
        # Common fabricated patterns to detect
        fabricated_patterns = [
            'john doe', 'jane doe', 'john smith', 'jane smith',
            'patient', 'unknown', 'n/a', 'na', 'none', 'null',
            'example', 'test', 'sample', 'placeholder'
        ]
        
        # Fields that should never contain fabricated data
        text_fields = [
            'claim_id', 'patient_name', 'document_type', 
            'diagnosis', 'provider_npi', 'total_billed_amount'
        ]
        
        # Validate each field
        for field in text_fields:
            if field in data:
                value = str(data[field]).strip()
                
                # Check for null/empty values
                if not value or value.lower() in ['null', 'none', '']:
                    data[field] = "Not found"
                # Check for fabricated patterns
                elif any(pattern in value.lower() for pattern in fabricated_patterns):
                    data[field] = "Not found"
                # Check for placeholder-like values (all same character repeated)
                elif len(set(value.lower())) <= 2 and len(value) > 3:
                    data[field] = "Not found"
        
        # Special validation for dates
        date_fields = ['date_of_loss', 'dob']
        for field in date_fields:
            if field in data:
                value = str(data[field]).strip()
                if not value or value.lower() in ['null', 'none', '', 'not found']:
                    data[field] = "Not found"
                # Validate date format
                elif not self._is_valid_date_format(value):
                    data[field] = "Not found"
        
        return data
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if date string is in a valid format."""
        if not date_str or date_str == "Not found":
            return False
        
        # Check for common date formats
        import re
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
        ]
        
        return any(re.match(pattern, date_str) for pattern in date_patterns)
    
    def _convert_date_format(self, date_str: str) -> str:
        """
        Convert various date formats to YYYY-MM-DD.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Date in YYYY-MM-DD format or original string if conversion fails
        """
        if not date_str or date_str == "null":
            return date_str
            
        # Common date formats to try
        formats = [
            "%m/%d/%Y", "%m-%d-%Y",  # MM/DD/YYYY, MM-DD-YYYY
            "%Y/%m/%d", "%Y-%m-%d",  # YYYY/MM/DD, YYYY-MM-DD (already correct)
            "%d/%m/%Y", "%d-%m-%Y",  # DD/MM/YYYY, DD-MM-YYYY
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        return date_str  # Return original if no format matches
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse JSON from VLM response and convert date formats.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed JSON dictionary or None
        """
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Validate data to prevent fabricated values
            if data:
                data = self._validate_extracted_data(data)
                
                # Convert date formats
                date_fields = ["date_of_loss", "dob"]
                for field in date_fields:
                    if field in data and data[field] and data[field] != "Not found":
                        data[field] = self._convert_date_format(data[field])
            
            return data
            
        except json.JSONDecodeError:
            # Try to find JSON object in text
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    data = json.loads(json_str)
                    
                    # Validate data to prevent fabricated values
                    if data:
                        data = self._validate_extracted_data(data)
                        
                        # Convert date formats
                        date_fields = ["date_of_loss", "dob"]
                        for field in date_fields:
                            if field in data and data[field] and data[field] != "Not found":
                                data[field] = self._convert_date_format(data[field])
                    
                    return data
            except:
                pass
            
            print(f"Failed to parse JSON from VLM response: {response_text[:200]}")
            return None
    
    async def check_health(self) -> bool:
        """
        Check if vLLM server is healthy and model is loaded.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        models = await response.json()
                        # Check if our model is in the list
                        model_ids = [m.get("id", "") for m in models.get("data", [])]
                        return any(self.model in mid for mid in model_ids)
                    return False
        except Exception as e:
            print(f"VLM health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Cleanup resources."""
        # No persistent connections to close
        pass
