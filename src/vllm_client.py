"""VLM client for multimodal document extraction using Qwen2.5-VL."""

import asyncio
import aiohttp
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from .config import Config
from .models import ExtractionResult


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
        prompt = """You are a medical document extraction expert. Analyze this document image and extract the following information.

**CRITICAL INSTRUCTIONS:**
1. Return ONLY valid JSON with the exact field names specified below
2. Use visual layout understanding to locate fields (tables, forms, headers)
3. For signatures, provide bounding boxes as [x1, y1, x2, y2] in pixels
4. For handwritten text, transcribe carefully
5. Include confidence scores (0.0-1.0) for each field
6. If a field is not found, set it to null

**Fields to Extract:**
"""
        
        for field_name, aliases in field_aliases.items():
            aliases_str = ", ".join(aliases)
            prompt += f"\n- {field_name} (may appear as: {aliases_str})"
        
        prompt += """

**Additional Visual Elements:**
- signatures: List of signature locations with bounding boxes
- stamps: List of stamp/seal locations with bounding boxes
- handwritten_notes: Transcribe any handwritten annotations
- tables: Extract structured table data if present
- form_fields: Identify form field labels and values

**Output Format:**
Return a JSON object with this structure:
{
  "claim_id": "value or null",
  "patient_name": "value or null",
  "document_type": "value or null",
  "date_of_loss": "YYYY-MM-DD or null",
  "diagnosis": "value or null",
  "dob": "YYYY-MM-DD or null",
  "provider_npi": "value or null",
  "total_billed_amount": "value or null",
  "confidence_scores": {
    "claim_id": 0.0-1.0,
    "patient_name": 0.0-1.0,
    ...
  },
  "visual_elements": {
    "signatures": [
      {"bbox": [x1, y1, x2, y2], "confidence": 0.95, "label": "patient signature"}
    ],
    "stamps": [...],
    "handwritten_notes": "transcribed text",
    "tables": [...],
    "form_fields": {...}
  }
}

Return ONLY the JSON, no explanation or markdown formatting.
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
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_multimodal_prompt(Config.FIELD_ALIASES)
        
        # Convert images to base64
        image_data = []
        for img in images:
            img_b64 = self._numpy_to_base64(img)
            image_data.append(img_b64)
        
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
                            "max_tokens": 4096,
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
                return None, (time.time() - start_time) * 1000, None
        
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
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse JSON from VLM response.
        
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
            return data
            
        except json.JSONDecodeError:
            # Try to find JSON object in text
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    data = json.loads(json_str)
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
