"""Ollama API client for local SLM inference (SOLAR 10.7B)."""

import asyncio
import json
import time
from typing import Optional
import aiohttp
from aiohttp import ClientError, ClientTimeout

from .config import Config
from .models import ExtractionResult


class OllamaClient:
    """Async Ollama API client for local SLM inference."""
    
    def __init__(self):
        self.host = Config.OLLAMA_HOST
        self.port = Config.OLLAMA_PORT
        self.model = Config.OLLAMA_MODEL
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = ClientTimeout(total=Config.API_TIMEOUT)
        self._semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
    
    def _build_extraction_prompt(self, document_text: str) -> str:
        """Build the structured extraction prompt."""
        aliases = Config.FIELD_ALIASES
        
        prompt = """Extract the following fields from the medical document.
Return ONLY valid JSON with the exact field names below. Do not include any explanation or markdown.

Field names and possible aliases in the document:
1. claim_id (aliases: """ + ", ".join(aliases["claim_id"]) + """)
2. patient_name (aliases: """ + ", ".join(aliases["patient_name"]) + """)
3. document_type (aliases: """ + ", ".join(aliases["document_type"]) + """)
4. date_of_loss (aliases: """ + ", ".join(aliases["date_of_loss"]) + """)
5. diagnosis (aliases: """ + ", ".join(aliases["diagnosis"]) + """)
6. dob (aliases: """ + ", ".join(aliases["dob"]) + """)
7. provider_npi (aliases: """ + ", ".join(aliases["provider_npi"]) + """)
8. total_billed_amount (aliases: """ + ", ".join(aliases["total_billed_amount"]) + """)

STRICT RULES:
- Do NOT hallucinate data not in the document.
- For missing fields, return null.
- For dates, use ISO 8601 format (YYYY-MM-DD) or "Unclear" if ambiguous.
- For NPI, extract 10-digit number if present, else null.
- For currency, return as $X.XX if possible, else raw text.
- Include a "confidence_scores" object with 0–1 confidence for each field.

Document text:
""" + document_text + """

Return JSON (and ONLY JSON, no markdown code blocks):
{
  "claim_id": "...",
  "patient_name": "...",
  "document_type": "...",
  "date_of_loss": "...",
  "diagnosis": "...",
  "dob": "...",
  "provider_npi": "...",
  "total_billed_amount": "...",
  "confidence_scores": { "claim_id": 0.99, ... }
}"""
        return prompt
    
    async def extract(self, document_text: str, document_id: str) -> tuple[Optional[ExtractionResult], float]:
        """
        Extract fields from document text using Ollama API.
        
        Returns:
            Tuple of (ExtractionResult or None, latency_ms)
        """
        prompt = self._build_extraction_prompt(document_text)
        start_time = time.time()
        
        async with self._semaphore:
            for attempt in range(4):  # Max 4 retries with exponential backoff
                try:
                    async with aiohttp.ClientSession(timeout=self.timeout) as session:
                        async with session.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "num_predict": 2048,
                                    "top_p": 0.9,
                                }
                            }
                        ) as response:
                            if response.status != 200:
                                raise ClientError(f"Ollama API returned status {response.status}")
                            
                            result = await response.json()
                            latency_ms = (time.time() - start_time) * 1000
                            
                            # Parse response
                            response_text = result.get("response", "")
                            extracted_data = self._parse_json_response(response_text)
                            
                            if extracted_data:
                                try:
                                    result_obj = ExtractionResult(**extracted_data)
                                    return result_obj, latency_ms
                                except Exception as e:
                                    # Validation error
                                    return None, latency_ms
                            
                            return None, latency_ms
                            
                except (ClientError, asyncio.TimeoutError) as e:
                    if attempt == 3:
                        raise
                    wait_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s, 8s
                    await asyncio.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    if attempt == 3:
                        raise
                    wait_time = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait_time)
                    continue
        
        return None, (time.time() - start_time) * 1000
    
    def _parse_json_response(self, response_text: str) -> Optional[dict]:
        """Parse JSON from Ollama response, handling various formats."""
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end > start:
                json_str = response_text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Try extracting from generic code block
        if "```" in response_text:
            start = response_text.find("```") + 3
            # Skip language identifier if present
            newline_pos = response_text.find("\n", start)
            if newline_pos > start and newline_pos - start < 20:
                start = newline_pos + 1
            end = response_text.find("```", start)
            if end > start:
                json_str = response_text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Try to find JSON object in text
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end > start:
            json_str = response_text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def check_health(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status != 200:
                        return False
                    
                    data = await response.json()
                    models = data.get("models", [])
                    
                    # Check if our model is available
                    for model in models:
                        if model.get("name", "").startswith("solar"):
                            return True
                    
                    return False
        except Exception:
            return False
    
    async def get_model_info(self) -> Optional[dict]:
        """Get information about the loaded model."""
        try:
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.post(
                    f"{self.base_url}/api/show",
                    json={"name": self.model}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception:
            return None
    
    async def close(self) -> None:
        """Close any open connections (no persistent connection for aiohttp)."""
        pass
