import asyncio
from src.ollama_client import OllamaClient

async def test():
    client = OllamaClient()
    
    # Test direct API call
    async with client._semaphore:
        import aiohttp
        async with aiohttp.ClientSession(timeout=client.timeout) as session:
            async with session.post(
                f'{client.base_url}/api/generate',
                json={
                    'model': client.model,
                    'prompt': 'Return JSON only: {"test": "value"}',
                    'stream': False,
                    'options': {'temperature': 0.1}
                }
            ) as response:
                print('Status:', response.status)
                result = await response.json()
                print('Response:', result.get('response', ''))
                
                # Try to parse
                parsed = client._parse_json_response(result.get('response', ''))
                print('Parsed:', parsed)
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(test())
