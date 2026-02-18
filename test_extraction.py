import asyncio
from src.ollama_client import OllamaClient

async def test():
    client = OllamaClient()
    
    # Test with sample medical text
    prompt = client._build_extraction_prompt("""
    PATIENT INFORMATION
    Name: John Smith
    DOB: 01/15/1985
    Claim ID: WC-2024-12345
    
    DISCHARGE SUMMARY
    Date of Loss: 02/10/2024
    Diagnosis: Fractured right tibia
    Provider NPI: 1234567890
    Total Billed: $15,750.00
    """)
    
    print('--- PROMPT ---')
    print(prompt[:500])
    print('...')
    
    async with client._semaphore:
        import aiohttp
        async with aiohttp.ClientSession(timeout=client.timeout) as session:
            async with session.post(
                f'{client.base_url}/api/generate',
                json={
                    'model': client.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1}
                }
            ) as response:
                print('Status:', response.status)
                result = await response.json()
                print('--- RESPONSE ---')
                print(result.get('response', ''))
                
                # Try to parse
                parsed = client._parse_json_response(result.get('response', ''))
                print('--- PARSED ---')
                print(json.dumps(parsed, indent=2))
    
    await client.close()

if __name__ == "__main__":
    import json
    asyncio.run(test())
