from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

MODELS = {
    "gpt4o_mini": {
        'api_key': OPENAI_API_KEY,
        'base_url': None, 
        'model': "gpt-4o-mini-2024-07-18",  
    },
    "gpt4o": {
        'api_key': OPENAI_API_KEY,
        'base_url': None, 
        'model': "gpt-4o-2024-08-06",  
    },
    "llama-v3p3-70b-instruct": {
        'api_key': FIREWORKS_API_KEY,
        'base_url': "https://api.fireworks.ai/inference/v1",
        'model': "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "qwen-2.5-72b-instruct": {
        'api_key': FIREWORKS_API_KEY,
        'base_url': "https://api.fireworks.ai/inference/v1",
        'model': "accounts/fireworks/models/qwen2p5-72b-instruct",
    },
    'llama-3.1-8b-instruct-abliterated': {
        'api_key': 'nonce',
        'base_url': "http://0.0.0.0:8001/v1",
        'model': "huihui-ai/Meta-Llama-3.1-8B-Instruct-abliterated",
    },
    'grok2': {
        'api_key': GROK_API_KEY,
        'base_url': "https://api.x.ai/v1",
        'model': "grok-2-latest",
    },
    'llama-3.1-8b-instruct': {
        'api_key': FIREWORKS_API_KEY,
        'base_url': "https://api.fireworks.ai/inference/v1",
        'model': "accounts/fireworks/models/llama-v3p1-8b-instruct",
    },
    'llama-4-scout': {
        'api_key': TOGETHER_API_KEY,
        'base_url': "https://api.together.xyz/v1",
        'model': "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    },
    'llama-4-maverick':{
        'api_key': TOGETHER_API_KEY,
        'base_url': "https://api.together.xyz/v1",
        'model': "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    }
}

class CustomModel:
    def __init__(self, model_name):
        self.limiter = AsyncLimiter(30, time_period=1)
        self.client = AsyncOpenAI(base_url=MODELS[model_name]['base_url'],
                    api_key = MODELS[model_name]['api_key'],)
        self.model_name = model_name
        
    async def generate(self, prompt, temperature, max_tokens):
        async with self.limiter:
            try:
                response = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODELS[self.model_name]['model'],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error: {str(e)}")
                return None

    async def batch_generate(self, list_of_prompts, temperature, max_tokens):
        tasks = [self.generate(prompt, temperature, max_tokens) for prompt in list_of_prompts]
        return await tqdm_asyncio.gather(*tasks)
