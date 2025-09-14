import logging
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import after_log, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

load_dotenv()
logger = logging.getLogger()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    after=after_log(logger, logging.WARNING),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def generate_function(model: str, prompt: str, temperature: float = 1.0) -> str:
    start_time = time.time()
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    elapsed_time = time.time() - start_time
    print(f"Function generation took {elapsed_time:.2f} seconds")
    print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
    return response.choices[0].message.content
