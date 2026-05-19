import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from ela_guided_llm_bench.llm.executor import BaseExecutor
from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import after_log, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger()

REASONING_MODELS = ["z-ai/glm-4.6:exacto", "minimax/minimax-m2"]
REASONING_EFFORT_MODELS = {
    "openai/gpt-5-mini": "low",
    "openai/gpt-5-nano": "low",
    "openai/gpt-5.4-nano": "low",
    "google/gemma-4-31b-it": "low",
    "openai/gpt-oss-120b": "low",
}


def _build_extra_body(model: str) -> dict:
    extra: dict = {}
    if model in REASONING_EFFORT_MODELS:
        extra["reasoning"] = {"effort": REASONING_EFFORT_MODELS[model]}
    elif model in REASONING_MODELS:
        extra["reasoning"] = {"enabled": True}
    return extra


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=5, min=10, max=120),
    retry=retry_if_exception_type((RateLimitError, APIError, asyncio.TimeoutError)),
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
        extra_body=_build_extra_body(model),
    )
    elapsed_time = time.time() - start_time
    print(f"Function generation took {elapsed_time:.2f} seconds")
    return response.choices[0].message.content


class OpenRouterExecutor(BaseExecutor):
    def __init__(self):
        self._delay_in_seconds = 0.0

    @property
    def delay_in_seconds(self) -> float:
        return self._delay_in_seconds

    async def process_tasks_in_batches(self, tasks: list) -> list:
        return await asyncio.gather(*tasks)

    async def log_key_usage_stats(self):
        pass
