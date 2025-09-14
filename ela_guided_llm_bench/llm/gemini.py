import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import after_log, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

load_dotenv()
logger = logging.getLogger()


@dataclass
class KeyUsageInfo:
    last_used: float
    requests_this_minute: int
    total_requests: int
    last_rate_limit: float | None = None
    is_rate_limited: bool = False


class GeminiKeyRotator:
    def __init__(self, requests_per_minute_per_key: int = 10):
        self.keys = self._load_keys()
        self.current_index = 0
        self._lock = asyncio.Lock()
        self.requests_per_minute_per_key = requests_per_minute_per_key

        self.key_usage: dict[str, KeyUsageInfo] = {
            key: KeyUsageInfo(last_used=0.0, requests_this_minute=0, total_requests=0) for key in self.keys
        }

        self.last_cleanup = time.time()

    def _load_keys(self):
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        if not keys_str:
            single_key = os.getenv("GEMINI_API_KEY")
            return [single_key] if single_key else []

        keys = [key.strip() for key in keys_str.split(",")]
        filtered_keys = [key for key in keys if key]
        random.shuffle(filtered_keys)
        return filtered_keys

    def _cleanup_minute_counters(self):
        current_time = time.time()
        if current_time - self.last_cleanup >= 60:
            for usage_info in self.key_usage.values():
                usage_info.requests_this_minute = 0
                if usage_info.last_rate_limit and current_time - usage_info.last_rate_limit > 300:
                    usage_info.is_rate_limited = False
            self.last_cleanup = current_time

    def _get_best_available_key(self) -> str | None:
        self._cleanup_minute_counters()
        available_keys = []
        for key, usage_info in self.key_usage.items():
            if not usage_info.is_rate_limited and usage_info.requests_this_minute < self.requests_per_minute_per_key:
                available_keys.append((key, usage_info))
        if not available_keys:
            best_key = min(
                self.key_usage.items(),
                key=lambda x: (x[1].last_used, x[1].total_requests),
            )[0]
            return best_key

        best_key = min(available_keys, key=lambda x: (x[1].last_used, x[1].total_requests))[0]
        return best_key

    async def get_client(self) -> tuple[str, genai.Client]:
        async with self._lock:
            if not self.keys:
                raise ValueError("No Gemini API keys available")

            key = self._get_best_available_key()
            if not key:
                raise ValueError("No available API keys")

            current_time = time.time()
            usage_info = self.key_usage[key]
            usage_info.last_used = current_time
            usage_info.requests_this_minute += 1
            usage_info.total_requests += 1

            print(f"Using key: {key[:8]}... (requests this minute: {usage_info.requests_this_minute})")
            return key, genai.Client(api_key=key)

    async def mark_key_rate_limited(self, key: str):
        async with self._lock:
            if key in self.key_usage:
                usage_info = self.key_usage[key]
                usage_info.is_rate_limited = True
                usage_info.last_rate_limit = time.time()
                print(f"Marked key {key[:8]}... as rate limited")

    def get_available_key_count(self) -> int:
        self._cleanup_minute_counters()
        return sum(1 for usage_info in self.key_usage.values() if not usage_info.is_rate_limited)

    def get_usage_stats(self) -> dict:
        self._cleanup_minute_counters()
        return {
            key[:8]
            + "...": {
                "requests_this_minute": usage_info.requests_this_minute,
                "total_requests": usage_info.total_requests,
                "is_rate_limited": usage_info.is_rate_limited,
                "last_used": usage_info.last_used,
            }
            for key, usage_info in self.key_usage.items()
        }


gemini_key_rotator = GeminiKeyRotator()


async def generate_with_gemini(
    model: str,
    prompt: str,
    temperature: float = 0.1,
) -> str:
    max_attempts = 3
    attempt = 0
    used_client = None

    while attempt < max_attempts:
        try:
            if used_client is None:
                key, used_client = await gemini_key_rotator.get_client()

            response = await used_client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            logger.warning("Prompt tokens: %d", response.usage_metadata.prompt_token_count)
            logger.warning("Output tokens: %d", response.usage_metadata.candidates_token_count)
            logger.warning("Total tokens: %d", response.usage_metadata.total_token_count)
            return response.text
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            attempt += 1

            logger.error(f"Exception on attempt {attempt}: {error_type}: {error_message}")
            if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
                await gemini_key_rotator.mark_key_rate_limited(key)
                logger.warning(f"Rate limit hit on attempt {attempt} {error_message} for key {key}")
                model_to_base_wait_time = {
                    "gemini-2.0-flash": 0,
                    "gemini-2.5-flash": 60,
                }
                wait_time = min(model_to_base_wait_time[model] + (attempt * 30), 180)
                await asyncio.sleep(wait_time)

                used_client = None
                if attempt == max_attempts:
                    max_attempts += 1
            else:
                logger.error(f"Non-rate-limit error: {error_type}: {error_message}")
                await asyncio.sleep(10)

            if attempt >= max_attempts:
                logger.error(f"Max attempts ({max_attempts}) reached, raising exception")
                raise

    raise RuntimeError("All attempts to generate with Gemini failed")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    after=after_log(logger, logging.WARNING),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def generate_function(
    prompt: str,
    model: str = "gemini-2.5-pro-exp-03-25",
    temperature: float = 1.0,
) -> str:
    start_time = time.time()
    result = await generate_with_gemini(model=model, prompt=prompt, temperature=temperature)
    elapsed_time = time.time() - start_time
    print(f"Function generation took {elapsed_time:.2f} seconds")
    return result
