import asyncio
import logging
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from typing import Any
from google import genai
import re
import ast

logger = logging.getLogger()


class GeminiKeyRotator:
    def __init__(self):
        self.keys = self._load_keys()
        self.current_index = 0
        self._lock = asyncio.Lock()

    def _load_keys(self):
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        if not keys_str:
            single_key = os.getenv("GEMINI_API_KEY")
            return [single_key] if single_key else []

        keys = [key.strip() for key in keys_str.split(",")]
        return [key for key in keys if key]

    async def get_client(self):
        async with self._lock:
            if not self.keys:
                raise ValueError("No Gemini API keys available")

            current_key = self.keys[self.current_index]
            return genai.Client(api_key=current_key)

    async def rotate_key(self):
        async with self._lock:
            if not self.keys:
                raise ValueError("No Gemini API keys available")

            self.current_index = (self.current_index + 1) % len(self.keys)
            return self.keys[self.current_index]


gemini_key_rotator = GeminiKeyRotator()


async def generate_with_gemini(model: str, prompt: str) -> str:
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            client = await gemini_key_rotator.get_client()
            response = await client.aio.models.generate_content(
                model=model, contents=prompt
            )
            # Rotate the key after generating the response to avoid rate limiting
            gemini_key_rotator.rotate_key()
            return response.text
        except Exception as e:
            error_message = str(e)
            attempt += 1
            if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
                try:
                    await gemini_key_rotator.rotate_key()
                    logger.info("Rotated to a new Gemini API key")
                    if attempt == max_attempts:
                        max_attempts += 1

                except ValueError as key_error:
                    logger.error(f"Key rotation failed: {key_error}")
                    break
            if attempt >= max_attempts:
                raise
    raise RuntimeError("All attempts to generate with Gemini failed")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def generate_function(
    prompt: str,
    model: str = "gemini-2.5-pro-exp-03-25",
) -> dict[str, Any]:
    return await generate_with_gemini(model, prompt)
