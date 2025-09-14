import asyncio
from typing import Any

from ela_guided_llm_bench.llm.gemini import GeminiKeyRotator


class Executor:
    def __init__(
        self,
        gemini_key_rotator: GeminiKeyRotator,
        batch_size: int = 5,
        delay_in_seconds: float = 15.0,
        adaptive_batching: bool = True,
    ) -> None:
        self.gemini_key_rotator = gemini_key_rotator
        self.batch_size = batch_size
        self.delay_in_seconds = delay_in_seconds
        self.adaptive_batching = adaptive_batching

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available API keys."""
        if not self.adaptive_batching:
            return self.batch_size

        available_keys = self.gemini_key_rotator.get_available_key_count()
        if available_keys == 0:
            return 1

        optimal_size = min(self.batch_size, available_keys)
        print(f"Using batch size: {optimal_size} (available keys: {available_keys})")
        return optimal_size

    def _get_adaptive_delay(self) -> float:
        if not self.adaptive_batching:
            return self.delay_in_seconds

        available_keys = self.gemini_key_rotator.get_available_key_count()
        total_keys = len(self.gemini_key_rotator.keys)

        if available_keys < total_keys * 0.5:
            delay = self.delay_in_seconds * 2
            print(f"Many keys rate limited, using extended delay: {delay}s")
            return delay

        delay = max(self.delay_in_seconds, 30.0)
        return delay

    async def process_tasks_in_batches(self, tasks: list) -> list:
        results = []
        i = 0
        batch_number = 0
        while i < len(tasks):
            batch_number += 1

            optimal_batch_size = self._get_optimal_batch_size()
            batch = tasks[i : i + optimal_batch_size]
            actual_batch_size = len(batch)

            total_batches = (len(tasks) + optimal_batch_size - 1) // optimal_batch_size
            print(
                f"Processing batch {batch_number}/{total_batches} "
                f"(tasks {i + 1}-{i + actual_batch_size}, batch size: {actual_batch_size})"
            )

            try:
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                processed_results: list[Any] = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"Task failed with exception: {result}")
                        processed_results.append(None)
                    else:
                        processed_results.append(result)

                results.extend(processed_results)

            except Exception as e:
                print(f"Batch {batch_number} failed completely: {e}")
                results.extend([None] * actual_batch_size)

            i += actual_batch_size

            if i < len(tasks):
                delay = self._get_adaptive_delay()
                print(f"Waiting {delay} seconds before next batch...")
                await asyncio.sleep(delay)

        return results

    async def log_key_usage_stats(self):
        try:
            stats = self.gemini_key_rotator.get_usage_stats()
            print("API Key Usage Stats:")
            for key_id, usage in stats.items():
                status = "RATE_LIMITED" if usage["is_rate_limited"] else "AVAILABLE"
                print(
                    f"  {key_id}: {usage['requests_this_minute']}/15 requests this minute, "
                    f"total: {usage['total_requests']}, status: {status}"
                )
        except Exception as e:
            print(f"Could not get key usage stats: {e}")


class NaiveExecutor(Executor):
    def __init__(
        self,
        gemini_key_rotator: GeminiKeyRotator,
        delay_in_seconds: float = 15.0,
    ) -> None:
        super().__init__(
            gemini_key_rotator=gemini_key_rotator,
            batch_size=1,
            delay_in_seconds=delay_in_seconds,
        )

    async def process_tasks_in_batches(self, tasks: list) -> list:
        results = []
        i = 0
        while i < len(tasks):
            try:
                result = await tasks[i]
            except Exception as e:
                print(f"Task failed with exception: {e}")
                result = None
            results.append(result)
            i += 1
            await asyncio.sleep(self.delay_in_seconds)

        return results
