from abc import ABC, abstractmethod


class BaseExecutor(ABC):
    @property
    @abstractmethod
    def delay_in_seconds(self) -> float:
        raise NotImplementedError

    @abstractmethod
    async def process_tasks_in_batches(self, tasks: list) -> list:
        pass

    @abstractmethod
    async def log_key_usage_stats(self):
        pass
