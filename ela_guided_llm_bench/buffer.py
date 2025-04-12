import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

from .function import FunctionInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")

    result = softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])  # type: ignore[misc]
    return result


@dataclass(frozen=True)
class ExperienceBufferConfig:
    functions_per_prompt: int = 5
    num_islands: int = 5
    reset_period: int = 100
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config: ExperienceBufferConfig = ExperienceBufferConfig(),
    ) -> None:
        self._config: ExperienceBufferConfig = config

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    config.functions_per_prompt,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                )
            )
        self._best_score_per_island: list[float] = [-float("inf")] * config.num_islands
        self._best_function_per_island: list[FunctionInfo | None] = [None] * config.num_islands

        self._all_functions: list[FunctionInfo] = []

    def get_prompt_context(self) -> tuple[str, int]:
        island_id = np.random.randint(len(self._islands))
        return self._islands[island_id].get_prompt_context(), island_id

    def _register_function_in_island(
        self,
        function_info: FunctionInfo,
        island_id: int,
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_function(function_info)
        score = function_info.distance_to_target
        if score < self._best_score_per_island[island_id]:
            self._best_function_per_island[island_id] = function_info
            self._best_score_per_island[island_id] = score
            logger.info("Best score of island %d increased to %s", island_id, score)
        self._all_functions.append(function_info)

    def register_function(
        self,
        function_info: FunctionInfo,
        island_id: int | None,
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_function_in_island(function_info, island_id)
        else:
            self._register_function_in_island(function_info, island_id)

        if len(self._all_functions) % self._config.reset_period == 0:
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
            )
            self._best_score_per_island[island_id] = -float("inf")
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_function_per_island[founder_island_id]
            self._register_function_in_island(founder, island_id)


class Island:
    def __init__(
        self,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
    ) -> None:
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._functions_info: list[FunctionInfo] = []
        self._num_programs: int = 0

    def register_function(
        self,
        function_info: FunctionInfo,
    ) -> None:
        self._functions_info.append(function_info)
        self._num_programs += 1

    def get_prompt_context(self) -> str:
        if not self._functions_info:
            return ""
        scores = np.array([function_info.distance_to_target for function_info in self._functions_info])

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (1 - (self._num_programs % period) / period)

        functions_per_prompt = min(len(self._functions_info), self._functions_per_prompt)

        idx = np.random.choice(
            len(self._functions_info),
            size=functions_per_prompt,
            p=_softmax(scores, temperature),
        )
        chosen_functions_info = [self._functions_info[i] for i in idx]
        chosen_functions_info_sorted = sorted(chosen_functions_info, key=lambda x: x.distance_to_target)
        return "\n".join(str(function_info) for function_info in chosen_functions_info_sorted)
