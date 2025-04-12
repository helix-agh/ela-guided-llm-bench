import numpy as np

from .function import FunctionInfo


def select_examples_by_roulette(examples: list[FunctionInfo], k: int = 3) -> list[FunctionInfo]:
    distances_to_target = [example.distance_to_target for example in examples]
    weights = 1.0 / (np.array(distances_to_target) + 1e-10)
    weights = weights / np.sum(weights)
    sampled_indices = np.random.choice(range(len(distances_to_target)), size=k, replace=False, p=weights)
    return [examples[i] for i in sampled_indices]
