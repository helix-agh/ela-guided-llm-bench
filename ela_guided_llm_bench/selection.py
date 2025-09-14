import random

import numpy as np
from ela_guided_llm_bench.function import FunctionInfo


def select_examples_by_roulette(examples: list[FunctionInfo], k: int = 3) -> list[FunctionInfo]:
    k = min(k, len(examples))
    if k == 0:
        return []
    distances_to_target = [example.distance_to_target for example in examples]
    weights = 1.0 / (np.array(distances_to_target) + 1e-10)
    weights = weights / np.sum(weights)
    sampled_indices = np.random.choice(range(len(distances_to_target)), size=k, replace=False, p=weights)
    return [examples[i] for i in sampled_indices]


def parent_selection(pop: list[FunctionInfo], m: int) -> list[FunctionInfo]:
    sorted_pop = sorted([x for x in pop if x is not None], key=lambda x: x.distance_to_target)
    ranks = [i for i in range(len(sorted_pop))]
    probs = [1 / (rank + 1 + len(sorted_pop)) for rank in ranks]
    parents = random.choices(sorted_pop, weights=probs, k=min(m, len(sorted_pop)))
    return parents
