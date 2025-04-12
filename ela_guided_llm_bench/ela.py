from typing import Callable

import numpy as np
from pflacco.classical_ela_features import calculate_ela_distribution, calculate_ela_meta, calculate_nbc
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.sampling import create_initial_sample

FEATURES = [
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
    "ela_distr.skewness",
    "nbc.nb_fitness.cor",
    "nbc.nn_nb.sd_ratio",
    "fitness_distance.fitness_std",
]


def get_ela_features(problem: Callable, dim: int, random_seed: int = 42) -> dict:
    X = create_initial_sample(
        dim,
        lower_bound=-5,
        upper_bound=5,
        sample_type="lhs",
        sample_coefficient=250,
        seed=random_seed,
    )
    y = X.apply(lambda x: problem(x), axis=1)
    y = (y - y.min()) / (y.max() - y.min())
    ela_meta = calculate_ela_meta(X, y)
    ela_distr = calculate_ela_distribution(X, y)
    nbc = calculate_nbc(X, y)
    fitness_distance = calculate_fitness_distance_correlation(X, y)
    all_features = {
        **ela_meta,
        **ela_distr,
        **nbc,
        **fitness_distance,
    }
    return {
        **{"dim": dim},
        **{k: v for k, v in all_features.items() if k in FEATURES},
    }


def get_distance(features: dict, target_features: dict) -> float:
    features_array = np.array([features[k] for k in FEATURES])
    target_array = np.array([target_features[k] for k in FEATURES])
    return np.linalg.norm(features_array - target_array)  # type: ignore[return-value]
