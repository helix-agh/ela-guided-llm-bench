from typing import Callable

import numpy as np
import pandas as pd
from pflacco.classical_ela_features import calculate_ela_distribution, calculate_ela_meta, calculate_nbc
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.sampling import create_initial_sample

FEATURES = [
    "ela_distr.skewness",
    "fitness_distance.fitness_std",
    "nbc.nn_nb.sd_ratio",
    "nbc.nb_fitness.cor",
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
]

MIN_MAX_VALUES = pd.read_csv("./02_ela_min_max.csv")


def normalize_features(features: dict, dim: int) -> dict:
    min_feature_values = (
        MIN_MAX_VALUES[(MIN_MAX_VALUES["dim"] == dim) & (MIN_MAX_VALUES["type"] == "min")].iloc[0].to_dict()
    )
    max_feature_values = (
        MIN_MAX_VALUES[(MIN_MAX_VALUES["dim"] == dim) & (MIN_MAX_VALUES["type"] == "max")].iloc[0].to_dict()
    )
    normalized_features = {}
    for feature in FEATURES:
        min_value = min_feature_values[feature]
        max_value = max_feature_values[feature]
        normalized_features[feature] = (features[feature] - min_value) / (max_value - min_value)
    return normalized_features


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
    normalized_features = normalize_features(all_features, dim)

    return {
        **{"dim": dim},
        **normalized_features,
    }


def get_distance(features: dict, target_features: dict) -> float:
    features_array = np.array([features[k] for k in FEATURES])
    target_array = np.array([target_features[k] for k in FEATURES])
    return np.power(np.sum(np.power(features_array - target_array, 2)), 0.5)  # type: ignore[return-value]
