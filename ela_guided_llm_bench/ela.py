from typing import Callable, Literal

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

ProblemType = Literal["bbob", "ma-bbob", "portal"]

MIN_MAX_VALUES = {
    "bbob": pd.read_csv("./data/ela_min_max.csv"),
    "ma-bbob": pd.read_csv("./data/ma_bbob_ela_min_max.csv"),
    "portal": pd.read_csv("./data/portal_ela_min_max.csv"),
}


def normalize_features(features: dict, dim: int, problem_type: ProblemType = "bbob") -> dict:
    min_max_df = MIN_MAX_VALUES[problem_type]
    if problem_type == "bbob":
        min_feature_values = min_max_df[(min_max_df["dim"] == dim) & (min_max_df["type"] == "min")].iloc[0].to_dict()
        max_feature_values = min_max_df[(min_max_df["dim"] == dim) & (min_max_df["type"] == "max")].iloc[0].to_dict()
    else:  # ma-bbob / portal — no dim filter
        min_feature_values = min_max_df[min_max_df["type"] == "min"].iloc[0].to_dict()
        max_feature_values = min_max_df[min_max_df["type"] == "max"].iloc[0].to_dict()

    normalized_features = {}
    for feature in FEATURES:
        min_value = min_feature_values[feature]
        max_value = max_feature_values[feature]
        normalized_features[feature] = (features[feature] - min_value) / (max_value - min_value)
    return normalized_features


def get_ela_features(
    problem: Callable,
    dim: int,
    random_seed: int = 42,
    problem_type: ProblemType = "bbob",
) -> dict:
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
    normalized_features = normalize_features(all_features, dim, problem_type)

    return {
        **{"dim": dim},
        **normalized_features,
    }


def get_target_ela_features(
    fid: int | None = None,
    iid: int | None = None,
    dim: int = 2,
    problem_type: ProblemType = "bbob",
    instance: str | None = None,
) -> dict:
    if problem_type == "bbob":
        if fid is None or iid is None:
            raise ValueError("fid and iid are required for bbob problem type")
        df = pd.read_csv("./data/ela_mean_values.csv")
        row = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["dim"] == dim)].iloc[0]
    elif problem_type == "portal":
        if instance is None:
            raise ValueError("instance is required for portal problem type")
        df = pd.read_csv("./data/portal_ela_mean_values.csv")
        row = df[df["instance"] == instance].iloc[0]
    else:
        df = pd.read_csv("./data/ma_bbob_ela_mean_values.csv")
        row = df[(df["fid"] == fid) & (df["dim"] == dim)].iloc[0]

    all_features = {feature: row[feature] for feature in FEATURES}
    normalized_features = normalize_features(all_features, dim, problem_type)
    return {
        **{"dim": dim},
        **normalized_features,
    }


def features_to_array(features: dict) -> np.ndarray:
    return np.array([features[k] for k in FEATURES])


def get_distance(features: dict, target_features: dict) -> float:
    features_array = features_to_array(features)
    target_array = features_to_array(target_features)
    return np.power(np.sum(np.power(features_array - target_array, 2)), 0.5)  # type: ignore[return-value]
