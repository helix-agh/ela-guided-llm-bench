from typing import Callable

from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
)
from pflacco.sampling import create_initial_sample
from sklearn.preprocessing import MinMaxScaler


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
    ela_level = calculate_ela_level(X, y)
    nbc = calculate_nbc(X, y)
    disp = calculate_dispersion(X, y)
    ic = calculate_information_content(X, y, seed=random_seed)
    return {
        **ic,
        **ela_meta,
        **ela_distr,
        **ela_level,
        **nbc,
        **disp,
        **{"dim": dim},
    }