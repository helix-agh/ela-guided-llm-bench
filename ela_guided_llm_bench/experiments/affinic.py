from typing import Callable

import numpy as np
from ioh import ProblemClass, get_problem


def get_shifted_affinic(
    fid1: int,
    iid1: int,
    fid2: int,
    iid2: int,
    dim: int,
    alpha: float,
    min_value: float = 1e-12,
    max_value: float = 1e12,
) -> Callable:
    f1 = get_problem(fid1, iid1, dim, problem_class=ProblemClass.BBOB)
    f2 = get_problem(fid2, iid2, dim, problem_class=ProblemClass.BBOB)

    o1 = f1.optimum.y
    o2 = f2.optimum.y

    return lambda x: np.exp(
        alpha * np.log(np.clip(f1(x) - o1, min_value, max_value))
        + (1 - alpha) * np.log(np.clip(f2(x - f1.optimum.x + f2.optimum.x) - o2, min_value, max_value))
    )


AFFINIC_PROBLEMS = [
    get_shifted_affinic(1, 1, 2, 1, 2, 0.5),
    get_shifted_affinic(2, 1, 3, 1, 2, 0.5),
    get_shifted_affinic(3, 1, 4, 1, 2, 0.5),
    get_shifted_affinic(4, 1, 5, 1, 2, 0.5),
    get_shifted_affinic(5, 1, 6, 1, 2, 0.5),
    get_shifted_affinic(6, 1, 7, 1, 2, 0.5),
    get_shifted_affinic(7, 1, 8, 1, 2, 0.5),
    get_shifted_affinic(8, 1, 9, 1, 2, 0.5),
    get_shifted_affinic(9, 1, 10, 1, 2, 0.5),
    get_shifted_affinic(10, 1, 11, 1, 2, 0.5),
    get_shifted_affinic(11, 1, 12, 1, 2, 0.5),
    get_shifted_affinic(12, 1, 13, 1, 2, 0.5),
    get_shifted_affinic(13, 1, 14, 1, 2, 0.5),
    get_shifted_affinic(14, 1, 15, 1, 2, 0.5),
    get_shifted_affinic(15, 1, 16, 1, 2, 0.5),
    get_shifted_affinic(16, 1, 17, 1, 2, 0.5),
    get_shifted_affinic(17, 1, 18, 1, 2, 0.5),
    get_shifted_affinic(18, 1, 19, 1, 2, 0.5),
    get_shifted_affinic(19, 1, 20, 1, 2, 0.5),
    get_shifted_affinic(20, 1, 21, 1, 2, 0.5),
    get_shifted_affinic(21, 1, 22, 1, 2, 0.5),
    get_shifted_affinic(22, 1, 23, 1, 2, 0.5),
    get_shifted_affinic(23, 1, 24, 1, 2, 0.5),
    get_shifted_affinic(24, 1, 1, 1, 2, 0.5),
]
