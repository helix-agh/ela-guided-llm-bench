from typing import Callable, Literal

import cma
import numpy as np
from scipy import optimize

from .ela import get_distance, get_ela_features

LOWER_BOUND = 0.0
UPPER_BOUND = 1.0


def wrap_problem(
    problem: Callable[[np.ndarray, np.ndarray], float], params: np.ndarray
) -> Callable[[np.ndarray], float]:
    def wrapped_problem(x: np.ndarray) -> float:
        return problem(x, params)

    return wrapped_problem


class HyperparameterOptimizer:
    def __init__(
        self,
        problem: Callable[[np.ndarray, np.ndarray], float],
        dim: int,
        number_of_params: int,
        target_ela_features: dict[str, float],
        random_seed: int = 42,
    ):
        self.problem = problem
        self.dim = dim
        self.number_of_params = number_of_params
        self.target_ela_features = target_ela_features
        self.random_seed = random_seed

    def wrapped_problem(self, params: np.ndarray) -> Callable[[np.ndarray], float]:
        def wrapped_problem(x: np.ndarray) -> float:
            return self.problem(x, params)

        return wrapped_problem

    def objective_function(self, params: np.ndarray) -> float:
        params = np.clip(params, LOWER_BOUND, UPPER_BOUND)

        func_with_fixed_params = wrap_problem(self.problem, params)

        calculated_ela_features = get_ela_features(func_with_fixed_params, self.dim, self.random_seed)

        distance = get_distance(calculated_ela_features, self.target_ela_features)

        if not np.isfinite(distance):
            print(f"Warning: Calculated non-finite distance for params {params}. Returning large penalty.")
            distance = 1e6
        return distance

    def optimize(
        self,
        initial_params: np.ndarray,
        max_evals: int = 100,
        algorithm: Literal["CMA-ES", "L-BFGS-B"] = "CMA-ES",
    ) -> tuple[np.ndarray, float]:
        if algorithm == "CMA-ES":
            return self.optimize_with_cma_es(initial_params, max_evals)
        elif algorithm == "L-BFGS-B":
            return self.optimize_with_lbfgs(initial_params, max_evals)
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

    def optimize_with_cma_es(self, initial_params: np.ndarray, max_evals: int = 100) -> tuple[np.ndarray, float]:
        bounds = [LOWER_BOUND, UPPER_BOUND]

        options = {"bounds": bounds, "maxfevals": max_evals, "verbose": -9}

        result = cma.fmin2(
            self.objective_function,
            initial_params,
            options=options,
            sigma0=0.3,
            restarts=10,
        )

        best_params = result[0]
        best_distance = result[1].result.fbest
        best_params = np.clip(best_params, LOWER_BOUND, UPPER_BOUND)
        return best_params, best_distance

    def optimize_with_lbfgs(self, initial_params: np.ndarray, max_evals: int = 100) -> tuple[np.ndarray, float]:
        bounds = [LOWER_BOUND, UPPER_BOUND]
        result = optimize.minimize(
            self.objective_function,
            initial_params,
            method="L-BFGS-B",
            bounds=[bounds for _ in range(self.number_of_params)],
            options={"maxiter": max_evals},
        )
        return result.x, result.fun
