import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import nevergrad as ng
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger.setLevel(logging.INFO)

ALGORITHMS = {
    "EMNA": ng.optimizers.EMNA,
    "NelderMead": ng.optimizers.NelderMead,
    "DE": ng.optimizers.DE,
    "PSO": ng.optimizers.PSO,
    "DiagonalCMA": ng.optimizers.DiagonalCMA,  # dCMA-ES
    "Cobyla": ng.optimizers.Cobyla,
    "RandomSearch": ng.optimizers.RandomSearch,
}


@dataclass
class AlgorithmProblemResult:
    algorithm_name: str
    problem_name: str
    problem_type: str
    dim: int
    budget: int
    fitness_values: list[float]

    def to_rows(self) -> list[dict]:
        return [
            {
                "algorithm_name": self.algorithm_name,
                "problem_name": self.problem_name,
                "problem_type": self.problem_type,
                "dim": self.dim,
                "budget": self.budget,
                "fitness_value": fitness,
            }
            for fitness in self.fitness_values
        ]


def _run_problem_task(
    problem_func: Callable[[np.ndarray], float],
    problem_name: str,
    problem_type: str,
    algorithms: dict,
    dim: int,
    bounds: tuple[float, float],
    budget_multiplier: int,
    n_replications: int,
) -> list[AlgorithmProblemResult]:
    budget = budget_multiplier * dim
    results = []

    for algo_name, algo_class in algorithms.items():
        fitness_values = []
        for rep in range(n_replications):
            seed = rep * 1000 + hash(problem_name) % 1000
            np.random.seed(seed)

            parametrization = ng.p.Array(shape=(dim,)).set_bounds(bounds[0], bounds[1])
            optimizer_class = algo_class
            if isinstance(optimizer_class, type) and issubclass(
                optimizer_class, ng.optimizers.base.ConfiguredOptimizer
            ):
                optimizer_class = optimizer_class()
            optimizer = optimizer_class(parametrization=parametrization, budget=budget)

            rep_start_time = time.time()
            try:
                for _ in range(budget):
                    x = optimizer.ask()
                    fitness = problem_func(x.value)
                    optimizer.tell(x, fitness)

                recommendation = optimizer.recommend()
                best_fitness = problem_func(recommendation.value)
                fitness_values.append(best_fitness)
            except Exception as e:
                logger.warning(f"Optimization failed for {algo_name}: {e}")
                fitness_values.append(float("inf"))

            if rep == 0:
                elapsed = time.time() - rep_start_time
                logger.info(f"{algo_name} on {problem_name} (d={dim}): first rep took {elapsed:.2f}s")

        result = AlgorithmProblemResult(
            algorithm_name=algo_name,
            problem_name=problem_name,
            problem_type=problem_type,
            dim=dim,
            fitness_values=fitness_values,
            budget=budget,
        )
        results.append(result)
        logger.info(
            f"{algo_name} on {problem_name} ({problem_type}, d={dim}): "
            f"median={np.median(result.fitness_values):.6f}"
        )

    return results


@dataclass
class AlgorithmRunner:
    dim: int = 2
    budget_multiplier: int = 10000
    n_replications: int = 10
    algorithms: dict = field(default_factory=lambda: ALGORITHMS.copy())
    bounds: tuple[float, float] = (-5.0, 5.0)
    results: list[AlgorithmProblemResult] = field(default_factory=list)
    max_workers: int = 8

    def benchmark_problems(
        self,
        problems: list[tuple[str, Callable[[np.ndarray], float]]],
        problem_type: str,
    ) -> list[AlgorithmProblemResult]:
        logger.info(
            f"Starting benchmark: {len(problems)} problems, "
            f"{len(self.algorithms)} algorithms, "
            f"{self.n_replications} replications each."
        )
        all_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _run_problem_task,
                    func,
                    name,
                    problem_type,
                    self.algorithms,
                    self.dim,
                    self.bounds,
                    self.budget_multiplier,
                    self.n_replications,
                ): name
                for name, func in problems
            }

            for future in as_completed(futures):
                problem_name = futures[future]
                results = future.result()
                all_results.extend(results)
                self.results.extend(results)
                logger.info(f"Completed benchmarking problem: {problem_name}")

        return all_results

    def save_results(self, dir_path: Path) -> None:
        with open(dir_path / "results.pkl", "wb") as f:
            pickle.dump(self.results, f)

        pd.DataFrame([row for result in self.results for row in result.to_rows()]).to_csv(
            dir_path / "results.csv", index=False
        )

    def load_results(self, dir_path: Path) -> list[AlgorithmProblemResult]:
        with open(dir_path / "results.pkl", "rb") as f:
            self.results = pickle.load(f)
        return self.results
