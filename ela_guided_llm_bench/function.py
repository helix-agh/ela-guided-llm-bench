import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from .ela import get_distance, get_ela_features
from .hyperparameter_optimization import HyperparameterOptimizer, wrap_problem

logger = logging.getLogger(__name__)


def features_to_prompt(features: dict) -> str:
    rounded_features = {k: round(v, 2) for k, v in features.items()}
    return json.dumps(rounded_features)


DEFAULT_PARAM_VALUE = 0.5
DEFAULT_NUMBER_OF_PARAMS = 5


@dataclass
class FunctionInfo:
    function: Callable
    source_code: str
    description: str
    ela_features: dict[str, float]
    distance_to_target: float
    number_of_params: int | None = None
    initial_distance_to_target: float | None = None
    params: np.ndarray | None = None
    function_with_params: Callable | None = None

    def __str__(self) -> str:
        ela_features_formatted = features_to_prompt(self.ela_features)
        ela_features_str = f"**ELA Features:**\n{ela_features_formatted}"
        source_code_formatted = f"```python\n{self.source_code.strip()}\n```"
        source_code_str = f"**Previously Generated Function:**\n{source_code_formatted}"
        error_str = f"**Error (Previous ELA - Target ELA):**\n{round(self.distance_to_target, 2)}"
        params_formatted = [round(param, 2) for param in self.params] if self.params is not None else []
        params_str = f"**Tuned parameters:**\n{params_formatted}" if self.params is not None else ""
        return f"<function_info>{source_code_str}\n{ela_features_str}\n{error_str}\n{params_str}\n</function_info>"

    def optimize_params(
        self,
        target_ela_features: dict[str, float],
        max_evals: int = 100,
        ela_dim: int = 2,
        algorithm: Literal["CMA-ES", "L-BFGS-B"] = "CMA-ES",
    ) -> None:
        optimizer = HyperparameterOptimizer(
            problem=self.function,
            dim=ela_dim,
            number_of_params=self.number_of_params,
            target_ela_features=target_ela_features,
        )
        final_params, final_distance = optimizer.optimize(self.params, max_evals=max_evals, algorithm=algorithm)
        if final_distance < self.distance_to_target:
            print(f"Improved from {self.distance_to_target} to {final_distance}")
            self.params = final_params
            self.ela_features = get_ela_features(
                optimizer.wrapped_problem(final_params),
                ela_dim,
            )
            self.distance_to_target = get_distance(self.ela_features, target_ela_features)
            self.function = wrap_problem(self.function_with_params, final_params)

    def sample_features(
        self,
        target_ela_features: dict[str, float],
        n_samples: int = 100,
        ela_dim: int = 2,
        start_seed: int = 42,
    ) -> tuple[list[dict[str, float]], list[float]]:
        ela_features_list = []
        distances = []

        for i in range(n_samples):
            seed = start_seed + i
            features = get_ela_features(self.function, ela_dim, random_seed=seed)
            distance = get_distance(features, target_ela_features)

            ela_features_list.append(features)
            distances.append(distance)

        return ela_features_list, distances


class FunctionParser:
    def __init__(
        self,
        ela_dim: int,
        target_ela_features: dict[str, float],
        random_seed: int = 42,
        problem_with_params: bool = False,
        max_evals: int = 100,
        algorithm: Literal["CMA-ES", "L-BFGS-B"] = "CMA-ES",
    ):
        self.ela_dim = ela_dim
        self.random_seed = random_seed
        self.target_ela_features = target_ela_features
        self.problem_with_params = problem_with_params
        self.max_evals = max_evals
        self.algorithm = algorithm

    def parse(self, model_response: str) -> FunctionInfo | None:
        function_str = self.extract_code(model_response)
        if not function_str:
            logger.error("No function found in response")
            return None

        if not self.validate_function_syntax(function_str):
            logger.error("Invalid function syntax")
            return None

        if "import numpy" not in function_str:
            function_str = "import numpy as np\n\n" + function_str

        docstring = self.extract_docstring(function_str)
        namespace: dict[str, Any] = {}
        exec(function_str, namespace)
        if self.problem_with_params:
            number_of_params = self.extract_number_of_params(function_str)
            initial_params = np.full(number_of_params, DEFAULT_PARAM_VALUE)
            optimizer = HyperparameterOptimizer(
                problem=namespace["problem"],
                dim=self.ela_dim,
                number_of_params=number_of_params,
                target_ela_features=self.target_ela_features,
                random_seed=self.random_seed,
            )
            initial_distance_to_target = optimizer.objective_function(initial_params)
            wrapped_problem = optimizer.wrapped_problem(initial_params)
            ela_features = get_ela_features(wrapped_problem, self.ela_dim, self.random_seed)
            final_params, final_distance = optimizer.optimize(
                initial_params, max_evals=self.max_evals, algorithm=self.algorithm
            )
            final_ela_features = get_ela_features(
                optimizer.wrapped_problem(final_params),
                self.ela_dim,
                self.random_seed,
            )
            distance_to_target = get_distance(final_ela_features, self.target_ela_features)
            final_wrapped_problem = optimizer.wrapped_problem(final_params)
            return FunctionInfo(
                function=final_wrapped_problem,
                source_code=function_str,
                description=docstring,
                ela_features=final_ela_features,
                distance_to_target=distance_to_target,
                initial_distance_to_target=initial_distance_to_target,
                number_of_params=number_of_params,
                params=final_params,
                function_with_params=namespace["problem"],
            )
        else:
            ela_features = get_ela_features(namespace["problem"], self.ela_dim, self.random_seed)
            distance_to_target = get_distance(ela_features, self.target_ela_features)
            return FunctionInfo(
                function=namespace["problem"],
                source_code=function_str,
                description=docstring,
                ela_features=ela_features,
                distance_to_target=distance_to_target,
            )

    def validate_function_syntax(self, function_str: str) -> bool:
        try:
            ast.parse(function_str)
            return True
        except SyntaxError:
            return False

    def extract_code(self, text: str) -> str | None:
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None

    def extract_docstring(self, function_str: str) -> str | None:
        pattern = r"\"\"\"(.*?)\"\"\""
        match = re.search(pattern, function_str, re.DOTALL)
        return match.group(1) if match else None

    def extract_number_of_params(self, function_str: str) -> int:
        pattern = r"#\s*n_params\s*=\s*(\d+)"
        match = re.search(pattern, function_str)
        if not match:
            return DEFAULT_NUMBER_OF_PARAMS
        return int(match.group(1))
