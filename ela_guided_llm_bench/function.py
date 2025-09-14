import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

from .ela import FEATURES, features_to_array, get_distance, get_ela_features
from .hyperparameter_optimization import HyperparameterOptimizer, wrap_problem

logger = logging.getLogger(__name__)


def features_to_prompt(features: dict) -> str:
    rounded_features = {k: round(v, 3) for k, v in features.items()}
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
        source_code_str = f"**Previous Code:**\n{source_code_formatted}"
        error_str = f"**Error (Previous ELA - Target ELA):**\n{round(self.distance_to_target, 3)}"
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
            problem=self.function_with_params,
            dim=ela_dim,
            number_of_params=self.number_of_params,
            target_ela_features=target_ela_features,
        )
        final_params, final_distance = optimizer.optimize(self.params, max_evals=max_evals, algorithm=algorithm)
        if final_distance < self.distance_to_target:
            print(f"Improved from {self.distance_to_target} to {final_distance}")
            self.params = final_params
            final_wrapped_problem = wrap_problem(self.function_with_params, final_params)
            self.ela_features = get_ela_features(
                final_wrapped_problem,
                ela_dim,
            )
            self.distance_to_target = get_distance(self.ela_features, target_ela_features)
            self.function = final_wrapped_problem

    @property
    def llamea_summary(self) -> str:
        return f"""<function_info>
Description: {self.description}
ELA Features: {features_to_prompt(self.ela_features)}
Error (Distance to Target): {round(self.distance_to_target, 3)}
</function_info>
"""

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

        description = self.extract_description(model_response) or self.extract_docstring(function_str)
        namespace: dict[str, Any] = {}
        try:
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
                wrapped_problem = wrap_problem(namespace["problem"], initial_params)
                ela_features = get_ela_features(wrapped_problem, self.ela_dim, self.random_seed)
                final_params, final_distance = optimizer.optimize(
                    initial_params, max_evals=self.max_evals, algorithm=self.algorithm
                )
                final_wrapped_problem = wrap_problem(namespace["problem"], final_params)
                final_ela_features = get_ela_features(
                    final_wrapped_problem,
                    self.ela_dim,
                    self.random_seed,
                )
                distance_to_target = get_distance(final_ela_features, self.target_ela_features)
                return FunctionInfo(
                    function=final_wrapped_problem,
                    source_code=function_str,
                    description=description,
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
                    description=description,
                    ela_features=ela_features,
                    distance_to_target=distance_to_target,
                )
        except Exception as e:
            print(f"Error executing function: {e}")
            return None

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

    def extract_description(self, function_str: str) -> str | None:
        pattern = r"# Description: (.*)"
        match = re.search(pattern, function_str)
        return match.group(1) if match else None


def save_to_df(generated_functions_info: list[FunctionInfo], save_path: str) -> None:
    rows = []
    for info in generated_functions_info:
        rows.append(
            {
                "source_code": info.source_code,
                "distance_to_target": info.distance_to_target,
                "description": info.description,
                "initial_distance_to_target": info.initial_distance_to_target,
                "number_of_params": info.number_of_params,
                "params": info.params,
            }
            | info.ela_features
        )
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


def row_to_function_info(row: pd.Series, problem_with_params: bool = False) -> FunctionInfo:
    source_code = row["source_code"]

    namespace = {}  # type: ignore[var-annotated]
    exec(source_code, namespace)

    ela_features = {}
    for key in row.index:
        if key in [
            "source_code",
            "distance_to_target",
            "description",
            "initial_distance_to_target",
            "number_of_params",
            "params",
        ]:
            continue
        ela_features[key] = row[key]

    params = None
    if "params" in row and row["params"] is not None:
        if isinstance(row["params"], str):
            params_str = row["params"].strip("[]")
            if params_str:
                if "," in params_str:
                    params = np.array([float(x.strip()) for x in params_str.split(",")])
                else:
                    params = np.array([float(x) for x in params_str.split()])
        else:
            params = row["params"]

    if problem_with_params:
        function_with_params = namespace["problem"]
        function = wrap_problem(namespace["problem"], params)
    else:
        function_with_params = None
        function = namespace["problem"]

    description = row["description"] if "description" in row else ""

    return FunctionInfo(
        function=function,
        function_with_params=function_with_params,
        source_code=source_code,
        description=description,
        ela_features=ela_features,
        distance_to_target=row["distance_to_target"],
        initial_distance_to_target=row["initial_distance_to_target"],
        number_of_params=row["number_of_params"] if "number_of_params" in row else 0,
        params=params,
    )


def load_from_df(file_path: str, problem_with_params: bool = False) -> list[FunctionInfo]:
    df = pd.read_csv(file_path)
    return [row_to_function_info(row, problem_with_params) for _, row in df.iterrows()]


@dataclass
class Experiment:
    method: str
    function_id: int
    function_infos: list[FunctionInfo]
    model: str
    target_ela_features: dict[str, float]

    @property
    def best_function_info(self) -> FunctionInfo:
        return min(self.function_infos, key=lambda x: x.distance_to_target)

    def plot_ela_scatter(self):
        # TODO: add all ELA features for BBOB and compare them against BBOB
        try:
            umap = UMAP(n_components=2)
            all_features = []
            for function_info in self.function_infos:
                features = features_to_array(function_info.ela_features)
                all_features.append(features)
            all_features = np.array(all_features)
            all_features_umap = umap.fit_transform(all_features)
            plt.scatter(
                all_features_umap[:, 0],
                all_features_umap[:, 1],
                color="blue",
                s=50,
                zorder=5,
                label="Generated Functions",
            )
            target_ela_features_umap = umap.transform(features_to_array(self.target_ela_features).reshape(1, -1))
            plt.scatter(
                target_ela_features_umap[:, 0],
                target_ela_features_umap[:, 1],
                color="red",
                s=50,
                zorder=5,
                label="Target",
            )
            plt.show()
        except Exception as e:
            print(f"Error plotting ELA scatter: {e}")

    def sample_ela_features_and_distances(self, n_samples: int = 30):
        all_features = []
        all_distances = []
        for random_seed in range(n_samples):
            ela_features = get_ela_features(self.best_function_info.function_with_params, 2, random_seed)
            features_array = features_to_array(ela_features)
            all_features.append(features_array)
            distance = get_distance(ela_features, self.target_ela_features)
            all_distances.append(distance)
        return np.array(all_features), all_distances

    def plot_ela_boxplot(self):
        target_features_array = features_to_array(self.target_ela_features)
        original_features_array = features_to_array(self.best_function_info.ela_features)
        all_features, all_distances = self.sample_ela_features_and_distances()

        # Create figure with proper size for feature names
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_features)
        plt.scatter(
            range(1, len(original_features_array) + 1),
            original_features_array,
            color="red",
            s=50,
            zorder=5,
            label="Generated Function",
        )
        plt.scatter(
            range(1, len(target_features_array) + 1),
            target_features_array,
            color="blue",
            s=50,
            zorder=5,
            label="Target",
        )
        plt.xticks(range(1, len(FEATURES) + 1), FEATURES, rotation=45, ha="right")
        plt.xlabel("ELA Features")
        plt.ylabel("Feature Values")
        plt.title(f"ELA Features Distribution - Function {self.function_id}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.boxplot(all_distances)
        plt.scatter(
            [1],
            [self.best_function_info.distance_to_target],
            color="red",
            s=50,
            zorder=5,
        )
        plt.xlabel("Distance Distribution")
        plt.ylabel("Distance to Target")
        plt.title(f"Distance to Target - Function {self.function_id}")
        plt.show()


@dataclass
class BenchmarkExperiment:
    experiments: list[Experiment]

    @property
    def method(self) -> str:
        return self.experiments[0].method

    @property
    def model(self) -> str:
        return self.experiments[0].model

    def plot_sampled_distances(self, path: str | None = None) -> None:
        all_distances_list = []
        labels = []

        for experiment in self.experiments:
            _, all_distances = experiment.sample_ela_features_and_distances()
            all_distances_list.append(all_distances)
            labels.append(experiment.function_id)

        plt.figure(figsize=(6, 10))
        plt.boxplot(all_distances_list, labels=labels, vert=False)
        plt.ylabel("FID")
        plt.xlabel("Euclidean Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if path is not None:
            plt.savefig(path, dpi=300)
        plt.show()
