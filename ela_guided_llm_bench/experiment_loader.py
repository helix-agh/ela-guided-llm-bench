from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ela_guided_llm_bench.ela import FEATURES, features_to_array, get_distance, get_ela_features
from ela_guided_llm_bench.function import FunctionInfo
from ela_guided_llm_bench.llm_sr.hyperparameter_optimization import wrap_problem
from umap import UMAP


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
