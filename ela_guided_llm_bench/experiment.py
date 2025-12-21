import itertools
import json
import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ela_guided_llm_bench.ela import FEATURES, features_to_array, get_distance, get_ela_features
from ela_guided_llm_bench.experiment_config import ExperimentConfig
from ela_guided_llm_bench.function import FunctionInfo
from ioh import ProblemClass, get_problem
from umap import UMAP


def save_to_df(generated_functions: list[list[FunctionInfo]], save_path: str) -> None:
    print(f"Saving results to {save_path}...")
    rows = []
    for iteration, functions in enumerate(generated_functions, start=1):
        for info in functions:
            if info is not None:
                rows.append(
                    {
                        "source_code": info.source_code,
                        "distance_to_target": info.distance_to_target,
                        "description": info.description,
                        "initial_distance_to_target": info.initial_distance_to_target,
                        "number_of_params": info.number_of_params,
                        "params": info.params,
                        "iteration": iteration,
                    }
                    | info.ela_features
                )
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


def read_from_df(df: pd.DataFrame, problem_with_params: bool = False) -> list[list[FunctionInfo]]:
    function_infos = [FunctionInfo.from_row(row, problem_with_params) for _, row in df.iterrows()]

    if "iteration" not in df.columns:
        return [function_infos]

    iterations = [row["iteration"] for _, row in df.iterrows()]
    result: list[list[FunctionInfo]] = [[] for _ in range(1, (max(iterations) + 1))]
    for iteration, function_info in zip(iterations, function_infos):
        result[iteration - 1].append(function_info)
    return result


@dataclass
class Experiment:
    function_infos: list[list[FunctionInfo]]
    target_ela_features: dict[str, float]
    config: ExperimentConfig

    @classmethod
    def from_dir(
        cls,
        dir_name: str,
        problem_with_params: bool = False,
        parent_dir: str | None = None,
    ) -> "Experiment":
        config = ExperimentConfig.from_dir(dir_name, parent_dir=parent_dir)
        df = pd.read_csv(config.csv_path)
        function_infos = read_from_df(df, problem_with_params)
        target_ela_features = json.load(open(config.target_ela_features_path))
        return cls(
            function_infos=function_infos,
            target_ela_features=target_ela_features,
            config=config,
        )

    def save_to_dir(self) -> None:
        os.makedirs(self.config.dir_name, exist_ok=True)
        with open(self.config.target_ela_features_path, "w") as f:
            json.dump(self.target_ela_features, f)

        save_to_df(self.function_infos, self.config.csv_path)

    @property
    def best_function_info(self) -> FunctionInfo:
        return min(self.all_function_infos, key=lambda x: x.distance_to_target)

    @property
    def all_function_infos(self) -> list[FunctionInfo]:
        return [function_info for functions in self.function_infos for function_info in functions]

    def plot_ela_scatter(self):
        # TODO: add all ELA features for BBOB and compare them against BBOB
        try:
            umap = UMAP(n_components=2)
            all_features = []
            for function_info in self.all_function_infos:
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
            function = (
                self.best_function_info.function_with_params
                if self.best_function_info.function_with_params is not None
                else self.best_function_info.function
            )
            ela_features = get_ela_features(function, self.config.dim, random_seed)
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
        plt.title(f"ELA Features Distribution - Function {self.config.fid}")
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
        plt.title(f"Distance to Target - Function {self.config.fid}")
        plt.show()

    def plot_operators(self):
        # TODO: test if works correctly
        markers = ["o", "s", "D", "^", "P"]
        colors = ["blue", "green", "red", "purple", "brown"]

        marker_cycle = itertools.cycle(markers)
        color_cycle = itertools.cycle(colors)

        min_results = [min([fi.distance_to_target for fi in funcs]) for funcs in self.function_infos]
        for idx in range(len(min_results)):
            plt.scatter(
                [idx],
                [min_results[idx]],
                color=next(color_cycle),
                marker=next(marker_cycle),
                s=80,
            )
        plt.plot(range(len(min_results)), min_results, color="gray", linewidth=1)
        plt.show()


@dataclass
class BenchmarkExperiment:
    experiments: list[Experiment]

    @property
    def config(self) -> ExperimentConfig:
        return self.experiments[0].config

    @property
    def method(self) -> str:
        return self.config.method

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def problems(self) -> list[tuple[int, Callable[[np.ndarray], float]]]:
        return [(experiment.config.fid, experiment.best_function_info.function) for experiment in self.experiments]

    @classmethod
    def from_dir(
        cls,
        dir_name: str,
    ) -> "BenchmarkExperiment":
        experiments = [Experiment.from_dir(d, parent_dir=dir_name) for d in os.listdir(dir_name)]
        return cls(experiments=experiments)

    def plot_sampled_distances(self, file_path: str | None = None) -> None:
        all_distances_list = []
        labels = []

        for experiment in self.experiments:
            _, all_distances = experiment.sample_ela_features_and_distances()
            all_distances_list.append(all_distances)
            labels.append(experiment.config.fid)

        plt.figure(figsize=(6, 10))
        plt.boxplot(all_distances_list, labels=labels, vert=False)
        plt.ylabel("FID")
        plt.xlabel("Euclidean Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if file_path is not None:
            plt.savefig(file_path, dpi=300)
        plt.show()

    def plot_contour_grid(
        self,
        function_ids: list[int],
        bounds: tuple[float, float] = (-5, 5),
        resolution: int = 100,
        file_path: str | None = None,
    ) -> None:
        fid_to_experiment = {experiment.config.fid: experiment for experiment in self.experiments}
        missing = [fid for fid in function_ids if fid not in fid_to_experiment]
        if missing:
            raise ValueError(f"Function ids {missing} are not available in the benchmark")

        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(len(function_ids), 3, figsize=(15, 5 * len(function_ids)))
        axes = np.atleast_2d(axes)

        for row_idx, fid in enumerate(function_ids):
            experiment = fid_to_experiment[fid]
            best_function = experiment.best_function_info
            target_problem = get_problem(
                fid,
                experiment.config.iid,
                experiment.config.dim,
                problem_class=ProblemClass.BBOB,
            )

            Z_generated = self._evaluate_problem_on_grid(best_function.function, X, Y, experiment.config.dim)
            Z_target = self._evaluate_problem_on_grid(target_problem, X, Y, experiment.config.dim)

            ax_generated, ax_features, ax_target = axes[row_idx]

            contour_generated = ax_generated.contourf(X, Y, Z_generated, levels=20, cmap="viridis")
            ax_generated.set_xlabel("$x_1$")
            ax_generated.set_ylabel("$x_2$")
            ax_generated.set_title(f"FID {fid} - Generated")
            plt.colorbar(contour_generated, ax=ax_generated)

            features = FEATURES
            feature_positions = np.arange(len(features))
            generated_values = [best_function.ela_features[f] for f in features]
            target_values = [experiment.target_ela_features[f] for f in features]

            ax_features.plot(
                feature_positions,
                generated_values,
                "x-",
                color="black",
                linewidth=1.5,
                label="Generated",
            )
            ax_features.plot(
                feature_positions,
                target_values,
                "x-",
                color="gray",
                alpha=0.7,
                linewidth=1.5,
                label="Target",
            )
            ax_features.set_xticks(feature_positions)
            ax_features.set_xticklabels([f.split(".")[-1] for f in features], rotation=90)
            ax_features.set_title("ELA Feature Comparison")
            ax_features.grid(True, linestyle="--", alpha=0.7)
            ax_features.legend()

            contour_target = ax_target.contourf(X, Y, Z_target, levels=20, cmap="viridis")
            ax_target.set_xlabel("$x_1$")
            ax_target.set_ylabel("$x_2$")
            ax_target.set_title(f"FID {fid} - Target")
            plt.colorbar(contour_target, ax=ax_target)

        plt.tight_layout()
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _evaluate_problem_on_grid(problem, X: np.ndarray, Y: np.ndarray, dim: int) -> np.ndarray:
        Z = np.zeros_like(X)
        if dim != 2:
            base_point = np.zeros(dim, dtype=float)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = base_point.copy()
                    point[0] = X[i, j]
                    point[1] = Y[i, j]
                    Z[i, j] = problem(point)
        else:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = problem(np.array([X[i, j], Y[i, j]]))
        return Z
