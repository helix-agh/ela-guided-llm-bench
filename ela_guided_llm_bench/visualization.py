from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ela_guided_llm_bench.ela import FEATURES
from ela_guided_llm_bench.experiment import BenchmarkExperiment
from matplotlib.patches import Patch


def compare_contours(
    problem1: Callable,
    problem2: Callable,
    ela_features1: dict[str, float] | None = None,
    ela_features2: dict[str, float] | None = None,
    bounds: tuple[float, float] = (-5, 5),
    resolution: int = 100,
    save_path: str | None = None,
    title: str | None = None,
    dim: int = 2,
) -> None:
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)

    if dim != 2:
        base_point = np.zeros(dim, dtype=float)

        def _make_point(x_val: float, y_val: float) -> np.ndarray:
            point = base_point.copy()
            point[0] = x_val
            point[1] = y_val
            return point

    else:

        def _make_point(x_val: float, y_val: float) -> np.ndarray:
            return np.array([x_val, y_val])

    for i in range(len(x)):
        for j in range(len(y)):
            point = _make_point(X[i, j], Y[i, j])
            Z1[i, j] = problem1(point)
            Z2[i, j] = problem2(point)

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    contour1 = ax1.contourf(X, Y, Z1, levels=20, cmap="viridis")
    ax1.set_xlabel("$x_1$", fontsize=12)
    ax1.set_ylabel("$x_2$", fontsize=12)
    plt.colorbar(contour1, ax=ax1)

    if ela_features1 is not None and ela_features2 is not None:
        ax2 = fig.add_subplot(1, 3, 2)

        features = FEATURES

        values1 = [ela_features1[f] for f in features]
        values2 = [ela_features2[f] for f in features]

        x_pos = np.arange(len(features))

        ax2.plot(x_pos, values1, "x-", color="black", linewidth=1.5, label="Generated")
        ax2.plot(x_pos, values2, "x-", color="gray", alpha=0.7, linewidth=1.5, label="Target")

        if title:
            ax2.set_title(f"Function: {title}")
        else:
            ax2.set_title("ELA Feature Comparison")

        ax2.set_xticks(x_pos)
        shortened_features = [f.split(".")[-1] for f in features]
        ax2.set_xticklabels(shortened_features, rotation=90)

        ax2.grid(True, linestyle="--", alpha=0.7)

        ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    contour2 = ax3.contourf(X, Y, Z2, levels=20, cmap="viridis")
    ax3.set_xlabel("$x_1$", fontsize=12)
    ax3.set_ylabel("$x_2$", fontsize=12)
    plt.colorbar(contour2, ax=ax3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def compare_ela_features(
    ela_features1: dict[str, float],
    ela_features2: dict[str, float],
    save_path: str | None = None,
) -> None:
    features = FEATURES
    values1 = [ela_features1[f] for f in features]
    values2 = [ela_features2[f] for f in features]

    x_pos = np.arange(len(features))

    plt.figure(figsize=(10, 6))
    plt.plot(x_pos, values1, "x-", color="black", linewidth=1.5, label="Generated")
    plt.plot(x_pos, values2, "x-", color="gray", alpha=0.7, linewidth=1.5, label="Target")

    plt.title("ELA Feature Comparison")
    plt.xticks(x_pos, [f.split(".")[-1] for f in features], rotation=90)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_target_values(
    values: list[float],
    title: str = "Target Values",
    xlabel: str = "Iteration",
    ylabel: str = "Value",
    save_path: str | None = None,
) -> None:
    if not values:
        raise ValueError("The values list cannot be empty")

    indices = list(range(len(values)))

    global_min = [values[0]]
    for i in range(1, len(values)):
        global_min.append(min(global_min[-1], values[i]))

    plt.figure(figsize=(10, 6))

    plt.plot(indices, values, "b-o", label="Current Value", alpha=0.7)

    plt.plot(indices, global_min, "r-^", label="Global Minimum")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def boxplot_comparison_benchmark_experiments(
    benchmark_experiments: list[BenchmarkExperiment],
    labels: list[str],
) -> None:
    label_to_best_distances: dict[str, list[float]] = {label: [] for label in labels}

    for benchmark_experiment, label in zip(benchmark_experiments, labels):
        for experiment in benchmark_experiment.experiments:
            best_distance = experiment.best_function_info.distance_to_target
            label_to_best_distances[label].append(best_distance)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in label_to_best_distances.items()]))
    boxplot = df.boxplot(column=labels, rot=45)
    boxplot.set_ylabel("Best Distance to Target Function")
    plt.show()


def compare_sampled_distance_boxplots(
    benchmark_a: BenchmarkExperiment,
    benchmark_b: BenchmarkExperiment,
    label_a: str,
    label_b: str,
    save_path: str | None = None,
) -> None:
    def _collect_distances(benchmark: BenchmarkExperiment) -> dict[int, list[float]]:
        fid_to_distances: dict[int, list[float]] = {}
        for experiment in benchmark.experiments:
            _, all_distances = experiment.sample_ela_features_and_distances()
            fid_to_distances.setdefault(experiment.config.fid, []).extend(all_distances)
        return fid_to_distances

    distances_a = _collect_distances(benchmark_a)
    distances_b = _collect_distances(benchmark_b)

    shared_fids = sorted(set(distances_a.keys()) & set(distances_b.keys()))
    if not shared_fids:
        raise ValueError("Benchmarks must share at least one function id to compare")

    base_positions = np.arange(1, len(shared_fids) + 1)
    half_width = 0.2
    positions_a = base_positions - half_width
    positions_b = base_positions + half_width

    fig_height = max(4, len(shared_fids) * 0.6)
    plt.figure(figsize=(12, fig_height))
    plt.boxplot(
        [distances_a[fid] for fid in shared_fids],
        positions=positions_a,
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor="C0", color="C0", alpha=0.5),
        medianprops=dict(color="C0"),
    )
    plt.boxplot(
        [distances_b[fid] for fid in shared_fids],
        positions=positions_b,
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor="C1", color="C1", alpha=0.5),
        medianprops=dict(color="C1"),
    )

    plt.xticks(base_positions, shared_fids, rotation=45)
    plt.xlabel("BBOB Function ID")
    plt.ylabel("Euclidean Distance to Target ELA")
    plt.title(f"Sampled Distance Comparison: {label_a} vs {label_b}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(
        handles=[Patch(facecolor="C0", label=label_a), Patch(facecolor="C1", label=label_b)],
        loc="best",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def barplot_function_comparison_benchmark_experiments(
    benchmark_experiments: list[BenchmarkExperiment],
    labels: list[str],
    file_path: str | None = None,
) -> None:
    fid_to_best_distance: dict[int, dict[str, float]] = {}

    for benchmark_experiment, label in zip(benchmark_experiments, labels):
        for experiment in benchmark_experiment.experiments:
            best_distance = experiment.best_function_info.distance_to_target
            fid = experiment.config.fid
            if fid not in fid_to_best_distance:
                fid_to_best_distance[fid] = {}
            fid_to_best_distance[fid][label] = best_distance

    function_ids = [f_id for f_id in fid_to_best_distance.keys()]
    function_ids.sort()

    method_distances = {}
    for method in labels:
        method_distances[method] = [fid_to_best_distance[f_id][method] for f_id in function_ids]

    plt.figure(figsize=(12, 6))

    x = np.arange(len(function_ids))
    width = 0.8 / len(labels)

    bars = []
    for i, method in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2) * width
        bar = plt.bar(x + offset, method_distances[method], width, label=method, alpha=0.8)
        bars.append(bar)

    plt.xlabel("BBOB Function ID")
    plt.ylabel("Euclidean Distance to Target ELA")
    plt.xticks(x, function_ids)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
