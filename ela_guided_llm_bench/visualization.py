from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
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


def compare_sampled_distance_boxplots(
    benchmark_experiments: list[BenchmarkExperiment],
    labels: list[str],
    n_samples: int = 50,
    file_path: str | None = None,
) -> list[dict[int, list[float]]]:
    BBOB_GROUPS = [(1, 5), (6, 9), (10, 14), (15, 19), (20, 24)]
    GROUP_COLORS = ["#2E5A87", "#4A7C59", "#8B6914", "#7B3B3B", "#5B4B8A"]

    def _get_group_index(fid: int) -> int:
        for i, (start, end) in enumerate(BBOB_GROUPS):
            if start <= fid <= end:
                return i
        return 0

    def _collect_distances(benchmark: BenchmarkExperiment) -> dict[int, list[float]]:
        fid_to_distances: dict[int, list[float]] = {}
        for experiment in benchmark.experiments:
            _, distances = experiment.sample_ela_features_and_distances(n_samples=n_samples)
            fid_to_distances.setdefault(experiment.config.fid, []).extend(distances)
        return fid_to_distances

    all_distances = [_collect_distances(b) for b in benchmark_experiments]

    all_fids: set[int] = set()
    for dist_dict in all_distances:
        all_fids.update(dist_dict.keys())
    shared_fids = sorted(all_fids)

    if not shared_fids:
        raise ValueError("Benchmarks must have at least one function id")

    n_benchmarks = len(benchmark_experiments)
    n_fids = len(shared_fids)

    fig_width = 3.5 + 1.5 * n_benchmarks
    fig_height = max(4, n_fids * 0.35)
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(fig_width, fig_height), sharey=True, sharex=True)

    if n_benchmarks == 1:
        axes = [axes]

    box_height = 0.6
    base_positions = np.arange(n_fids)

    for ax_idx, (ax, label) in enumerate(zip(axes, labels)):
        dist_dict = all_distances[ax_idx]

        for i, fid in enumerate(shared_fids):
            data = dist_dict.get(fid, [0])
            color = GROUP_COLORS[_get_group_index(fid)]

            ax.boxplot(
                [data],
                positions=[i],
                widths=box_height,
                vert=False,
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor="black", linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
                flierprops=dict(
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=3,
                    alpha=0.6,
                ),
            )

        for start, end in BBOB_GROUPS:
            fids_in_group = [f for f in shared_fids if start <= f <= end]
            if fids_in_group:
                first_idx = shared_fids.index(min(fids_in_group))
                separator_y = first_idx - 0.5
                if separator_y > -0.5:
                    ax.axhline(y=separator_y, color="gray", linewidth=1.0, linestyle="-")

        ax.set_yticks(base_positions)
        ax.set_yticklabels(shared_fids)
        ax.set_ylim(-0.5, n_fids - 0.5)
        ax.invert_yaxis()

        ax.grid(True, axis="x", alpha=0.4, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=9)

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[0].set_ylabel("FID", fontsize=10)

    legend_patches = [
        Patch(facecolor=GROUP_COLORS[i], edgecolor="black", label=f"F{s}-F{e}") for i, (s, e) in enumerate(BBOB_GROUPS)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        frameon=True,
        edgecolor="black",
        title="Group",
        title_fontsize=9,
        ncol=5,
    )

    fig.supxlabel("Euclidean Distance", fontsize=10, y=-0.06)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.95, wspace=0.08)

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return all_distances


def barplot_avg_sampled_distance(
    all_distances: list[dict[int, list[float]]],
    labels: list[str],
    file_path: str | None = None,
):
    label_to_avg_distance = {}
    for label, dist_dict in zip(labels, all_distances):
        median_distances = [np.median(distances) for distances in dist_dict.values()]
        avg_distance = np.mean(median_distances)
        label_to_avg_distance[label] = avg_distance

    label_to_avg_distance = dict(sorted(label_to_avg_distance.items(), key=lambda item: item[1]))

    n_bars = len(label_to_avg_distance)
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(n_bars)]

    plt.figure(figsize=(8, 5))
    plt.bar(label_to_avg_distance.keys(), label_to_avg_distance.values(), color=colors)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Average Median Sampled Distance", fontsize=12)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


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


def barplot_function_comparison_faceted(
    benchmark_experiments: list[BenchmarkExperiment],
    labels: list[str],
    file_path: str | None = None,
) -> None:
    BBOB_GROUPS = [
        (1, 5, "FGroup: 1"),
        (6, 9, "FGroup: 2"),
        (10, 14, "FGroup: 3"),
        (15, 19, "FGroup: 4"),
        (20, 24, "FGroup: 5"),
    ]

    METHOD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fid_to_best_distance: dict[int, dict[str, float]] = {}
    for benchmark_experiment, label in zip(benchmark_experiments, labels):
        for experiment in benchmark_experiment.experiments:
            best_distance = experiment.best_function_info.distance_to_target
            fid = experiment.config.fid
            if fid not in fid_to_best_distance:
                fid_to_best_distance[fid] = {}
            fid_to_best_distance[fid][label] = best_distance

    function_ids = sorted(fid_to_best_distance.keys())

    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

    n_methods = len(labels)
    total_bar_width = 0.75
    width = total_bar_width / n_methods

    global_max = 0
    for fid in function_ids:
        for label in labels:
            if label in fid_to_best_distance[fid]:
                global_max = max(global_max, fid_to_best_distance[fid][label])  # type: ignore[assignment]

    for ax_idx, (start, end, title) in enumerate(BBOB_GROUPS):
        ax = axes[ax_idx]
        group_fids = [fid for fid in function_ids if start <= fid <= end]

        if not group_fids:
            ax.set_visible(False)
            continue

        x = np.arange(len(group_fids))

        for i, method in enumerate(labels):
            offset = (i - (n_methods - 1) / 2) * width
            distances = [fid_to_best_distance[fid].get(method, 0) for fid in group_fids]
            color = METHOD_COLORS[i % len(METHOD_COLORS)]
            ax.bar(
                x + offset,
                distances,
                width,
                label=method if ax_idx == 0 else "",
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{fid}" for fid in group_fids], fontsize=8)
        ax.yaxis.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Euclidean Distance to Target ELA", fontsize=10)

    fig.legend(
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, wspace=0.08)

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def heatmap_function_comparison(
    benchmark_experiments: list[BenchmarkExperiment],
    labels: list[str],
    file_path: str | None = None,
    annotate: bool = True,
    cmap: str = "Blues",
) -> None:
    """
    Heatmap visualization for comparing methods across functions.

    Research-paper-ready with professional grayscale/blue colormap.
    Lower values (better) are lighter, higher values (worse) are darker.

    Args:
        cmap: Recommended colormaps for papers:
            - "Blues" (default): Professional, prints well in B&W
            - "Greys": Pure grayscale for B&W printing
            - "YlOrRd": Yellow-Orange-Red, good for color papers
            - "viridis": Perceptually uniform, colorblind-friendly
    """
    BBOB_GROUPS = [(1, 5), (6, 9), (10, 14), (15, 19), (20, 24)]

    fid_to_best_distance: dict[int, dict[str, float]] = {}
    for benchmark_experiment, label in zip(benchmark_experiments, labels):
        for experiment in benchmark_experiment.experiments:
            best_distance = experiment.best_function_info.distance_to_target
            fid = experiment.config.fid
            if fid not in fid_to_best_distance:
                fid_to_best_distance[fid] = {}
            fid_to_best_distance[fid][label] = best_distance

    function_ids = sorted(fid_to_best_distance.keys())

    data = np.zeros((len(labels), len(function_ids)))
    for i, method in enumerate(labels):
        for j, fid in enumerate(function_ids):
            data[i, j] = fid_to_best_distance[fid].get(method, np.nan)

    fig_width = max(10, len(function_ids) * 0.5)
    fig_height = max(3, len(labels) * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    for group_idx, (start, end) in enumerate(BBOB_GROUPS):
        fids_in_group = [i for i, fid in enumerate(function_ids) if start <= fid <= end]
        if fids_in_group and group_idx > 0:
            x_sep = min(fids_in_group) - 0.5
            ax.axvline(x=x_sep, color="black", linewidth=1.2)

    ax.set_xticks(np.arange(len(function_ids)))
    ax.set_xticklabels([f"F{fid}" for fid in function_ids], fontsize=10)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)

    if annotate:
        vmin, vmax = data.min(), data.max()
        threshold = vmin + 0.6 * (vmax - vmin)
        for i in range(len(labels)):
            for j in range(len(function_ids)):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > threshold else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="medium",
                        color=text_color,
                    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Distance to Target ELA", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("BBOB Function", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
