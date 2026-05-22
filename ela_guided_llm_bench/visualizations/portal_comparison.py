"""Win matrix + contour grid analysis for the PORTAL benchmark runs.

Mirrors the structure of `method_comparison.py` and `contour_grid.py`, but:
- Computes ELA distances using the PORTAL normalization (not BBOB).
- Uses PORTAL_PROBLEMS[fid-1] as the target surface in contour plots
  instead of the BBOB ``get_problem`` lookup.

Usage (from project root):
    poetry run python ela_guided_llm_bench/visualizations/portal_comparison.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ela_guided_llm_bench.ela import FEATURES, get_distance, get_ela_features
from ela_guided_llm_bench.experiment import BenchmarkExperiment, Experiment
from ela_guided_llm_bench.experiments.portal.problems import PORTAL_INSTANCE_FILES, PORTAL_PROBLEMS
from ela_guided_llm_bench.visualization import barplot_sampled_distances_faceted, heatmap_win_percentage_matrix


def sample_portal_distances(experiment: Experiment, n_samples: int = 50) -> list[float]:
    """Recompute ELA distances using the PORTAL normalization."""
    distances: list[float] = []
    best = experiment.best_function_info
    function = best.function_with_params if best.function_with_params is not None else best.function
    for seed in range(n_samples):
        features = get_ela_features(
            function,
            experiment.config.dim,
            random_seed=seed,
            problem_type="portal",
        )
        distances.append(get_distance(features, experiment.target_ela_features))
    return distances


def collect_distances(benchmark: BenchmarkExperiment, n_samples: int = 50) -> dict[int, list[float]]:
    fid_to_distances: dict[int, list[float]] = {}
    for experiment in benchmark.experiments:
        fid_to_distances.setdefault(experiment.config.fid, []).extend(
            sample_portal_distances(experiment, n_samples=n_samples)
        )
    return fid_to_distances


def plot_portal_contour_grid(
    benchmark: BenchmarkExperiment,
    function_ids: list[int],
    bounds: tuple[float, float] = (-5.0, 5.0),
    resolution: int = 100,
    file_path: str | None = None,
) -> None:
    fid_to_experiment = {exp.config.fid: exp for exp in benchmark.experiments}
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
        best = experiment.best_function_info
        target_problem = PORTAL_PROBLEMS[fid - 1]
        instance_label = Path(PORTAL_INSTANCE_FILES[fid - 1]).stem

        Z_generated = _evaluate_on_grid(best.function, X, Y, experiment.config.dim)
        Z_target = _evaluate_on_grid(target_problem, X, Y, experiment.config.dim)

        ax_generated, ax_features, ax_target = axes[row_idx]

        cg = ax_generated.contourf(X, Y, Z_generated, levels=20, cmap="viridis")
        ax_generated.set_xlabel("$x_1$")
        ax_generated.set_ylabel("$x_2$")
        ax_generated.set_title(f"FID {fid} ({instance_label}) — Generated")
        plt.colorbar(cg, ax=ax_generated)

        feature_positions = np.arange(len(FEATURES))
        generated_values = [best.ela_features[f] for f in FEATURES]
        target_values = [experiment.target_ela_features[f] for f in FEATURES]
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
        ax_features.set_xticklabels([f.split(".")[-1] for f in FEATURES], rotation=90)
        ax_features.set_title("ELA Feature Comparison")
        ax_features.grid(True, linestyle="--", alpha=0.7)
        ax_features.legend()

        ct = ax_target.contourf(X, Y, Z_target, levels=20, cmap="viridis")
        ax_target.set_xlabel("$x_1$")
        ax_target.set_ylabel("$x_2$")
        ax_target.set_title(f"FID {fid} ({instance_label}) — Target")
        plt.colorbar(ct, ax=ax_target)

    plt.tight_layout()
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved contour grid to {file_path}")
    else:
        plt.show()
    plt.close(fig)


def _evaluate_on_grid(problem, X: np.ndarray, Y: np.ndarray, dim: int) -> np.ndarray:
    Z = np.zeros_like(X)
    if dim != 2:
        base = np.zeros(dim, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = base.copy()
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = problem(point)
    else:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = problem(np.array([X[i, j], Y[i, j]]))
    return Z


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse PORTAL benchmark results")
    parser.add_argument("--eotf-dir", default="./eotf_dim2_2026_05_22")
    parser.add_argument("--llamea-dir", default="./llamea_dim2_2026_05_22")
    parser.add_argument("--zero-shot-dir", default="./zero_shot_dim2_2026_05_22")
    parser.add_argument("--gp-dir", default="./gp_baseline_dim2_2026_05_22")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--out-dir", default="./images")
    parser.add_argument(
        "--contour-method",
        choices=["eotf", "llamea", "zero_shot", "gp", "all"],
        default="all",
        help="Which method's best functions to render in the contour grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = {
        "EoTF": BenchmarkExperiment.from_dir(args.eotf_dir),
        "LLaMEA": BenchmarkExperiment.from_dir(args.llamea_dir),
        "Zero Shot": BenchmarkExperiment.from_dir(args.zero_shot_dir),
        "GP": BenchmarkExperiment.from_dir(args.gp_dir),
    }

    labels = list(benchmarks.keys())
    all_distances = [collect_distances(b, n_samples=args.n_samples) for b in benchmarks.values()]

    coverage_per_method = {lbl: sorted(d.keys()) for lbl, d in zip(labels, all_distances)}
    expected_fids = sorted(coverage_per_method[labels[0]])
    for lbl, fids in coverage_per_method.items():
        if fids != expected_fids:
            raise ValueError(
                f"Method '{lbl}' has FID coverage {fids}, " f"expected {expected_fids} (matching '{labels[0]}')."
            )
    print(f"FIDs used in win matrix: {expected_fids}")

    win_matrix_path = out_dir / "portal_win_matrix.png"
    heatmap_win_percentage_matrix(
        all_distances,
        labels=labels,
        file_path=str(win_matrix_path),
    )
    print(f"Saved win matrix to {win_matrix_path}")

    barplot_path = out_dir / "portal_sampled_distances_faceted.png"
    barplot_sampled_distances_faceted(
        all_distances,
        labels=labels,
        file_path=str(barplot_path),
    )
    print(f"Saved faceted barplot to {barplot_path}")

    method_dir_map = {
        "eotf": ("EoTF", benchmarks["EoTF"]),
        "llamea": ("LLaMEA", benchmarks["LLaMEA"]),
        "zero_shot": ("Zero Shot", benchmarks["Zero Shot"]),
        "gp": ("GP", benchmarks["GP"]),
    }
    methods_to_plot = list(method_dir_map.keys()) if args.contour_method == "all" else [args.contour_method]
    for key in methods_to_plot:
        label, benchmark = method_dir_map[key]
        fids = sorted(exp.config.fid for exp in benchmark.experiments)
        contour_path = out_dir / f"portal_contour_grid_{key}.png"
        plot_portal_contour_grid(
            benchmark,
            function_ids=fids,
            file_path=str(contour_path),
        )


if __name__ == "__main__":
    main()
