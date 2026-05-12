from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats
from ela_guided_llm_bench.visualization import barplot_sampled_distances_faceted, heatmap_win_probability_matrix
from scipy import stats


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    """Vargha-Delaney A12: P(a < b) + 0.5 * P(a == b). Lower is better, so
    win_prob > 0.5 means `a` tends to produce smaller distances than `b`."""
    less = float(np.sum(a[:, None] < b[None, :]))
    equal = float(np.sum(a[:, None] == b[None, :]))
    return (less + 0.5 * equal) / (a.size * b.size)


def _per_fid_median_table(
    distances_by_method: dict[str, dict[int, list[float]]],
) -> pd.DataFrame:
    """Build a (fid x method) table of per-fid median distances. Rows are
    intersection of fids available across all methods."""
    fid_sets = [set(d.keys()) for d in distances_by_method.values()]
    shared_fids = sorted(set.intersection(*fid_sets))
    data = {
        method: [float(np.median(samples[fid])) for fid in shared_fids]
        for method, samples in distances_by_method.items()
    }
    return pd.DataFrame(data, index=shared_fids)


def pairwise_stats(
    distances_by_method: dict[str, dict[int, list[float]]],
    dim: int,
) -> pd.DataFrame:
    """Mann-Whitney U + win probability (A12) for every unordered method pair
    on every shared fid. Lower distance is better, so win_prob > 0.5 means
    method_a is better than method_b on that fid."""
    methods = list(distances_by_method.keys())
    fid_sets = [set(d.keys()) for d in distances_by_method.values()]
    shared_fids = sorted(set.intersection(*fid_sets))

    rows: list[dict[str, Any]] = []
    for fid in shared_fids:
        for method_a, method_b in combinations(methods, 2):
            samples_a = np.asarray(distances_by_method[method_a][fid], dtype=float)
            samples_b = np.asarray(distances_by_method[method_b][fid], dtype=float)
            mw = stats.mannwhitneyu(samples_a, samples_b, alternative="two-sided")
            rows.append(
                {
                    "dim": dim,
                    "fid": fid,
                    "method_a": method_a,
                    "method_b": method_b,
                    "median_a": float(np.median(samples_a)),
                    "median_b": float(np.median(samples_b)),
                    "p_value": float(mw.pvalue),
                    "win_prob": _a12(samples_a, samples_b),
                }
            )
    return pd.DataFrame(rows)


def average_win_probability_matrix(
    distances_by_method: dict[str, dict[int, list[float]]],
) -> tuple[np.ndarray, list[str]]:
    """Mean A12 across shared fids for every (method_i, method_j) pair. Cell
    (i, j) = average probability that method_i produces a smaller distance
    than method_j."""
    methods = list(distances_by_method.keys())
    fid_sets = [set(d.keys()) for d in distances_by_method.values()]
    shared_fids = sorted(set.intersection(*fid_sets))

    n = len(methods)
    matrix = np.full((n, n), np.nan)
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i == j:
                continue
            values = [
                _a12(
                    np.asarray(distances_by_method[method_i][fid], dtype=float),
                    np.asarray(distances_by_method[method_j][fid], dtype=float),
                )
                for fid in shared_fids
            ]
            matrix[i, j] = float(np.mean(values))
    return matrix, methods


def friedman_with_cd(
    distances_by_method: dict[str, dict[int, list[float]]],
    output_path: str | Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Friedman chi-squared via scipy + Nemenyi critical-difference diagram via
    autorank (forced to non-parametric so the test is consistent across calls).
    Observations are per-fid medians, one row per fid, one column per method."""
    df = _per_fid_median_table(distances_by_method)
    columns = [df[col].to_numpy() for col in df.columns]
    friedman = stats.friedmanchisquare(*columns)

    result = autorank(df, alpha=alpha, order="ascending", verbose=False, force_mode="nonparametric")

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_stats(result, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    avg_ranks = df.rank(axis=1, method="average").mean(axis=0)
    return {
        "p_value": float(friedman.pvalue),
        "statistic": float(friedman.statistic),
        "cd": float(result.cd) if result.cd is not None else float("nan"),
        "avg_ranks": avg_ranks.to_dict(),
        "n_observations": len(df),
    }


def one_way_anova(
    distances_by_method: dict[str, dict[int, list[float]]],
) -> dict[str, float]:
    """One-way ANOVA on per-fid medians (one observation per fid per method)."""
    df = _per_fid_median_table(distances_by_method)
    columns = [df[col].to_numpy() for col in df.columns]
    result = stats.f_oneway(*columns)
    return {
        "F": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_observations": len(df),
    }


def omnibus_summary(
    distances_by_method: dict[str, dict[int, list[float]]],
    dim: int,
    cd_diagram_path: str | Path,
) -> dict[str, Any]:
    """Run Friedman (with CD diagram) and one-way ANOVA. Return a single row
    summary suitable for appending to a results CSV."""
    friedman = friedman_with_cd(distances_by_method, cd_diagram_path)
    anova = one_way_anova(distances_by_method)
    return {
        "dim": dim,
        "n_functions": friedman["n_observations"],
        "n_methods": len(distances_by_method),
        "friedman_statistic": friedman["statistic"],
        "friedman_p": friedman["p_value"],
        "friedman_cd": friedman["cd"],
        "anova_F": anova["F"],
        "anova_p": anova["p_value"],
    }


def append_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Append rows to CSV, creating it with header if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def run_stats_and_plots(
    distances_by_method: dict[str, dict[int, list[float]]],
    dim: int,
    file_prefix: str = "method",
    images_dir: str = "./images",
    tables_dir: str = "./tables",
    dim_in_filename: bool = True,
) -> None:
    """Run the full stats + plots pipeline for one (dim, set-of-methods) combo.

    Produces sampled-distance barplot, A12 win-probability heatmap, per-fid
    pairwise stats CSV, and (when >=3 methods) Friedman + Nemenyi CD diagram +
    one-way ANOVA. Output filenames are prefixed by `file_prefix`. Pass
    `dim_in_filename=False` to omit the `_dim{N}` suffix when comparing a
    single dim.
    """
    labels = list(distances_by_method.keys())
    distances = list(distances_by_method.values())
    suffix = f"_dim{dim}" if dim_in_filename else ""

    barplot_sampled_distances_faceted(
        distances,
        labels=labels,
        file_path=f"{images_dir}/{file_prefix}_sampled_barplot{suffix}.png",
    )
    heatmap_win_probability_matrix(
        distances,
        labels=labels,
        file_path=f"{images_dir}/{file_prefix}_win_probability{suffix}.png",
    )

    pairwise_df = pairwise_stats(distances_by_method, dim=dim)
    append_csv(pairwise_df, f"{tables_dir}/{file_prefix}_pairwise_stats.csv")

    if len(labels) >= 3:
        summary = omnibus_summary(
            distances_by_method,
            dim=dim,
            cd_diagram_path=f"{images_dir}/{file_prefix}_cd_diagram{suffix}.png",
        )
        append_csv(
            pd.DataFrame([summary]),
            f"{tables_dir}/{file_prefix}_omnibus_tests.csv",
        )
        print(
            f"[{file_prefix} dim={dim}] Friedman p={summary['friedman_p']:.4g}, "
            f"ANOVA p={summary['anova_p']:.4g}, n_functions={summary['n_functions']}"
        )
    else:
        print(
            f"[{file_prefix} dim={dim}] omnibus skipped (only {len(labels)} methods); "
            f"see pairwise CSV for per-fid Mann-Whitney + win probability"
        )
