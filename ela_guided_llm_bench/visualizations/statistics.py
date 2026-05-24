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


def summary_table(
    distances_by_method: dict[str, dict[int, list[float]]],
    families: dict[str, str],
    output_path: str | Path,
    caption: str = (
        "Per-model summary on the 24 two-dimensional BBOB functions. The median "
        "and interquartile range are computed over the per-function median ELA "
        "distances (one observation per function), and the average rank is the "
        "mean Friedman rank across functions. Lower is better on all three "
        "columns; models are grouped by family and ordered best-first within "
        "each group."
    ),
    label: str = "tab:llm_model_comparison",
) -> pd.DataFrame:
    """Family-grouped LaTeX summary table of model performance.

    For each method we report the median, IQR ($q_{0.25}$--$q_{0.75}$) and mean
    Friedman rank of its per-fid median distances -- the same per-function
    observations used by the omnibus tests, so the table, the CD diagram and the
    figures all describe the *same* aggregation level. Computing the IQR over
    these 24 per-fid medians (rather than the pooled raw samples) keeps it tight
    and interpretable: it is the spread of typical performance across functions,
    not the spread of within-function sampling noise. Writes both ``.tex`` and a
    sibling ``.csv``; returns the per-method summary frame.
    """
    df = _per_fid_median_table(distances_by_method)  # rows: fids, cols: methods
    ranks = df.rank(axis=1, method="average").mean(axis=0)  # lower distance => rank 1

    summary = pd.DataFrame(
        {
            "family": [families.get(m, "Other") for m in df.columns],
            "median": [float(df[m].median()) for m in df.columns],
            "q1": [float(df[m].quantile(0.25)) for m in df.columns],
            "q3": [float(df[m].quantile(0.75)) for m in df.columns],
            "avg_rank": [float(ranks[m]) for m in df.columns],
        },
        index=pd.Index(df.columns, name="model"),
    )

    # Family order = order of first appearance among the methods; within a family
    # models are listed best-first by median distance.
    family_order = list(dict.fromkeys(summary["family"]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Median & IQR $[q_{0.25}, q_{0.75}]$ & Avg.\ rank \\",
        r"\midrule",
    ]
    for fam_idx, fam in enumerate(family_order):
        if fam_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{fam}}}}} \\")
        block = summary[summary["family"] == fam].sort_values("median")
        for model, row in block.iterrows():
            lines.append(
                rf"\quad {model} & {row['median']:.3g} & "
                rf"$[{row['q1']:.3g},\, {row['q3']:.3g}]$ & {row['avg_rank']:.1f} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    summary.to_csv(output_path.with_suffix(".csv"))
    print(f"Saved summary table to {output_path} (+ .csv)")
    return summary


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


# ---------------------------------------------------------------------------
# Rigorous head-to-head comparison of two methods across several benchmarks.
#
# Unit of analysis = the function. For each fid we collapse the 100 ELA
# resamples to a single per-fid median (those resamples measure ELA seed-noise
# of *one* generated function, not method variability, so testing on them
# directly is pseudo-replication). Within a suite the two methods are paired by
# fid, so the per-suite test is a paired Wilcoxon signed-rank test, with the
# Vargha-Delaney A12 and win-rate as effect sizes. With only two methods the
# Friedman/Nemenyi machinery used elsewhere degenerates to a sign test, which is
# why this path is separate. The suites are then combined into one directional
# aggregate via Stouffer's weighted-Z method (weights = sqrt(n)), which respects
# the block structure and is immune to per-suite scale differences.
# ---------------------------------------------------------------------------


def _a12_magnitude(a12: float) -> str:
    """Vargha-Delaney magnitude label from the distance of A12 to 0.5.

    Thresholds (0.56 / 0.64 / 0.71) are the standard small/medium/large cut-offs
    expressed as |A12 - 0.5| = 0.06 / 0.14 / 0.21."""
    delta = abs(a12 - 0.5)
    if delta < 0.06:
        return "negligible"
    if delta < 0.14:
        return "small"
    if delta < 0.21:
        return "medium"
    return "large"


def _fmt_p(p: float) -> str:
    """LaTeX-friendly p-value formatting."""
    if p <= 0 or p < 1e-4:
        exp = int(np.floor(np.log10(p))) if p > 0 else -300
        return rf"$<\!10^{{{exp}}}$"
    if p < 1e-3:
        return f"{p:.4f}"
    return f"{p:.3f}"


def paired_suite_comparison(
    distances_a: dict[int, list[float]],
    distances_b: dict[int, list[float]],
    suite: str,
    method_a: str = "EoTF",
    method_b: str = "LLaMEA",
) -> dict[str, Any]:
    """Paired head-to-head of two methods on one benchmark suite.

    Pairs ``method_a`` and ``method_b`` by fid on their per-fid median ELA
    distances (lower is better) and returns win-rate, A12, a paired Wilcoxon
    signed-rank p-value, and a *signed* z (positive when ``method_a`` is better)
    suitable for Stouffer aggregation.
    """
    shared = sorted(set(distances_a) & set(distances_b))
    if not shared:
        raise ValueError(f"No shared fids between {method_a} and {method_b} on suite '{suite}'")
    x = np.array([float(np.median(distances_a[f])) for f in shared])  # method_a
    y = np.array([float(np.median(distances_b[f])) for f in shared])  # method_b
    n = len(shared)

    diff = x - y  # < 0 => method_a has the lower (better) distance
    wins_a = int(np.sum(x < y))
    ties = int(np.sum(x == y))
    win_pct_a = (wins_a + 0.5 * ties) / n
    a12 = _a12(x, y)  # > 0.5 => method_a tends to lower distances => better

    try:
        p = float(stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:  # e.g. all paired differences are zero
        p = 1.0
    p = float(min(max(p, 1e-300), 1.0))

    direction = float(np.sign(-np.median(diff)))  # +1 => method_a better
    signed_z = direction * float(stats.norm.isf(p / 2.0))

    if a12 > 0.5:
        better = method_a
    elif a12 < 0.5:
        better = method_b
    else:
        better = "tie"

    return {
        "suite": suite,
        "method_a": method_a,
        "method_b": method_b,
        "n": n,
        "median_a": float(np.median(x)),
        "median_b": float(np.median(y)),
        f"{method_a}_win_pct": win_pct_a,
        "a12": a12,
        "a12_magnitude": _a12_magnitude(a12),
        "wilcoxon_p": p,
        "signed_z": signed_z,
        "better": better,
    }


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (controls FWER)."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def stouffer_combination(signed_zs: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Stouffer's weighted-Z combination. Returns (Z, two-sided p)."""
    z = np.asarray(signed_zs, dtype=float)
    w = np.asarray(weights, dtype=float)
    combined_z = float(np.sum(w * z) / np.sqrt(np.sum(w**2)))
    p = float(2.0 * stats.norm.sf(abs(combined_z)))
    return combined_z, p


def aggregate_paired_comparison(
    suites: dict[str, tuple[dict[int, list[float]], dict[int, list[float]]]],
    method_a: str = "EoTF",
    method_b: str = "LLaMEA",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-suite paired comparison + Holm correction + Stouffer aggregate.

    ``suites`` maps a benchmark name to a ``(distances_a, distances_b)`` pair,
    each a ``{fid: [distance, ...]}`` dict. Returns the per-suite table (one row
    per suite, with a ``holm_p`` column) and an aggregate summary dict combining
    the suites into a single directional test.
    """
    rows = [paired_suite_comparison(da, db, suite, method_a, method_b) for suite, (da, db) in suites.items()]
    df = pd.DataFrame(rows)
    df["holm_p"] = holm_bonferroni(df["wilcoxon_p"].to_numpy())

    weights = np.sqrt(df["n"].to_numpy(dtype=float))
    combined_z, stouffer_p = stouffer_combination(df["signed_z"].to_numpy(), weights)

    # n-weighted aggregate effect sizes.
    n = df["n"].to_numpy(dtype=float)
    overall_win = float(np.sum(df[f"{method_a}_win_pct"] * n) / np.sum(n))
    overall_a12 = float(np.sum(df["a12"] * n) / np.sum(n))

    # Pooled paired Wilcoxon across every function as a robustness check.
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for _, (da, db) in suites.items():
        shared = sorted(set(da) & set(db))
        pooled_x.extend(float(np.median(da[f])) for f in shared)
        pooled_y.extend(float(np.median(db[f])) for f in shared)
    try:
        pooled_p = float(stats.wilcoxon(pooled_x, pooled_y, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        pooled_p = 1.0

    aggregate = {
        "n_total": int(np.sum(n)),
        "n_suites": len(df),
        f"{method_a}_win_pct": overall_win,
        "a12": overall_a12,
        "a12_magnitude": _a12_magnitude(overall_a12),
        "stouffer_z": combined_z,
        "stouffer_p": stouffer_p,
        "pooled_wilcoxon_p": pooled_p,
        "better": method_a if overall_a12 > 0.5 else method_b,
    }
    return df, aggregate


def write_paired_comparison_table(
    df: pd.DataFrame,
    aggregate: dict[str, Any],
    output_path: str | Path,
    method_a: str = "EoTF",
    method_b: str = "LLaMEA",
    caption: str | None = None,
    label: str = "tab:llamea_comparison",
) -> None:
    """Write the per-suite + aggregate comparison as a LaTeX table (and .csv).

    Bolds the Holm-adjusted p-value of suites where the difference is
    significant at 0.05. The aggregate row reports the Stouffer combined test.
    """
    win_col = f"{method_a}_win_pct"

    def _bold(s: str) -> str:
        # Bolds both text (\bfseries) and math (\boldmath) content.
        return rf"{{\boldmath\bfseries {s}}}"

    if caption is None:
        caption = (
            rf"Head-to-head comparison of {method_a} against {method_b} across six "
            rf"benchmarks. Win\,\% and $A_{{12}}$ are computed from per-function "
            rf"median ELA distances (one observation per function) and favour "
            rf"{method_a} when above $50\%$ / $0.5$ ($A_{{12}}$ magnitude: "
            rf"$0.56/0.64/0.71$ = small/medium/large). $p$ is a paired Wilcoxon "
            rf"signed-rank test per suite, Holm-corrected across the six suites. "
            rf"The aggregate row combines the suites with Stouffer's weighted-$Z$ "
            rf"(weights $\sqrt{{n}}$)."
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        rf"Benchmark & $n$ & {method_a} win\% & $A_{{12}}$ & " rf"Wilcoxon $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        holm = row["holm_p"]
        holm_str = _fmt_p(holm)
        if holm < 0.05:
            holm_str = _bold(holm_str)
        lines.append(
            rf"{row['suite']} & {int(row['n'])} & "
            rf"{100 * row[win_col]:.1f} & {row['a12']:.2f} & "
            rf"{_fmt_p(row['wilcoxon_p'])} & {holm_str} \\"
        )
    lines.append(r"\midrule")
    agg_p = _fmt_p(aggregate["stouffer_p"])
    if aggregate["stouffer_p"] < 0.05:
        agg_p = _bold(agg_p)
    lines.append(
        rf"\textit{{Overall (Stouffer)}} & {aggregate['n_total']} & "
        rf"{100 * aggregate[win_col]:.1f} & {aggregate['a12']:.2f} & "
        rf"-- & {agg_p} \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    csv_df = df.copy()
    csv_df.to_csv(output_path.with_suffix(".csv"), index=False)
    pd.DataFrame([aggregate]).to_csv(output_path.with_name(output_path.stem + "_aggregate.csv"), index=False)
    print(f"Saved comparison table to {output_path} (+ .csv, + _aggregate.csv)")
