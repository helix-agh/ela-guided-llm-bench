"""Rigorous head-to-head comparison of EoTF against LLaMEA across six benchmarks.

This addresses the reviewers' request to compare EoTF and LLaMEA with more than a
single win percentage on BBOB. We treat the six settings as separate benchmarks:

    BBOB 2D, BBOB 3D, BBOB 4D, BBOB 5D, MA-BBOB 2D, PORTAL 2D

For every benchmark the two methods are paired by function id on their per-function
median ELA distance (one observation per function -- the 100 ELA resamples are
collapsed to a median first, since they only measure seed-noise of a single
generated function). Each suite gets a paired Wilcoxon signed-rank test plus
Vargha-Delaney A12 and win-rate effect sizes; the suites are Holm-corrected and
combined into one directional aggregate via Stouffer's weighted-Z method.

Sampling reuses the shared parallel + on-disk-cached sampler in ``ela_sampling``
(default 8 workers); the ELA feature cache makes re-runs near-instant.

Output: a LaTeX table (+ csv) at ``tables/llamea_comparison.tex``.

Usage (from project root):
    poetry run python ela_guided_llm_bench/visualizations/llamea_comparison.py
    poetry run python ela_guided_llm_bench/visualizations/llamea_comparison.py --workers 16
"""

import argparse

from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualizations.ela_sampling import DEFAULT_CACHE_DIR, collect_distances_parallel
from ela_guided_llm_bench.visualizations.statistics import aggregate_paired_comparison, write_paired_comparison_table

# (suite label, EoTF dir default, LLaMEA dir default, problem_type for sampling).
# problem_type must match the existing pipeline: BBOB and MA-BBOB are sampled
# with "bbob" normalization (the get_benchmark_distances default), PORTAL with
# "portal".
SUITE_SPECS = [
    ("BBOB 2D", "./eotf_dim2_2_flash", "./llamea_dim2", "bbob"),
    ("BBOB 3D", "./eotf_dim3_2_flash", "./llamea_dim3", "bbob"),
    ("BBOB 4D", "./eotf_dim4_2_flash", "./llamea_dim4", "bbob"),
    ("BBOB 5D", "./eotf_dim5_2_flash", "./llamea_dim5", "bbob"),
    ("MA-BBOB 2D", "./eotf_dim2_2_flash_ma_bbob", "./llamea_dim2_ma_bbob", "bbob"),
    ("PORTAL 2D", "./eotf_dim2_portal", "./llamea_dim2_portal", "portal"),
]


def collect_all_distances(
    n_samples: int,
    workers: int,
    cache_dir: str | None,
    eotf_dirs: dict[str, str],
    llamea_dirs: dict[str, str],
) -> dict[str, tuple[dict[int, list[float]], dict[int, list[float]]]]:
    """Sample EoTF and LLaMEA on every suite in parallel; return per-suite pairs."""
    benchmarks: dict[str, tuple[BenchmarkExperiment, str]] = {}
    for suite, _, _, problem_type in SUITE_SPECS:
        benchmarks[f"{suite}||EoTF"] = (BenchmarkExperiment.from_dir(eotf_dirs[suite]), problem_type)
        benchmarks[f"{suite}||LLaMEA"] = (
            BenchmarkExperiment.from_dir(llamea_dirs[suite]),
            problem_type,
        )

    distances = collect_distances_parallel(benchmarks, n_samples=n_samples, workers=workers, cache_dir=cache_dir)
    return {suite: (distances[f"{suite}||EoTF"], distances[f"{suite}||LLaMEA"]) for suite, *_ in SUITE_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EoTF vs LLaMEA across BBOB (2-5D), MA-BBOB and PORTAL.")
    for suite, eotf_default, llamea_default, _ in SUITE_SPECS:
        flag = suite.lower().replace(" ", "-")
        parser.add_argument(f"--eotf-{flag}", default=eotf_default, dest=f"eotf::{suite}")
        parser.add_argument(f"--llamea-{flag}", default=llamea_default, dest=f"llamea::{suite}")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="Process-pool workers (default 8).")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory for the ELA feature cache (default ./.ela_cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute ELA features from scratch without reading/writing the cache.",
    )
    parser.add_argument("--out", default="./tables/llamea_comparison.tex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args_dict = vars(args)
    eotf_dirs = {suite: args_dict[f"eotf::{suite}"] for suite, *_ in SUITE_SPECS}
    llamea_dirs = {suite: args_dict[f"llamea::{suite}"] for suite, *_ in SUITE_SPECS}
    cache_dir = None if args.no_cache else args.cache_dir

    suites = collect_all_distances(
        n_samples=args.n_samples,
        workers=args.workers,
        cache_dir=cache_dir,
        eotf_dirs=eotf_dirs,
        llamea_dirs=llamea_dirs,
    )

    df, aggregate = aggregate_paired_comparison(suites, method_a="EoTF", method_b="LLaMEA")

    print("\nPer-suite EoTF vs LLaMEA (lower distance is better):")
    print(
        df[
            [
                "suite",
                "n",
                "EoTF_win_pct",
                "a12",
                "a12_magnitude",
                "wilcoxon_p",
                "holm_p",
                "better",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nAggregate (Stouffer): Z={aggregate['stouffer_z']:.3f}, "
        f"p={aggregate['stouffer_p']:.3g}; "
        f"weighted win%={100 * aggregate['EoTF_win_pct']:.1f}, "
        f"A12={aggregate['a12']:.3f} ({aggregate['a12_magnitude']}); "
        f"pooled Wilcoxon p={aggregate['pooled_wilcoxon_p']:.3g}; "
        f"better overall: {aggregate['better']}"
    )

    write_paired_comparison_table(df, aggregate, args.out)


if __name__ == "__main__":
    main()
