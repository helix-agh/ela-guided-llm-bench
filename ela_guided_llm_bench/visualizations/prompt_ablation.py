"""Prompt ablation: does the full prompt beat the prompt without ELA descriptions?

Compares the default EoTF prompt ("Full Prompt") against a variant with the ELA
feature descriptions removed ("No Description") on the 24 BBOB functions in 3D.
The two variants are paired by function id on their per-function median ELA
distance (the 100 ELA resamples are collapsed to a median first, since they only
measure seed-noise of a single generated function), and compared with a paired
Wilcoxon signed-rank test plus Vargha-Delaney A12 and win-rate effect sizes --
the same machinery used for the EoTF-vs-LLaMEA comparison.

Sampling reuses the shared parallel + on-disk-cached sampler; the Full Prompt
directory is also sampled by ``llamea_comparison.py`` (as BBOB 3D EoTF), so its
features are served from the shared cache.

Usage (from project root):
    poetry run python ela_guided_llm_bench/visualizations/prompt_ablation.py
"""

import argparse

import pandas as pd
from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import barplot_sampled_distances_faceted, heatmap_win_percentage_matrix
from ela_guided_llm_bench.visualizations.ela_sampling import DEFAULT_CACHE_DIR, collect_distances_parallel
from ela_guided_llm_bench.visualizations.statistics import paired_suite_comparison

FULL = "Full Prompt"
NO_DESC = "No Description"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt ablation: full vs no-description prompt.")
    parser.add_argument("--full-dir", default="./eotf_dim3_2_flash")
    parser.add_argument("--no-desc-dir", default="./eotf_dim3_no_ela_desc")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="./tables/prompt_ablation_stats.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = None if args.no_cache else args.cache_dir

    benchmarks = {
        FULL: (BenchmarkExperiment.from_dir(args.full_dir), "bbob"),
        NO_DESC: (BenchmarkExperiment.from_dir(args.no_desc_dir), "bbob"),
    }
    distances = collect_distances_parallel(
        benchmarks,
        n_samples=args.n_samples,
        workers=args.workers,
        cache_dir=cache_dir,
    )

    result = paired_suite_comparison(
        distances[FULL],
        distances[NO_DESC],
        suite="BBOB 3D",
        method_a=FULL,
        method_b=NO_DESC,
    )

    print("\nPrompt ablation -- Full Prompt vs No Description (lower distance is better):")
    print(f"  functions (n)         : {result['n']}")
    print(f"  median dist (Full)    : {result['median_a']:.4f}")
    print(f"  median dist (No Desc) : {result['median_b']:.4f}")
    print(f"  Full Prompt win %     : {100 * result[f'{FULL}_win_pct']:.1f}")
    print(f"  A12 (Full vs No Desc) : {result['a12']:.3f} ({result['a12_magnitude']})")
    print(f"  Wilcoxon signed-rank p: {result['wilcoxon_p']:.4g}")
    print(f"  better                : {result['better']}")

    pd.DataFrame([result]).to_csv(args.out, index=False)
    print(f"\nSaved stats to {args.out}")

    all_distances = [distances[FULL], distances[NO_DESC]]
    labels = [FULL, NO_DESC]
    barplot_sampled_distances_faceted(
        all_distances,
        labels=labels,
        file_path="images/prompt_ablation_sampled_distances_faceted.png",
    )
    heatmap_win_percentage_matrix(
        all_distances,
        labels=labels,
        file_path="images/prompt_ablation_win_percentage_matrix.png",
    )


if __name__ == "__main__":
    main()
