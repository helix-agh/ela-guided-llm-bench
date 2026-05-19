import os

from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import (
    barplot_sampled_distances_faceted,
    compare_sampled_distance_boxplots,
    heatmap_win_percentage_matrix,
)

if __name__ == "__main__":
    for stale in (
        "./tables/llm_benchmark_pairwise_stats.csv",
        "./tables/llm_benchmark_omnibus_tests.csv",
    ):
        if os.path.exists(stale):
            os.remove(stale)

    full_prompt_results = BenchmarkExperiment.from_dir("./eotf_dim2_2_flash")
    no_desc_results = BenchmarkExperiment.from_dir("./eotf_dim2_no_ela_desc")

    labels = [
        "Full Prompt",
        "No Description",
    ]
    experiments = [
        full_prompt_results,
        no_desc_results,
    ]

    all_distances = compare_sampled_distance_boxplots(
        experiments,
        labels=labels,
        n_samples=100,
        file_path="images/prompt_ablation_sample_boxplots.png",
    )

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
