import os

from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import (
    barplot_sampled_distances_faceted,
    compare_sampled_distance_boxplots,
    heatmap_win_percentage_matrix,
)
from ela_guided_llm_bench.visualizations.statistics import run_stats_and_plots

if __name__ == "__main__":
    for stale in (
        "./tables/llm_benchmark_pairwise_stats.csv",
        "./tables/llm_benchmark_omnibus_tests.csv",
    ):
        if os.path.exists(stale):
            os.remove(stale)

    bbob_results_gemini_2 = BenchmarkExperiment.from_dir("./eotf_dim2_2_flash")
    bbob_results_gemini_2_5 = BenchmarkExperiment.from_dir("./eotf_dim2_2_5_flash")
    bbob_results_gemini_3 = BenchmarkExperiment.from_dir("./eotf_dim2_3_flash")
    bbob_results_gpt_5_nano = BenchmarkExperiment.from_dir("./eotf_dim2_gpt_5_nano")
    bbob_results_gpt_5_4_nano = BenchmarkExperiment.from_dir("./eotf_dim2_gpt_5_4_nano")
    bbob_results_gpt_120b = BenchmarkExperiment.from_dir("./eotf_dim2_gpt_oss_120b")
    bbob_results_gemma_4 = BenchmarkExperiment.from_dir("./eotf_dim2_gemma_4")

    labels = [
        "Gemini 2.0 Flash",
        "Gemini 2.5 Flash",
        "Gemini 3.0 Flash",
        "Gemma 4 31B",
        "GPT-5 Nano",
        "GPT-5.4 Nano",
        "GPT-OSS 120B",
    ]
    experiments = [
        bbob_results_gemini_2,
        bbob_results_gemini_2_5,
        bbob_results_gemini_3,
        bbob_results_gemma_4,
        bbob_results_gpt_5_nano,
        bbob_results_gpt_5_4_nano,
        bbob_results_gpt_120b,
    ]

    all_distances = compare_sampled_distance_boxplots(
        experiments,
        labels=labels,
        n_samples=100,
        file_path="images/llm_benchmark_sample_boxplots.png",
    )

    barplot_sampled_distances_faceted(
        all_distances,
        labels=labels,
        file_path="images/llm_benchmark_sampled_distances_faceted.png",
    )

    heatmap_win_percentage_matrix(
        all_distances,
        labels=labels,
        file_path="images/llm_benchmark_win_percentage_matrix.png",
    )

    run_stats_and_plots(
        dict(zip(labels, all_distances)),
        dim=2,
        file_prefix="llm_benchmark",
        dim_in_filename=False,
    )
