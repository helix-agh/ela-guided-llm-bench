from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import (
    barplot_sampled_distances_faceted,
    compare_sampled_distance_boxplots,
    ecdf_median_comparison,
    heatmap_win_percentage_matrix,
)

if __name__ == "__main__":
    bbob_results_gemini_2 = BenchmarkExperiment.from_dir("./eotf_dim2_2_flash")
    bbob_results_gemini_2_5 = BenchmarkExperiment.from_dir("./eotf_dim2_2_5_flash")
    bbob_results_gemini_3 = BenchmarkExperiment.from_dir("./eotf_dim2_3_flash")

    labels = [
        "Gemini 2.0 Flash",
        "Gemini 2.5 Flash",
        "Gemini 3.0 Flash",
    ]
    experiments = [
        bbob_results_gemini_2,
        bbob_results_gemini_2_5,
        bbob_results_gemini_3,
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

    ks_result = ecdf_median_comparison(
        all_distances,
        labels=labels,
        file_path="./images/llm_benchmark_ecdf_median.png",
    )
