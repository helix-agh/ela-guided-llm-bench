from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import compare_sampled_distance_boxplots, heatmap_win_percentage_matrix

if __name__ == "__main__":
    ma_bbob_results = BenchmarkExperiment.from_dir("./eoh_dim2_2025_11_17")
    bbob_results = BenchmarkExperiment.from_dir("./eoh_dim2_2025_06_19")

    labels = [
        "BBOB",
        "MA-BBOB",
    ]
    experiments = [
        bbob_results,
        ma_bbob_results,
    ]

    all_distances = compare_sampled_distance_boxplots(
        experiments,
        labels=labels,
        n_samples=100,
        file_path="images/ma_bbob_sample_boxplots.png",
    )

    heatmap_win_percentage_matrix(
        all_distances,
        labels=labels,
        file_path="images/ma_bbob_win_percentage_matrix.png",
    )
