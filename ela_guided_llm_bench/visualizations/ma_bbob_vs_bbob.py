import argparse

from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import (
    compare_sampled_distance_boxplots,
    ecdf_median_comparison,
    heatmap_win_percentage_matrix,
    histogram_aggregated_distances,
)


def main():
    parser = argparse.ArgumentParser(description="Compare MA-BBOB and BBOB benchmark results")
    parser.add_argument(
        "--ma-dir",
        type=str,
        default="./eotf_dim2_2_flash_ma_bbob",
        help="Directory containing MA-BBOB results",
    )
    parser.add_argument(
        "--bbob-dir",
        type=str,
        default="./eotf_dim2_2_flash",
        help="Directory containing BBOB results",
    )
    parser.add_argument(
        "--image-suffix",
        type=str,
        default="dim2",
        help="Suffix for image file names",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for distance comparison",
    )
    args = parser.parse_args()

    ma_bbob_results = BenchmarkExperiment.from_dir(args.ma_dir)
    bbob_results = BenchmarkExperiment.from_dir(args.bbob_dir)

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
        n_samples=args.n_samples,
        file_path=f"./images/ma_bbob_sample_boxplots_{args.image_suffix}.png",
    )

    heatmap_win_percentage_matrix(
        all_distances,
        labels=labels,
        file_path=f"./images/ma_bbob_win_percentage_matrix_{args.image_suffix}.png",
    )

    histogram_aggregated_distances(
        all_distances,
        labels=labels,
        file_path=f"./images/ma_bbob_distance_histogram_{args.image_suffix}.png",
    )

    ks_result = ecdf_median_comparison(
        all_distances,
        labels=labels,
        file_path=f"./images/ma_bbob_ecdf_median_{args.image_suffix}.png",
    )
    if ks_result is not None:
        print(f"KS statistic: {ks_result.statistic:.4f}, p-value: {ks_result.pvalue:.4f}")


if __name__ == "__main__":
    main()
