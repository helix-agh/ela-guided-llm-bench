import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import heatmap_win_percentage_matrix


def get_eotf_distances(benchmark: BenchmarkExperiment, n_samples: int = 50) -> dict[int, list[float]]:
    fid_to_distances: dict[int, list[float]] = {}
    for experiment in benchmark.experiments:
        _, distances = experiment.sample_ela_features_and_distances(n_samples=n_samples)
        fid_to_distances.setdefault(experiment.config.fid, []).extend(distances)
    return fid_to_distances


def get_foga_distances(foga_df: pd.DataFrame, dim: int) -> dict[int, list[float]]:
    filtered = foga_df[foga_df["dim"] == dim]
    fid_to_distances: dict[int, list[float]] = {}
    for _, row in filtered.iterrows():
        fid = int(row["fid"])
        distance = float(row["distance"])
        fid_to_distances[fid] = [distance]
    return fid_to_distances


def compute_avg_median_distance(fid_to_distances: dict[int, list[float]]) -> float:
    medians = [np.median(distances) for distances in fid_to_distances.values()]
    return float(np.mean(medians))


def plot_dimension_comparison(
    eotf_distances_by_dim: dict[int, dict[int, list[float]]],
    foga_distances_by_dim: dict[int, dict[int, list[float]]],
    file_path: str | None = None,
) -> None:
    eotf_dims = sorted(eotf_distances_by_dim.keys())
    eotf_values = [compute_avg_median_distance(eotf_distances_by_dim[dim]) for dim in eotf_dims]

    foga_dims = sorted(foga_distances_by_dim.keys())
    foga_values = [compute_avg_median_distance(foga_distances_by_dim[dim]) for dim in foga_dims]

    plt.figure(figsize=(8, 5))
    plt.plot(eotf_dims, eotf_values, "o-", label="EoTF", markersize=8, linewidth=2)
    plt.plot(foga_dims, foga_values, "s--", label="FOGA", markersize=8, linewidth=2)

    plt.xlabel("Dimension")
    plt.ylabel("Avg. Median ELA Distance")
    plt.xticks(eotf_dims)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_eotf_vs_foga(
    eotf_benchmark: BenchmarkExperiment,
    foga_df: pd.DataFrame,
    dim: int,
    n_samples: int = 100,
    file_path: str | None = None,
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    eotf_distances = get_eotf_distances(eotf_benchmark, n_samples)
    foga_distances = get_foga_distances(foga_df, dim=dim)

    all_distances = [eotf_distances, foga_distances]
    labels = [f"EoTF (dim={dim})", f"FOGA (dim={dim})"]
    heatmap_win_percentage_matrix(all_distances, labels, file_path=file_path)
    return eotf_distances, foga_distances


if __name__ == "__main__":
    n_samples = 100
    eotf_results_dim_2 = BenchmarkExperiment.from_dir("./eoh_dim2_2025_12_27")
    eotf_results_dim_3 = BenchmarkExperiment.from_dir("./eoh_dim3_2025_12_27")
    eotf_results_dim_4 = BenchmarkExperiment.from_dir("./eoh_dim4_2025_12_27")
    eotf_results_dim_5 = BenchmarkExperiment.from_dir("./eoh_dim5_2025_12_27")
    foga_nn_median_distances = pd.read_csv("./data/median_ela_distances_foga_nn.csv")

    eotf_distances_dim_2, foga_distances_dim_2 = compare_eotf_vs_foga(
        eotf_benchmark=eotf_results_dim_2,
        foga_df=foga_nn_median_distances,
        dim=2,
        file_path="./images/eotf_vs_foga_dim2.png",
        n_samples=n_samples,
    )

    eotf_distances_dim_3, foga_distances_dim_3 = compare_eotf_vs_foga(
        eotf_benchmark=eotf_results_dim_3,
        foga_df=foga_nn_median_distances,
        dim=3,
        file_path="./images/eotf_vs_foga_dim3.png",
        n_samples=n_samples,
    )

    eotf_distances_dim_4 = get_eotf_distances(
        benchmark=eotf_results_dim_4,
        n_samples=n_samples,
    )
    eotf_distances_dim_5 = get_eotf_distances(
        benchmark=eotf_results_dim_5,
        n_samples=n_samples,
    )

    plot_dimension_comparison(
        eotf_distances_by_dim={
            2: eotf_distances_dim_2,
            3: eotf_distances_dim_3,
            4: eotf_distances_dim_4,
            5: eotf_distances_dim_5,
        },
        foga_distances_by_dim={
            2: foga_distances_dim_2,
            3: foga_distances_dim_3,
        },
        file_path="./images/dimension_comparison.png",
    )
