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


def compare_eotf_vs_foga(
    eotf_benchmark: BenchmarkExperiment,
    foga_df: pd.DataFrame,
    dim: int,
    n_samples: int = 50,
    file_path: str | None = None,
) -> None:
    eotf_distances = get_eotf_distances(eotf_benchmark, n_samples)
    foga_distances = get_foga_distances(foga_df, dim=dim)

    all_distances = [eotf_distances, foga_distances]
    labels = [f"EoTF (dim={dim})", f"FOGA (dim={dim})"]
    heatmap_win_percentage_matrix(all_distances, labels, file_path=file_path)


if __name__ == "__main__":
    eotf_results_dim_2 = BenchmarkExperiment.from_dir("./eoh_dim2_2025_06_19")
    eotf_results_dim_3 = BenchmarkExperiment.from_dir("./eoh_dim3_2025_11_30")
    foga_nn_median_distances = pd.read_csv("./data/median_ela_distances_foga_nn.csv")

    compare_eotf_vs_foga(
        eotf_benchmark=eotf_results_dim_2,
        foga_df=foga_nn_median_distances,
        dim=2,
        file_path="./images/eotf_vs_foga_dim2.png",
    )

    compare_eotf_vs_foga(
        eotf_benchmark=eotf_results_dim_3,
        foga_df=foga_nn_median_distances,
        dim=3,
        file_path="./images/eotf_vs_foga_dim3.png",
    )
