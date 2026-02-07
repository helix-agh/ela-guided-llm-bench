import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats


def read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_median = df.groupby(["budget", "problem_name", "dim", "algorithm_name"]).median(numeric_only=True).reset_index()
    return df_median.pivot(
        index=["budget", "problem_name", "dim"],
        columns="algorithm_name",
        values="fitness_value",
    )


if __name__ == "__main__":
    np.random.seed(42)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    df_llm_2d = read_df("./eoh_dim2_2025_12_27_algorithm_benchmark/results.csv")
    df_llm_3d = read_df("./eoh_dim3_2025_12_27_algorithm_benchmark/results.csv")
    df_bbob_2d = read_df("./bbob_dim2_algorithm_benchmark_new/results.csv")
    df_bbob_3d = read_df("./bbob_dim3_algorithm_benchmark/results.csv")

    plot_stats(autorank(df_bbob_2d, alpha=0.05, order="ascending", verbose=True), ax=ax[0, 0])
    plot_stats(autorank(df_llm_2d, alpha=0.05, order="ascending", verbose=True), ax=ax[0, 1])
    plot_stats(autorank(df_bbob_3d, alpha=0.05, order="ascending", verbose=True), ax=ax[1, 0])
    plot_stats(autorank(df_llm_3d, alpha=0.05, order="ascending", verbose=True), ax=ax[1, 1])
    ax[0, 0].set_title("a) 2D BBOB", y=-0.3)
    ax[0, 1].set_title("b) 2D EoTF", y=-0.3)
    ax[1, 0].set_title("c) 3D BBOB", y=-0.3)
    ax[1, 1].set_title("d) 3D EoTF", y=-0.3)
    plt.rcParams.update({"font.size": 5})
    fig.tight_layout()
    plt.savefig("images/algorithm_benchmark_comparison.png", dpi=300)
