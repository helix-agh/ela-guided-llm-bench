import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, plot_stats
from scipy.stats import kendalltau


def read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_median = df.groupby(["budget", "problem_name", "dim", "algorithm_name"]).median(numeric_only=True).reset_index()
    return df_median.pivot(
        index=["budget", "problem_name", "dim"],
        columns="algorithm_name",
        values="fitness_value",
    )


def mean_ranks(df: pd.DataFrame) -> pd.Series:
    return df.rank(axis=1, method="average").mean(axis=0).sort_index()


def kendall_tau(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[float, float]:
    common = df_a.columns.intersection(df_b.columns)
    ranks_a = mean_ranks(df_a[common])
    ranks_b = mean_ranks(df_b[common])
    print("Ranks A:\n", ranks_a)
    print("Ranks B:\n", ranks_b)
    result = kendalltau(ranks_a.values, ranks_b.values)
    return float(result.correlation), float(result.pvalue)


if __name__ == "__main__":
    np.random.seed(42)
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    df_llm_2d = read_df("./eotf_dim2_2_flash_algorithm_benchmark/results.csv")
    df_llm_3d = read_df("./eotf_dim3_2_flash_algorithm_benchmark/results.csv")
    df_bbob_2d = read_df("./bbob_dim2_algorithm_benchmark/results.csv")
    df_bbob_3d = read_df("./bbob_dim3_algorithm_benchmark/results.csv")

    tau_2d, p_2d = kendall_tau(df_bbob_2d, df_llm_2d)
    tau_3d, p_3d = kendall_tau(df_bbob_3d, df_llm_3d)
    print(f"Kendall's tau (2D, BBOB vs EoTF): tau={tau_2d:.4f}, p={p_2d:.4f}")
    print(f"Kendall's tau (3D, BBOB vs EoTF): tau={tau_3d:.4f}, p={p_3d:.4f}")

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
