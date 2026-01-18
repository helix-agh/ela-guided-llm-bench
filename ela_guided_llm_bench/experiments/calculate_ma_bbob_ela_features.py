import random
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
import pflacco.classical_ela_features as pf
import pflacco.misc_features as mf
from ela_guided_llm_bench.ela import FEATURES
from ela_guided_llm_bench.experiments.affinic import AFFINIC_PROBLEMS
from pflacco.sampling import create_initial_sample


def run_experiment(problem_idx: int) -> pd.DataFrame:
    start = time()
    dim = 2
    results = []
    problem = AFFINIC_PROBLEMS[problem_idx]

    for rep in range(100):
        seed = int(problem_idx + 1) * int(rep + 1)
        np.random.seed(seed)
        random.seed(seed)

        X = create_initial_sample(
            dim,
            sample_coefficient=250,
            sample_type="lhs",
            lower_bound=-5,
            upper_bound=5,
        )
        y = X.apply(lambda x: problem(x), axis=1)
        y = (y - y.min()) / (y.max() - y.min())

        ela_distr = pf.calculate_ela_distribution(X, y)
        ela_meta = pf.calculate_ela_meta(X, y)
        fd = mf.calculate_fitness_distance_correlation(X, y)
        nbc = pf.calculate_nbc(X, y)

        data = pd.DataFrame({**ela_distr, **ela_meta, **fd, **nbc}, index=[0])
        data = data[FEATURES]
        data[["fid", "dim", "rep"]] = [problem_idx + 1, dim, rep]

        results.append(data)

    df = pd.concat(results).reset_index(drop=True)
    end = time()
    print(f"Finished problem_idx={problem_idx + 1} in {end - start:.2f}s")
    return df


if __name__ == "__main__":
    problem_indices = list(range(len(AFFINIC_PROBLEMS)))

    with Pool(8) as p:
        results = p.map(run_experiment, problem_indices)

    data = pd.concat(results).reset_index(drop=True)
    data.to_csv("./data/ma_bbob_ela_values.csv", index=False)

    cols = [x for x in data.columns if "costs_runtime" not in x][:-3]
    data_min = data[cols].min()
    data_min["type"] = "min"
    data_max = data[cols].max()
    data_max["type"] = "max"
    pd.concat([data_min, data_max], axis=1).T.to_csv("./data/ma_bbob_ela_min_max.csv")

    # Compute mean ELA values:
    data.groupby(["fid", "dim"])[cols].mean().to_csv("./data/ma_bbob_ela_mean_values.csv")
