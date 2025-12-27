import itertools
import random
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
import pflacco.classical_ela_features as pf
import pflacco.misc_features as mf
from ela_guided_llm_bench.ela import FEATURES
from ioh import ProblemClass, get_problem
from pflacco.sampling import create_initial_sample


def run_experiment(experiment: tuple) -> pd.DataFrame:
    start = time()
    fid, dim, iid = experiment
    results = []
    problem = get_problem(fid, dimension=dim, instance=iid, problem_class=ProblemClass.BBOB)
    for rep in range(100):
        seed = int(fid) * int(iid) * int(dim) * int(rep + 1)
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
        data[["fid", "dim", "iid", "rep"]] = [fid, dim, iid, rep]

        results.append(data)

    df = pd.concat(results).reset_index(drop=True)
    end = time()
    print(f"Finished fid={fid}, dim={dim}, iid={iid} in {end - start:.2f}s")
    return df


if __name__ == "__main__":
    fids = range(1, 25)
    dims = [2, 3, 4, 5]
    iids = [1]

    cart_prod = list(itertools.product(fids, dims, iids))

    with Pool(8) as p:
        results = p.map(run_experiment, cart_prod)

    data = pd.concat(results).reset_index(drop=True)
    data.to_csv("./data/ela_values.csv", index=False)

    cols = [x for x in data.columns if "costs_runtime" not in x][:-4]
    data_min = data.groupby("dim")[cols].min()
    data_min["type"] = "min"
    data_max = data.groupby("dim")[cols].max()
    data_max["type"] = "max"
    pd.concat([data_min, data_max]).to_csv("./data/ela_min_max.csv")

    # Compute mean ELA values:
    data.groupby(["fid", "dim", "iid"])[cols].mean().to_csv("./data/ela_mean_values.csv")
