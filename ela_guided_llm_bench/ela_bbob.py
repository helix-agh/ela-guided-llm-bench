from multiprocessing import Pool

import pandas as pd
from ioh import ProblemClass, get_problem

from .ela import get_ela_features


def process_fid(fid):
    fid_features = []
    for dim in [2, 3]:
        for iid in [2]:
            problem = get_problem(fid, iid, dim, problem_class=ProblemClass.BBOB)
            features = get_ela_features(problem, dim)
            fid_features.append(features | {"fid": fid, "iid": iid, "dim": dim})
    return fid_features


if __name__ == "__main__":
    with Pool() as pool:
        all_features = pool.map(process_fid, range(1, 25))

    bbob_features = [f for fid_features in all_features for f in fid_features]
    pd.DataFrame(bbob_features).to_csv(
        "./data/ela/bbob_ela_features.csv", index=False
    )
