"""
ela_space.py – Compute ELA features for PORTAL instances and BBOB functions,
then compare them in 2-D UMAP space.

Run from the ela_guided_llm_bench/experiments/ directory:
    python ela_space.py

Outputs (relative to the working directory):
    ./data/ela_space_features.csv   – raw ELA feature table
    ./data/ela_space.html           – interactive UMAP plot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import umap
import plotly.express as px
from ioh import ProblemClass, get_problem
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_nbc,
)
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.sampling import create_initial_sample

from ela_guided_llm_bench.experiments.portal.problems import (
    PORTAL_INSTANCE_FILES,
    PORTAL_PROBLEMS,
)

FEATURES = [
    "ela_distr.skewness",
    "fitness_distance.fitness_std",
    "nbc.nn_nb.sd_ratio",
    "nbc.nb_fitness.cor",
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
]

BBOB_NAMES = {
    1: "Sphere", 2: "Ellipsoid Sep.", 3: "Rastrigin Sep.", 4: "Büche-Rastrigin",
    5: "Linear Slope", 6: "Attractive Sector", 7: "Step Ellipsoid",
    8: "Rosenbrock", 9: "Rosenbrock Rot.", 10: "Ellipsoid",
    11: "Discus", 12: "Bent Cigar", 13: "Sharp Ridge",
    14: "Sum Diff. Powers", 15: "Rastrigin", 16: "Weierstrass",
    17: "Schaffers F7", 18: "Schaffers F7 Ill-cond.", 19: "Griewank-Rosenbrock",
    20: "Schwefel", 21: "Gallagher 101", 22: "Gallagher 21",
    23: "Katsuura", 24: "Lunacek bi-Rastrigin",
}

DIM = 2
BBOB_IIDS = [1, 2, 3]
SAMPLE_COEFFICIENT = 250


def compute_raw_ela_features(problem_fn, dim: int, seed: int = 42) -> dict:
    """Compute raw (unnormalized) ELA features for one callable."""
    X = create_initial_sample(
        dim,
        lower_bound=-5,
        upper_bound=5,
        sample_type="lhs",
        sample_coefficient=SAMPLE_COEFFICIENT,
        seed=seed,
    )
    y = X.apply(lambda x: problem_fn(x), axis=1)
    y = (y - y.min()) / (y.max() - y.min())

    ela_meta = calculate_ela_meta(X, y)
    ela_distr = calculate_ela_distribution(X, y)
    nbc = calculate_nbc(X, y)
    fitness_distance = calculate_fitness_distance_correlation(X, y)

    all_features = {**ela_meta, **ela_distr, **nbc, **fitness_distance}
    return {feat: all_features[feat] for feat in FEATURES}


def main():
    Path("./data").mkdir(parents=True, exist_ok=True)
    records = []

    print("Computing ELA features for BBOB functions (dim=2)...")
    for fid in range(1, 25):
        for iid in BBOB_IIDS:
            problem = get_problem(fid, iid, DIM, problem_class=ProblemClass.BBOB)
            features = compute_raw_ela_features(problem, DIM, seed=iid * 100 + fid)
            records.append({
                "source": "BBOB",
                "group": f"f{fid}: {BBOB_NAMES[fid]}",
                "label": f"BBOB f{fid} i{iid}: {BBOB_NAMES[fid]}",
                "fid": fid,
                **features,
            })
        print(f"  fid={fid} ({BBOB_NAMES[fid]}) done")

    print("\nComputing ELA features for PORTAL instances (dim=2)...")
    for problem_fn, filename in zip(PORTAL_PROBLEMS, PORTAL_INSTANCE_FILES):
        name = filename.replace(".json", "")
        features = compute_raw_ela_features(problem_fn, DIM, seed=42)
        records.append({
            "source": "PORTAL",
            "group": name,
            "label": name,
            "fid": None,
            **features,
        })
        print(f"  {name} done")

    df = pd.DataFrame(records)
    df.to_csv("./data/ela_space_features.csv", index=False)
    print(f"\nSaved ./data/ela_space_features.csv  ({len(df)} rows)")

    feature_matrix = df[FEATURES].values.astype(float)
    feat_min = np.nanmin(feature_matrix, axis=0)
    feat_max = np.nanmax(feature_matrix, axis=0)
    feat_range = np.where(feat_max - feat_min == 0, 1.0, feat_max - feat_min)
    normalized = (feature_matrix - feat_min) / feat_range
    normalized = np.nan_to_num(normalized, nan=0.0)

    print("Running UMAP...")
    n_neighbors = min(15, len(df) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(normalized)

    df["umap_1"] = embedding[:, 0]
    df["umap_2"] = embedding[:, 1]

    fig = px.scatter(
        df,
        x="umap_1",
        y="umap_2",
        color="source",
        symbol="source",
        hover_name="label",
        hover_data={feat: True for feat in FEATURES},
        title="ELA Feature Space: PORTAL vs BBOB (UMAP, dim=2)",
        color_discrete_map={"BBOB": "#4C72B0", "PORTAL": "#DD4444"},
        labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2", "source": "Benchmark"},
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig.update_layout(
        legend_title_text="Benchmark",
        font=dict(size=13),
        width=950,
        height=680,
    )
    fig.write_html("./data/ela_space.html")
    print("Saved ./data/ela_space.html")

    try:
        fig.write_image("./data/ela_space.png", width=950, height=680)
        print("Saved ./data/ela_space.png")
    except Exception:
        print("Note: kaleido not installed – PNG skipped (pip install kaleido for static export)")


if __name__ == "__main__":
    main()
