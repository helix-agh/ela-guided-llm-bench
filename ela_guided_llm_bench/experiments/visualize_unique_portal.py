"""
visualize_unique_portal.py

Plots 3-D surface + 2-D contour for the N most unique PORTAL instances
(ranked by L2 distance to the nearest BBOB function in jointly-normalised
8-D ELA feature space).

Requires ./data/ela_space_features.csv (run ela_space.py first).

Usage (from project root):
    python ela_guided_llm_bench/experiments/visualize_unique_portal.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

TOP_N = 8
GRID_RES = 80
BOUNDS = (-5.0, 5.0)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rank_by_bbob_distance(csv_path: Path) -> list[tuple[float, str]]:
    """Return (dist, group_name) sorted descending for PORTAL rows."""
    df = pd.read_csv(csv_path)
    feat = df[FEATURES].values.astype(float)
    fmin = np.nanmin(feat, axis=0)
    fmax = np.nanmax(feat, axis=0)
    rng = np.where(fmax - fmin == 0, 1.0, fmax - fmin)
    norm = np.nan_to_num((feat - fmin) / rng, nan=0.0)

    bbob_norm = norm[df["source"].values == "BBOB"]
    portal_mask = df["source"].values == "PORTAL"
    portal_norm = norm[portal_mask]
    portal_df = df[portal_mask].reset_index(drop=True)

    ranked = []
    for i in range(len(portal_norm)):
        d = np.linalg.norm(bbob_norm - portal_norm[i], axis=1).min()
        ranked.append((d, portal_df.iloc[i]["group"]))
    ranked.sort(reverse=True)
    return ranked


def _eval_surface(fn, res: int = GRID_RES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = BOUNDS
    xs = np.linspace(lo, hi, res)
    ys = np.linspace(lo, hi, res)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.array(
        [fn(pd.Series([x, y])) for x, y in zip(XX.ravel(), YY.ravel())]
    ).reshape(res, res)
    return xs, ys, Z


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_path = Path("./data/ela_space_features.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run ela_space.py first."
        )
    Path("./data").mkdir(parents=True, exist_ok=True)

    ranked = _rank_by_bbob_distance(csv_path)[:TOP_N]
    stem_to_fn = {
        Path(f).stem: fn for fn, f in zip(PORTAL_PROBLEMS, PORTAL_INSTANCE_FILES)
    }

    # ── layout: TOP_N columns, 2 rows per instance (surface + contour) ────────
    ncols = 4
    nrows_per_instance = 2          # row 0: surface, row 1: contour
    nrows = (TOP_N // ncols) * nrows_per_instance

    specs = []
    for block_row in range(TOP_N // ncols):
        specs.append([{"type": "surface"}] * ncols)
        specs.append([{"type": "xy"}] * ncols)

    subplot_titles = []
    for dist, name in ranked:
        stem = name.replace(".json", "")
        label = stem.split("_", 1)[1] if "_" in stem else stem
        subplot_titles.append(f"{label}  (dist {dist:.3f})")
        subplot_titles.append("")          # contour row has no separate title

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    COLORSCALE = "Plasma"

    for idx, (dist, name) in enumerate(ranked):
        block_row = idx // ncols           # 0 or 1
        col = idx % ncols + 1
        surf_row = block_row * nrows_per_instance + 1
        cont_row = surf_row + 1

        stem = name.replace(".json", "")
        fn = stem_to_fn.get(stem)
        if fn is None:
            print(f"  WARNING: {stem} not in loaded problems — skipped")
            continue

        print(f"  Evaluating {stem} ...")
        xs, ys, Z = _eval_surface(fn)

        # 3-D surface
        fig.add_trace(
            go.Surface(
                x=xs, y=ys, z=Z,
                colorscale=COLORSCALE,
                showscale=False,
                name=stem,
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.4f}",
            ),
            row=surf_row, col=col,
        )

        # 2-D filled contour
        fig.add_trace(
            go.Contour(
                x=xs, y=ys, z=Z,
                colorscale=COLORSCALE,
                showscale=False,
                contours_coloring="heatmap",
                line_width=0,
                name=stem,
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.4f}",
            ),
            row=cont_row, col=col,
        )

    fig.update_layout(
        title=dict(
            text=f"Top {TOP_N} most unique PORTAL instances — 3-D surface & contour",
            x=0.5,
        ),
        font=dict(size=11),
        width=1600,
        height=500 * (TOP_N // ncols) * nrows_per_instance,
        margin=dict(t=80, b=40, l=20, r=20),
    )

    # Remove axis labels on surface subplots for cleanliness
    for ann in fig.layout.annotations:
        ann.font.size = 11

    out = "./data/portal_unique_surfaces.html"
    fig.write_html(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
