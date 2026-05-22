"""
visualize_unique_portal.py

Renders a paper-ready PNG with 2-D contour plots of all 8 PORTAL instances
in ela_guided_llm_bench/experiments/portal/instances/.

Usage (from project root):
    python ela_guided_llm_bench/experiments/visualize_unique_portal.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ela_guided_llm_bench.experiments.portal.problems import PORTAL_INSTANCE_FILES, PORTAL_PROBLEMS, _portal_id

NROWS, NCOLS = 4, 5
GRID_RES = 200
BOUNDS = (-5.0, 5.0)
CMAP = "viridis"
# Render only the novel 20-instance batch (IDs 300-399). Set to None to render all.
ID_FILTER_MIN: int | None = 300
ID_FILTER_MAX: int | None = 399
OUTPUT_PATH = Path("./data/portal_surfaces_novel.png")


def evaluate_grid(fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = BOUNDS
    xs = np.linspace(lo, hi, GRID_RES)
    ys = np.linspace(lo, hi, GRID_RES)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.array([fn(pd.Series([x, y])) for x, y in zip(XX.ravel(), YY.ravel())]).reshape(GRID_RES, GRID_RES)
    return XX, YY, Z


def main() -> None:
    pairs = list(zip(PORTAL_PROBLEMS, PORTAL_INSTANCE_FILES))
    if ID_FILTER_MIN is not None:
        pairs = [(fn, f) for fn, f in pairs if _portal_id(f) >= ID_FILTER_MIN]
    if ID_FILTER_MAX is not None:
        pairs = [(fn, f) for fn, f in pairs if _portal_id(f) <= ID_FILTER_MAX]

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(4 * NCOLS, 4 * NROWS))
    axes = axes.ravel()

    for idx, (fn, filename) in enumerate(pairs):
        stem = Path(filename).stem
        print(f"  Evaluating {stem} ...")
        XX, YY, Z = evaluate_grid(fn)

        ax = axes[idx]
        cs = ax.contourf(XX, YY, Z, levels=20, cmap=CMAP)
        label = stem.split("_", 1)[1] if "_" in stem else stem
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(pairs) :]:
        ax.set_visible(False)

    fig.tight_layout()

    out = OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
