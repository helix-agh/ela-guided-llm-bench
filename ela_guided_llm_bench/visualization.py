from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .ela import FEATURES


def compare_contours(
    problem1: Callable,
    problem2: Callable,
    ela_features1: dict[str, float] = None,
    ela_features2: dict[str, float] = None,
    bounds: tuple[float, float] = (-5, 5),
    resolution: int = 100,
    save_path: str | None = None,
    title: str | None = None,
) -> None:
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            Z1[i, j] = problem1(np.array([X[i, j], Y[i, j]]))
            Z2[i, j] = problem2(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    contour1 = ax1.contourf(X, Y, Z1, levels=20, cmap="viridis")
    ax1.set_xlabel("$x_1$", fontsize=12)
    ax1.set_ylabel("$x_2$", fontsize=12)
    plt.colorbar(contour1, ax=ax1)

    if ela_features1 is not None and ela_features2 is not None:
        ax2 = fig.add_subplot(1, 3, 2)

        features = FEATURES

        values1 = [ela_features1[f] for f in features]
        values2 = [ela_features2[f] for f in features]

        x_pos = np.arange(len(features))

        ax2.plot(x_pos, values1, "x-", color="black", linewidth=1.5, label="Generated")
        ax2.plot(x_pos, values2, "x-", color="gray", alpha=0.7, linewidth=1.5, label="Target")

        if title:
            ax2.set_title(f"Function: {title}")
        else:
            ax2.set_title("ELA Feature Comparison")

        ax2.set_xticks(x_pos)
        shortened_features = [f.split(".")[-1] for f in features]
        ax2.set_xticklabels(shortened_features, rotation=90)

        ax2.grid(True, linestyle="--", alpha=0.7)

        ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    contour2 = ax3.contourf(X, Y, Z2, levels=20, cmap="viridis")
    ax3.set_xlabel("$x_1$", fontsize=12)
    ax3.set_ylabel("$x_2$", fontsize=12)
    plt.colorbar(contour2, ax=ax3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_target_values(
    values: list[float],
    title: str = "Target Values",
    xlabel: str = "Iteration",
    ylabel: str = "Value",
    save_path: str | None = None,
) -> None:
    if not values:
        raise ValueError("The values list cannot be empty")

    indices = list(range(len(values)))

    global_min = [values[0]]
    for i in range(1, len(values)):
        global_min.append(min(global_min[-1], values[i]))

    plt.figure(figsize=(10, 6))

    plt.plot(indices, values, "b-o", label="Current Value", alpha=0.7)

    plt.plot(indices, global_min, "r-^", label="Global Minimum")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()
