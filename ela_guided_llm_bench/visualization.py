from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def plot_contour(problem: Callable) -> None:
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = problem(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Function Value")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.tight_layout()
    plt.show()
