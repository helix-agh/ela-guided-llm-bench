"""
Linear-scaling adapter for PORTAL benchmark instances.

A PORTAL instance is generated on a fixed native domain of
[bounds_min, bounds_max] = [-100, 100]. Many standard benchmark
functions (Sphere, Rastrigin, etc.) are conventionally evaluated on
[-5, 5] or [-5.12, 5.12]. This module provides a small adapter that
linearly maps a user-side input domain to the native PORTAL domain
*before* calling fitness(), so a PORTAL instance can be plugged into
an optimizer that searches any rectangular box [user_min, user_max]^d
without re-generating the instance.

The instance itself is unchanged — only the input is rescaled. The
landscape *character* (multimodality, conditioning, basin structure,
seed-determined geometry) is preserved exactly; only the coordinates
of the search space change.

Typical usage:

    from Utils.scaled_fitness import make_scaled_evaluator
    f = make_scaled_evaluator(portal, user_min=-5.0, user_max=5.0)
    f(np.array([0.3, -1.2]))   # evaluates as if at native (6, -24)
"""

from typing import Any, Callable, Dict, Union

import numpy as np

try:
    from .fitness import fitness
except ImportError:
    from Utils.fitness import fitness  # type: ignore[no-redef]


Number = Union[float, int]


def linear_scale(
    x: np.ndarray, user_min: Number, user_max: Number, native_min: Number, native_max: Number
) -> np.ndarray:
    """Affine map [user_min, user_max] -> [native_min, native_max].

    Works elementwise so x may be a scalar, 1-D vector, or array.
    """
    if user_max == user_min:
        raise ValueError("user_max must differ from user_min")
    scale = (native_max - native_min) / (user_max - user_min)
    return native_min + (np.asarray(x) - user_min) * scale


def make_scaled_evaluator(
    portal: Dict[str, Any], user_min: Number = -5.0, user_max: Number = 5.0
) -> Callable[[np.ndarray], float]:
    """Return a fitness function that accepts inputs in [user_min, user_max]^d.

    The returned function rescales its argument into the PORTAL native
    domain [portal['bounds_min'], portal['bounds_max']] and delegates
    to the canonical fitness().
    """
    native_min = portal["bounds_min"]
    native_max = portal["bounds_max"]

    def f(x: np.ndarray) -> float:
        x = np.atleast_1d(np.asarray(x, dtype=float)).flatten()
        x_native = linear_scale(x, user_min, user_max, native_min, native_max)
        return fitness(x_native, portal)

    f.user_min = float(user_min)  # type: ignore[attr-defined]
    f.user_max = float(user_max)  # type: ignore[attr-defined]
    f.native_min = float(native_min)  # type: ignore[attr-defined]
    f.native_max = float(native_max)  # type: ignore[attr-defined]
    return f


def scaled_portal_view(portal: Dict[str, Any], user_min: Number = -5.0, user_max: Number = 5.0) -> Dict[str, Any]:
    """Return a shallow-copy view of `portal` with bounds rewritten to
    [user_min, user_max] and component centers mapped to user
    coordinates.

    The view is intended for *plotting and reporting only*: tools like
    plot_landscape() read bounds_min / bounds_max and P['c'] to build
    a grid and draw markers, and they should display user coordinates
    when the user has chosen a custom domain. The returned view shares
    the same `P` dictionary as the original except that `P['c']` is a
    rescaled copy. Do NOT call fitness() directly on this view — it is
    not a real PORTAL instance; use make_scaled_evaluator() instead.
    """
    view = dict(portal)
    view["bounds_min"] = float(user_min)
    view["bounds_max"] = float(user_max)
    P = dict(portal["P"])
    P["c"] = linear_scale(
        portal["P"]["c"],
        portal["bounds_min"],
        portal["bounds_max"],
        user_min,
        user_max,
    )
    view["P"] = P
    view["_scaled_view"] = True
    return view


def user_to_native(
    x_user: np.ndarray, portal: Dict[str, Any], user_min: Number = -5.0, user_max: Number = 5.0
) -> np.ndarray:
    """Convenience: map a user-domain point to native coordinates."""
    return linear_scale(x_user, user_min, user_max, portal["bounds_min"], portal["bounds_max"])


def native_to_user(
    x_native: np.ndarray, portal: Dict[str, Any], user_min: Number = -5.0, user_max: Number = 5.0
) -> np.ndarray:
    """Convenience: map a native-domain point to user coordinates."""
    return linear_scale(x_native, portal["bounds_min"], portal["bounds_max"], user_min, user_max)
