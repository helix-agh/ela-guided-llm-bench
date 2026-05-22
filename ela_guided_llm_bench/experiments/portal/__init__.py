"""PORTAL benchmark integration.

Ported from https://github.com/EvoMindLab/PORTAL (GPL-3.0).
Reference: http://arxiv.org/abs/2512.00288
Authors: Delaram Yazdani, Danial Yazdani, Mai Peng.
"""

from .fitness import fitness
from .load_instance import load_instance
from .problems import PORTAL_INSTANCE_FILES, PORTAL_PROBLEMS
from .save_instance import save_instance
from .scaled_fitness import make_scaled_evaluator, scaled_portal_view

__all__ = [
    "PORTAL_INSTANCE_FILES",
    "PORTAL_PROBLEMS",
    "fitness",
    "load_instance",
    "make_scaled_evaluator",
    "save_instance",
    "scaled_portal_view",
]
