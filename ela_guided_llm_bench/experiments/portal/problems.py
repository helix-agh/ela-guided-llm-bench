"""PORTAL benchmark problems, packaged as callables on [-5, 5]^d.

Mirrors the structure of `experiments/affinic.py`: pre-builds a list of
target functions that can be plugged into the benchmark harness as
`target_problem = PORTAL_PROBLEMS[fid - 1]`.

Each instance JSON in `instances/` was generated on PORTAL's native
[-100, 100]^d domain. We expose them on [-5, 5]^d via the linear-scaling
adapter so they share the convention used elsewhere in the suite (BBOB,
MA-BBOB).
"""

from pathlib import Path
from typing import Callable, List

from .load_instance import load_instance
from .scaled_fitness import make_scaled_evaluator

USER_MIN = -5.0
USER_MAX = 5.0

INSTANCES_DIR = Path(__file__).parent / "instances"

_ORIGINAL_FILES = [
    "P1_DeceptiveWaveletMaze.json",
    "P2_TensorCoupledTwinBowl.json",
    "P3_IllConditionedCuspRidge.json",
    "P4_StackedMultiTransformChaos.json",
    "P5_FiveRoundBasins.json",
    "P6_FiveRotatedEllipsoids.json",
    "P7_FourCoupledFormB.json",
    "P8_FiveMildPeriodic.json",
]

# All JSON files in instances/ sorted by name – original 8 first, then any
# generated instances added later (e.g. by generate_diverse_portal_instances.py).
_generated = sorted(
    p.name for p in INSTANCES_DIR.glob("*.json")
    if p.name not in _ORIGINAL_FILES
)
PORTAL_INSTANCE_FILES: List[str] = _ORIGINAL_FILES + _generated


def _build_problem(filename: str) -> Callable:
    portal = load_instance(str(INSTANCES_DIR / filename))
    return make_scaled_evaluator(portal, user_min=USER_MIN, user_max=USER_MAX)


PORTAL_PROBLEMS: List[Callable] = [_build_problem(f) for f in PORTAL_INSTANCE_FILES]
