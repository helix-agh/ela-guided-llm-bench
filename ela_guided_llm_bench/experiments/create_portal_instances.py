"""
create_portal_instances.py

Creates 40 hand-designed PORTAL instances that span the 8-dimensional ELA
feature space as broadly as possible.  Each instance targets a specific
combination of structural properties; together they cover:

  A. Smooth quadratic basins         (high ela_meta.quad)
  B. Curvature variations            (sub/super-quadratic, Form B)
  C. Additive-periodic ruggedness    (regular ripples)
  D. Log-sinusoidal ruggedness       (self-similar, GNBG-style)
  E. Wavelet modulation              (localised oscillations)
  F. Coupling transforms             (tensor interference, radial rings)
  G. Pure multimodality              (multiple smooth basins)
  H. Multimodal + rugged             (basins with ruggedness)
  I. Unique structural combos        (sub-quad+rugged, FormB+tensor, etc.)
  J. Complex / mixed                 (high K, mixed forms, compositions)

Usage (from ela_guided_llm_bench/experiments/):
    python create_portal_instances.py
"""

import contextlib
import io
from pathlib import Path

from ela_guided_llm_bench.experiments.portal.generator import benchmark_generator
from ela_guided_llm_bench.experiments.portal.save_instance import save_instance

INSTANCES_DIR = Path(__file__).parent / "portal" / "instances"


# ── helpers ───────────────────────────────────────────────────────────────────

def _gen(name: str, **kwargs):
    """Generate one instance and save it; suppress the generator's stdout."""
    kwargs.setdefault("Dimension", 2)
    with contextlib.redirect_stdout(io.StringIO()):
        portal = benchmark_generator(**kwargs)
    path = save_instance(portal, name, str(INSTANCES_DIR))
    k = portal["K"]
    bl = portal["baseline"]
    ts = portal["transform_seq"]
    print(f"  {name}  K={k}  baseline={bl}  transforms={ts}")
    return path


# ── instance definitions ──────────────────────────────────────────────────────

def create_all():
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in INSTANCES_DIR.glob("*.json")}
    created = 0

    specs = _define_instances()
    for name, kwargs in specs:
        if name in existing:
            print(f"  SKIP {name}  (already exists)")
            continue
        _gen(name, **kwargs)
        created += 1

    print(f"\nDone – created {created} new instances in {INSTANCES_DIR}")


def _define_instances():
    """
    Returns list of (filename_stem, generator_kwargs) for all 40 instances.
    Seeds are fixed so the files are always identical on re-run.
    """
    D2 = {"Dimension": 2}

    specs = []

    # ── A: Smooth quadratic basins ────────────────────────────────────────────
    # A1 – sphere  (isotropic, no rotation, no transform)
    specs += [("P9_A1_Sphere", {**D2, "Seed": 101,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0, "BetaOffsetMode": 1})]

    # A2 – anisotropic ellipsoid, no rotation
    specs += [("P10_A2_Ellipsoid", {**D2, "Seed": 102,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 4,
        "PMode": 2, "FixedP": 1.0})]

    # A3 – rotated ellipsoid
    specs += [("P11_A3_RotatedEllipsoid", {**D2, "Seed": 103,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 4,
        "PMode": 2, "FixedP": 1.0})]

    # A4 – high-conditioning, rotated
    specs += [("P12_A4_IllConditioned", {**D2, "Seed": 104,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 2,
        "BetaAlpha": 0.1, "BetaBeta": 5.0,
        "PMode": 2, "FixedP": 1.0})]

    # ── B: Curvature variations ───────────────────────────────────────────────
    # B1 – sub-quadratic curvature (p=0.3, grows slowly → very flat sides)
    specs += [("P13_B1_SubQuadratic", {**D2, "Seed": 201,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 0.3})]

    # B2 – near-linear curvature (p=0.5 → |z| linear growth)
    specs += [("P14_B2_LinearBasin", {**D2, "Seed": 202,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 5,
        "PMode": 2, "FixedP": 0.5})]

    # B3 – double log-sinusoidal sequence, separable → high skewness
    specs += [("P15_B3_LogSin_Double", {**D2, "Seed": 5001,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal", "log_sinusoidal"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0})]

    # B4 – Form B (non-separability via exponent), p=0.6
    specs += [("P16_B4_FormB_Coupled", {**D2, "Seed": 204,
        "NumComponents": 1, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "B",
        "RotationMode": 2, "KappaMode": 4,
        "PMode": 2, "FixedP": 0.6})]

    # ── C: Additive-periodic ruggedness ───────────────────────────────────────
    # C1 – mild ripples, separable
    specs += [("P17_C1_MildRipples_Sep", {**D2, "Seed": 301,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0, "TransformParamMode": 4})]

    # C2 – strong ripples, rotated
    specs += [("P18_C2_StrongRipples_Rot", {**D2, "Seed": 302,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 2, "FixedP": 1.0, "TransformParamMode": 1})]

    # C3 – high-frequency ripples, ill-conditioned
    specs += [("P19_C3_HighFreqRipples", {**D2, "Seed": 303,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 2,
        "BetaAlpha": 0.1, "BetaBeta": 5.0,
        "PMode": 2, "FixedP": 1.0})]

    # C4 – ripples + multimodal (K=3)
    specs += [("P20_C4_RippleMultimodal", {**D2, "Seed": 304,
        "NumComponents": 3,
        "TransformMode": 4, "TransformSeq": [["additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 1, "BetaOffsetMode": 1})]

    # ── D: Log-sinusoidal ruggedness ──────────────────────────────────────────
    # D1 – mild log-sin, separable
    specs += [("P21_D1_MildLogSin_Sep", {**D2, "Seed": 401,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0, "TransformParamMode": 4})]

    # D2 – strong log-sin, rotated
    specs += [("P22_D2_StrongLogSin_Rot", {**D2, "Seed": 402,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 2,
        "BetaAlpha": 0.2, "BetaBeta": 0.2,
        "PMode": 2, "FixedP": 1.0, "TransformParamMode": 1})]

    # D3 – log-sin + strong conditioning
    specs += [("P23_D3_LogSin_IllCond", {**D2, "Seed": 403,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 2,
        "BetaAlpha": 0.1, "BetaBeta": 5.0,
        "PMode": 2, "FixedP": 1.0})]

    # D4 – log-sin + multimodal K=3
    specs += [("P24_D4_LogSin_Multimodal", {**D2, "Seed": 404,
        "NumComponents": 3,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 5})]

    # ── E: Wavelet modulation ─────────────────────────────────────────────────
    # E1 – mild wavelet, separable
    specs += [("P25_E1_MildWavelet_Sep", {**D2, "Seed": 501,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["wavelet_mod"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0, "TransformParamMode": 4})]

    # E2 – additive-periodic + FormB, separable → near P17 but FormB coupling
    specs += [("P26_E2_Additive_FormB_Sep", {**D2, "Seed": 5002,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "B",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 0.6})]

    # E3 – wavelet + K=3 multimodal
    specs += [("P27_E3_Wavelet_Multimodal", {**D2, "Seed": 503,
        "NumComponents": 3,
        "TransformMode": 4, "TransformSeq": [["wavelet_mod"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1})]

    # E4 – wavelet, FormB
    specs += [("P28_E4_Wavelet_FormB", {**D2, "Seed": 504,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["wavelet_mod"]],
        "BaselineMode": 3, "BaselineType": "B",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 2, "FixedP": 0.7})]

    # ── F: Coupling transforms ────────────────────────────────────────────────
    # F1 – tensor interference, separable (no rotation)
    specs += [("P29_F1_Tensor_Sep", {**D2, "Seed": 601,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["tensor_interference"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0})]

    # F2 – tensor interference, rotated
    specs += [("P30_F2_Tensor_Rot", {**D2, "Seed": 602,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["tensor_interference"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 2, "FixedP": 1.0})]

    # F3 – radial polytrig (concentric rings)
    specs += [("P31_F3_RadialRings", {**D2, "Seed": 603,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["radial_polytrig"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 5,
        "PMode": 2, "FixedP": 1.0})]

    # F4 – radial + additive composition
    specs += [("P32_F4_Radial_Additive", {**D2, "Seed": 604,
        "NumComponents": 1,
        "TransformMode": 4,
        "TransformSeq": [["radial_polytrig", "additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 2, "FixedP": 1.0})]

    # ── G: Pure multimodal ────────────────────────────────────────────────────
    # G1 – K=8 tensor interference → push nbc_cor more negative
    specs += [("P33_G1_K8_Tensor", {**D2, "Seed": 5003,
        "NumComponents": 8,
        "TransformMode": 4, "TransformSeq": [["tensor_interference"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # G2 – 4 basins, rotated, anisotropic
    specs += [("P34_G2_FourBasins", {**D2, "Seed": 702,
        "NumComponents": 4, "TransformMode": 3,
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # G3 – 7 basins, rotated
    specs += [("P35_G3_SevenBasins", {**D2, "Seed": 703,
        "NumComponents": 7, "TransformMode": 3,
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # G4 – 10 basins (dense, many local optima)
    specs += [("P36_G4_TenBasins", {**D2, "Seed": 704,
        "NumComponents": 10, "TransformMode": 3,
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # ── H: Multimodal + rugged ────────────────────────────────────────────────
    # H1 – K=12 smooth basins, no transform → very low lin_simple, crowded
    specs += [("P37_H1_K12_Smooth", {**D2, "Seed": 5004,
        "NumComponents": 12, "TransformMode": 3,
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # H2 – K=3, log_sinusoidal
    specs += [("P38_H2_K3_LogSin", {**D2, "Seed": 802,
        "NumComponents": 3,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # H3 – K=5, tensor_interference
    specs += [("P39_H3_K5_Tensor", {**D2, "Seed": 803,
        "NumComponents": 5,
        "TransformMode": 4, "TransformSeq": [["tensor_interference"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1, "BetaOffsetMode": 1})]

    # H4 – K=3, wavelet + log_sinusoidal composition
    specs += [("P40_H4_K3_WaveletLogSin", {**D2, "Seed": 804,
        "NumComponents": 3,
        "TransformMode": 4,
        "TransformSeq": [["wavelet_mod", "log_sinusoidal"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 1, "PMode": 1})]

    # ── I: Unique structural combinations ────────────────────────────────────
    # I1 – log-sinusoidal + sub-quadratic curvature, sep → rugged + flat basin
    specs += [("P41_I1_LogSin_SubQuad", {**D2, "Seed": 5005,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 0.3})]

    # I2 – K=3 tensor, FormB, sub-quadratic → coupled multimodal + curvature
    specs += [("P42_I2_Tensor_K3_FormB", {**D2, "Seed": 5006,
        "NumComponents": 3,
        "TransformMode": 4, "TransformSeq": [["tensor_interference"]],
        "BaselineMode": 3, "BaselineType": "B",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 2, "FixedP": 0.5})]

    # I3 – K=4 log-sinusoidal + mild conditioning → rugged multimodal valley
    specs += [("P43_I3_LogSin_K4_IllCond", {**D2, "Seed": 5007,
        "NumComponents": 4,
        "TransformMode": 4, "TransformSeq": [["log_sinusoidal"]],
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 2, "BetaAlpha": 0.1, "BetaBeta": 5.0,
        "PMode": 1, "BetaOffsetMode": 1})]

    # I4 – additive-periodic + sub-quadratic, iso, sep → very flat rippled bowl
    specs += [("P44_I4_Additive_SubQuad", {**D2, "Seed": 5008,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["additive_periodic"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 1, "KappaMode": 5,
        "PMode": 2, "FixedP": 0.3})]

    # ── J: Complex / mixed ────────────────────────────────────────────────────
    # J1 – high multimodality K=10, strong conditioning
    specs += [("P45_J1_K10_IllCond", {**D2, "Seed": 1001,
        "NumComponents": 10, "TransformMode": 3,
        "BaselineMode": 1, "RotationMode": 2,
        "KappaMode": 2, "BetaAlpha": 0.2, "BetaBeta": 3.0,
        "PMode": 1, "BetaOffsetMode": 1})]

    # J2 – Form B, K=4, rotated
    specs += [("P46_J2_FormB_K4", {**D2, "Seed": 1002,
        "NumComponents": 4, "TransformMode": 3,
        "BaselineMode": 3, "BaselineType": "B",
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 2, "FixedP": 0.5})]

    # J3 – tensor interference + severe conditioning (checkerboard in narrow valley)
    specs += [("P47_J3_Tensor_IllCond", {**D2, "Seed": 1003,
        "NumComponents": 1,
        "TransformMode": 4, "TransformSeq": [["tensor_interference"]],
        "BaselineMode": 3, "BaselineType": "A",
        "RotationMode": 2, "KappaMode": 2,
        "BetaAlpha": 0.1, "BetaBeta": 5.0,
        "PMode": 2, "FixedP": 1.0})]

    # J4 – K=5, mixed FormA+B, triple-transform composition
    specs += [("P48_J4_K5_TripleTransform", {**D2, "Seed": 1004,
        "NumComponents": 5,
        "TransformMode": 4,
        "TransformSeq": [["wavelet_mod", "tensor_interference", "additive_periodic"]],
        "BaselineMode": 1,  # random mix A/B
        "RotationMode": 2, "KappaMode": 1,
        "PMode": 1, "BetaOffsetMode": 1})]

    return specs


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Creating PORTAL instances in {INSTANCES_DIR}\n")
    create_all()
