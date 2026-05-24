"""Convergence curves: best-so-far ELA distance vs. number of candidate evaluations.

Averages over all functions in a benchmark directory and shades the volatility
(mean +/- std across functions). The x-axis is the cumulative number of candidate
functions evaluated -- NOT the iteration index -- so methods with different
population sizes / iteration counts (e.g. EoTF at ~10/iter vs GP at ~50/gen) are
compared on the same budget footing.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from ela_guided_llm_bench.experiment import BenchmarkExperiment


def best_so_far_curve(experiment) -> np.ndarray:
    """Best (minimum) distance-to-target seen up to each candidate evaluation.

    Candidates are flattened in chronological (iteration) order. A failed
    candidate (NaN distance) still consumes one unit of budget but never lowers
    the running minimum.
    """
    distances = np.array(
        [info.distance_to_target for gen in experiment.function_infos for info in gen],
        dtype=float,
    )
    # Map failed candidates (NaN) to +inf rather than dropping them: a failure
    # still consumes one unit of budget, so keeping it preserves the curve's
    # length. This is what lets methods stay comparable on a shared evaluation
    # axis -- a NaN/missing distance changes the curve's *value*, never its
    # *length*.
    distances = np.where(np.isnan(distances), np.inf, distances)
    return np.minimum.accumulate(distances)


def aggregate_curves(
    benchmark: BenchmarkExperiment, length: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of best-so-far curves across all functions in the benchmark.

    Curves are truncated to ``length`` evaluations so they align on a common
    evaluation grid; when ``length`` is None it defaults to the shortest
    function's budget within this benchmark. Pass an explicit ``length`` to
    force alignment across *different* benchmarks (see ``plot_convergence``).
    Returns (evaluations, mean, std).
    """
    curves = [best_so_far_curve(exp) for exp in benchmark.experiments]
    if length is None:
        length = min(len(c) for c in curves)
    matrix = np.vstack([c[:length] for c in curves])
    evaluations = np.arange(1, length + 1)
    return evaluations, matrix.mean(axis=0), matrix.std(axis=0)


def plot_convergence(
    benchmarks: dict[str, str],
    file_path: str | None = None,
    extended_budget_labels: set[str] | None = None,
    log_scale: bool = True,
) -> None:
    # Methods listed in `extended_budget_labels` keep their full recorded budget;
    # every other method is truncated to a shared length (see below). Defaults to
    # {"GP"} because GP is the baseline we deliberately give a *larger* budget.
    if extended_budget_labels is None:
        extended_budget_labels = {"GP"}

    loaded = {label: BenchmarkExperiment.from_dir(dir_name) for label, dir_name in benchmarks.items()}

    # Align the fixed-budget LLM methods (EoTF and LLaMEA) to the same final point
    # on the x-axis by truncating their curves to the shortest shared budget.
    # Without this, runs that recorded a few more/fewer candidates
    # would end at different x positions and be
    # visually incomparable. GP is intentionally excluded and plotted to its full
    # length: the whole point of the comparison is to check whether GP, given a
    # *larger* evaluation budget, can reach a quality similar to the fixed-budget
    # LLM methods -- so capping it to their budget would defeat the experiment.
    # Because failed candidates are kept as +inf in best_so_far_curve (they still
    # cost budget), this shared length is driven purely by how far each run
    # actually got -- never by NaNs/missing values.
    common_length = min(
        (
            len(best_so_far_curve(exp))
            for label, benchmark in loaded.items()
            if label not in extended_budget_labels
            for exp in benchmark.experiments
        ),
        default=None,
    )

    plt.figure(figsize=(8, 5))
    min_mean = np.inf
    max_eval = 0
    # (last_eval, final_mean, color) for each fixed-budget method, so we can hold
    # its final value as a dotted line out to the largest budget on the plot.
    fixed_budget_tails = []
    for label, benchmark in loaded.items():
        length = None if label in extended_budget_labels else common_length
        evaluations, mean, std = aggregate_curves(benchmark, length=length)
        max_eval = max(max_eval, int(evaluations[-1]))
        positive = mean[mean > 0]  # for the log-axis lower limit (0 is invalid)
        if positive.size:
            min_mean = min(min_mean, float(positive.min()))
        line = plt.plot(
            evaluations,
            mean,
            linewidth=2,
            label=label,
        )[0]
        plt.fill_between(
            evaluations,
            # Distance is non-negative; on a log axis the lower band must also be
            # strictly positive (a 0 floor maps to -inf and the band disappears).
            np.clip(mean - std, 1e-3 if log_scale else 0, None),
            mean + std,
            color=line.get_color(),
            alpha=0.2,
        )
        if label not in extended_budget_labels:
            fixed_budget_tails.append((int(evaluations[-1]), float(mean[-1]), line.get_color()))

    # Extend the fixed-budget methods (EoTF, LLaMEA) as dotted horizontal lines
    # from where their budget ends out to the largest budget on the plot (GP's).
    # The dotted segment is NOT new data: it simply holds each method's final
    # best-so-far value so it can be read off against GP at the same x -- making
    # "does GP's larger budget beat the LLM methods' final result?" obvious. Same
    # colour as the solid curve, no confidence band (nothing was evaluated there).
    # Endpoints come from the data (common budget vs. global max), never hardcoded.
    for last_eval, final_mean, color in fixed_budget_tails:
        if last_eval < max_eval:
            plt.plot(
                [last_eval, max_eval],
                [final_mean, final_mean],
                color=color,
                linestyle=":",
                linewidth=2,
            )

    # Optional log scale on the distance axis: best-so-far ELA distance spans an
    # order of magnitude (GP starts high, all methods converge toward a small
    # value), so on a linear axis the early values stretch the range and crush the
    # converged region into a thin strip. A log axis applies the same transform to
    # every method -- fairer than clipping ylim, which would hide GP's genuine
    # starting disadvantage -- and reveals the region where the methods separate.
    if log_scale:
        plt.yscale("log")
        if np.isfinite(min_mean):
            plt.ylim(bottom=0.8 * min_mean)

    plt.xlabel("Candidate functions evaluated (budget)")
    ylabel = "Best-so-far ELA distance to target"
    plt.ylabel(f"{ylabel} (log scale)" if log_scale else ylabel)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"Saved convergence plot to {file_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves for EoTF, GP and LLaMEA")
    parser.add_argument("--eotf-dir", type=str, default="eotf_dim2_2_flash")
    parser.add_argument("--gp-dir", type=str, default="gp_baseline_dim2_long")
    parser.add_argument("--llamea-dir", type=str, default="llamea_dim2")
    parser.add_argument("--file-path", type=str, default="images/convergence_dim2.png")
    parser.add_argument(
        "--no-log-scale",
        dest="log_scale",
        action="store_false",
        help="Use a linear y-axis instead of the default log scale",
    )
    args = parser.parse_args()

    plot_convergence(
        {"EoTF": args.eotf_dir, "GP": args.gp_dir, "LLaMEA": args.llamea_dir},
        file_path=args.file_path,
        log_scale=args.log_scale,
    )


if __name__ == "__main__":
    main()
