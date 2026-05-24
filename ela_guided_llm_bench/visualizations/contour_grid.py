import argparse
from pathlib import Path

from ela_guided_llm_bench.experiment import BenchmarkExperiment
from ela_guided_llm_bench.visualization import plot_method_contour_grid


def save_best_functions_code(
    benchmark: BenchmarkExperiment,
    function_ids: list[int],
    output_path: Path,
) -> None:
    fid_to_experiment = {exp.config.fid: exp for exp in benchmark.experiments}

    content = _format_as_markdown(benchmark, function_ids, fid_to_experiment)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Best functions code saved to: {output_path}")


def _format_as_markdown(
    benchmark: BenchmarkExperiment,
    function_ids: list[int],
    fid_to_experiment: dict,
) -> str:
    lines = []
    lines.append("# Best Functions Code Review")
    lines.append("")
    lines.append(f"**Method:** {benchmark.method}")
    lines.append(f"**Model:** {benchmark.model}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for fid in function_ids:
        experiment = fid_to_experiment[fid]
        best_function = experiment.best_function_info

        lines.append(f"## FID {fid}")
        lines.append("")
        lines.append(f"- **Distance to target:** `{best_function.distance_to_target:.6f}`")
        if best_function.initial_distance_to_target is not None:
            lines.append(f"- **Initial distance:** `{best_function.initial_distance_to_target:.6f}`")
        lines.append("")
        lines.append("```python")
        lines.append(best_function.source_code.strip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a transposed contour grid comparing target (BBOB), EoTF, LLaMEA and GP landscapes"
    )
    parser.add_argument(
        "--eotf-dir",
        type=str,
        default="./eotf_dim2_2_flash",
        help="Directory containing EoTF benchmark results",
    )
    parser.add_argument(
        "--llamea-dir",
        type=str,
        default="./llamea_dim2",
        help="Directory containing LLaMEA benchmark results",
    )
    parser.add_argument(
        "--gp-dir",
        type=str,
        default="./gp_baseline_dim2",
        help="Directory containing GP baseline benchmark results",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="images/bbob_2d_contours.png",
        help="Output path for contour grid plot",
    )
    parser.add_argument(
        "--code-file-path",
        type=str,
        default=None,
        help="Output path for best functions code file (default: derived from file-path)",
    )
    parser.add_argument(
        "--function-ids",
        type=int,
        nargs="+",
        default=[1, 2, 7, 16, 22],
        help="Function IDs to plot (columns of the grid)",
    )
    parser.add_argument(
        "--n-distance-samples",
        type=int,
        default=100,
        help="Number of ELA resamples used for the median [IQR] distance annotation",
    )

    args = parser.parse_args()
    method_benchmarks = {
        "EoTF": BenchmarkExperiment.from_dir(args.eotf_dir),
        "LLaMEA": BenchmarkExperiment.from_dir(args.llamea_dir),
        "GP": BenchmarkExperiment.from_dir(args.gp_dir),
    }
    plot_method_contour_grid(
        method_benchmarks,
        function_ids=args.function_ids,
        n_distance_samples=args.n_distance_samples,
        file_path=args.file_path,
    )
    code_file_path = args.code_file_path
    if code_file_path is None:
        plot_path = Path(args.file_path)
        code_file_path = plot_path.parent / f"{plot_path.stem}_code.md"
    save_best_functions_code(method_benchmarks["EoTF"], args.function_ids, code_file_path)


if __name__ == "__main__":
    main()
