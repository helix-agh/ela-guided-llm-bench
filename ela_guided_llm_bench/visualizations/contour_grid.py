import argparse
from pathlib import Path

from ela_guided_llm_bench.experiment import BenchmarkExperiment


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
    parser = argparse.ArgumentParser(description="Generate contour grid plots for best functions")
    parser.add_argument(
        "--dir-path",
        type=str,
        default="./eoh_dim2_2025_12_27",
        help="Directory containing benchmark experiment results",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="images/eoh_bbob_2d_contours.png",
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
        default=list(range(1, 25)),
        help="Function IDs to plot",
    )

    args = parser.parse_args()
    bbob_results = BenchmarkExperiment.from_dir(args.dir_path)
    bbob_results.plot_contour_grid(function_ids=args.function_ids, file_path=args.file_path)
    code_file_path = args.code_file_path
    if code_file_path is None:
        plot_path = Path(args.file_path)
        code_file_path = plot_path.parent / f"{plot_path.stem}_code.md"
    save_best_functions_code(bbob_results, args.function_ids, code_file_path)


if __name__ == "__main__":
    main()
