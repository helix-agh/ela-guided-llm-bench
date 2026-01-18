import argparse
from pathlib import Path

from ela_guided_llm_bench.algorithm_benchmark.algorithm_runner import AlgorithmRunner
from ela_guided_llm_bench.experiment import BenchmarkExperiment


def main():
    parser = argparse.ArgumentParser(description="Run algorithm benchmark on a BenchmarkExperiment")
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to the BenchmarkExperiment directory",
        default="./eoh_dim2_2025_12_27",
    )
    parser.add_argument(
        "target_path", type=str, help="Path where results will be saved", default="./bbob_dim2_algorithm_benchmark"
    )

    args = parser.parse_args()
    target_path = Path(args.target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    benchmark_experiment = BenchmarkExperiment.from_dir(args.experiment_path)

    runner = AlgorithmRunner(dim=benchmark_experiment.config.dim)

    runner.benchmark_problems(
        problems=benchmark_experiment.problems,
        problem_type=args.experiment_path,
    )
    runner.save_results(target_path)


if __name__ == "__main__":
    main()
