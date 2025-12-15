import argparse
from pathlib import Path

from ela_guided_llm_bench.algorithm_benchmark.algorithm_runner import AlgorithmRunner
from ioh import ProblemClass, get_problem


def main():
    parser = argparse.ArgumentParser(description="Run algorithm benchmark on a BenchmarkExperiment")
    parser.add_argument(
        "target_path",
        type=str,
        help="Path where results will be saved",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Problem dimension (default: 2)",
    )

    args = parser.parse_args()
    target_path = Path(args.target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    problems = [(str(fid), get_problem(fid, 1, args.dim, problem_class=ProblemClass.BBOB)) for fid in range(1, 25)]
    runner = AlgorithmRunner(dim=args.dim, max_workers=2)

    runner.benchmark_problems(
        problems=problems,
        problem_type=f"BBOB_{args.dim}D",
    )
    runner.save_results(target_path)


if __name__ == "__main__":
    main()
