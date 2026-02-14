import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from time import time

import pandas as pd
from ela_guided_llm_bench.ela import ProblemType, get_target_ela_features
from ela_guided_llm_bench.experiment_config import ExperimentConfig
from ela_guided_llm_bench.gp_baseline.gp_generator import GPBaseline


@dataclass
class TimingResult:
    fid: int
    time_seconds: float
    success: bool
    error: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run GP baseline experiment with specified parameters")
    parser.add_argument("--start-fid", type=int, default=1, help="Starting function ID (default: 1)")
    parser.add_argument("--end-fid", type=int, default=24, help="Ending function ID (default: 24)")
    parser.add_argument("--dim", type=int, default=2, help="Problem dimension (default: 2)")
    parser.add_argument("--iid", type=int, default=1, help="Instance ID (default: 1)")
    parser.add_argument(
        "--problem-class",
        choices=["bbob", "ma-bbob"],
        default="bbob",
        help="Problem class (default: bbob)",
    )
    parser.add_argument("--population", type=int, default=50, help="GP population size (default: 50)")
    parser.add_argument("--ngen", type=int, default=5, help="Number of GP generations (default: 5)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    return parser.parse_args()


def run_single_fid(
    fid: int, iid: int, dim: int, problem_class: ProblemType, population: int, ngen: int, seed: int
) -> TimingResult:
    start_time = time()
    try:
        target_ela_features = get_target_ela_features(fid, iid, dim, problem_type=problem_class)
        config = ExperimentConfig(
            model="gp",
            fid=fid,
            iid=iid,
            dim=dim,
            method="gp_baseline",
        )
        os.makedirs(config.dir_name, exist_ok=True)

        gp_baseline = GPBaseline(
            target_ela_features=target_ela_features,
            dim=dim,
            experiment_config=config,
            population=population,
            ngen=ngen,
            seed=seed,
            problem_type=problem_class,
        )
        experiment = gp_baseline.run()
        experiment.save_to_dir()

        elapsed = time() - start_time
        print(f"FID {fid} completed in {elapsed:.1f}s")
        return TimingResult(fid=fid, time_seconds=elapsed, success=True)
    except Exception as e:
        elapsed = time() - start_time
        print(f"Error running GP baseline for function {fid}: {e}")
        return TimingResult(fid=fid, time_seconds=elapsed, success=False, error=str(e))


def main():
    args = parse_args()
    start = time()

    fids = list(range(args.start_fid, args.end_fid + 1))

    if args.workers == 1:
        timing_results = [
            run_single_fid(fid, args.iid, args.dim, args.problem_class, args.population, args.ngen, args.seed)
            for fid in fids
        ]
    else:
        timing_results: list[TimingResult] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_single_fid, fid, args.iid, args.dim, args.problem_class, args.population, args.ngen, args.seed
                ): fid
                for fid in fids
            }
            for future in as_completed(futures):
                timing_results.append(future.result())
        timing_results.sort(key=lambda r: r.fid)

    total_time = time() - start

    experiment_dir = f"time_measurements/gp_baseline_gp_dim{args.dim}_iid{args.iid}"
    os.makedirs(experiment_dir, exist_ok=True)
    timing_csv_path = os.path.join(experiment_dir, "timing_results.csv")

    df = pd.DataFrame(
        [
            {
                **asdict(result),
                "method": "gp_baseline",
                "model": "gp",
                "dim": args.dim,
                "iid": args.iid,
                "problem_class": args.problem_class,
                "total_time_seconds": total_time,
            }
            for result in timing_results
        ]
    )
    df.to_csv(timing_csv_path, index=False)
    print(f"Timing results saved to {timing_csv_path}")


if __name__ == "__main__":
    main()
