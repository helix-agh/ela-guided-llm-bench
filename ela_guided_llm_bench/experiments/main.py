import argparse
import asyncio
import os

from dotenv import load_dotenv
from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.eoh.eoh import EOH
from ela_guided_llm_bench.experiment_config import ExperimentConfig
from ela_guided_llm_bench.experiments.affinic import AFFINIC_PROBLEMS
from ela_guided_llm_bench.llamea.llamea import LLaMEA
from ela_guided_llm_bench.llm.executor import Executor, NaiveExecutor
from ela_guided_llm_bench.llm.gemini import gemini_key_rotator, generate_function
from ela_guided_llm_bench.naive.zero_shot import ZeroShot
from ioh import ProblemClass, get_problem


def parse_args():
    parser = argparse.ArgumentParser(description="Run EOH experiment with specified parameters")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--method",
        choices=["llamea", "eoh", "zero_shot"],
        default="eoh",
        help="Method to use (default: eoh)",
    )
    parser.add_argument("--start-fid", type=int, default=1, help="Starting function ID (default: 1)")
    parser.add_argument("--end-fid", type=int, default=24, help="Ending function ID (default: 24)")
    parser.add_argument(
        "--executor",
        choices=["Executor", "NaiveExecutor"],
        default="NaiveExecutor",
        help="Executor type to use (default: NaiveExecutor)",
    )
    parser.add_argument("--dim", type=int, default=2, help="Problem dimension (default: 2)")
    parser.add_argument("--iid", type=int, default=1, help="Instance ID (default: 1)")
    parser.add_argument(
        "--problem-class",
        choices=["BBOB", "AFFINIC"],
        default="BBOB",
        help="Problem class (default: BBOB)",
    )
    return parser.parse_args()


async def main():
    load_dotenv()
    args = parse_args()

    async def generate_function_wrapped(prompt: str) -> str:
        return await generate_function(prompt=prompt, model=args.model, temperature=1.0)

    for fid in range(args.start_fid, args.end_fid + 1):
        attempts = 0
        while attempts < 3:
            try:
                if args.problem_class == "BBOB":
                    target_problem = get_problem(fid, args.iid, args.dim, problem_class=ProblemClass.BBOB)
                elif args.problem_class == "AFFINIC":
                    target_problem = AFFINIC_PROBLEMS[fid - 1]
                target_ela_features = get_ela_features(target_problem, args.dim)
                config = ExperimentConfig(
                    model=args.model,
                    fid=fid,
                    iid=args.iid,
                    dim=args.dim,
                    method=args.method,
                )
                os.makedirs(config.dir_name, exist_ok=True)

                if args.executor == "Executor":
                    executor = Executor(
                        gemini_key_rotator=gemini_key_rotator,
                        batch_size=5,
                        delay_in_seconds=0.5,
                    )
                else:
                    executor = NaiveExecutor(
                        gemini_key_rotator=gemini_key_rotator,
                        delay_in_seconds=10.0,
                    )
                if args.method == "eoh":
                    eoh = EOH(
                        target_problem=target_problem,
                        target_ela_features=target_ela_features,
                        generate_function=generate_function_wrapped,
                        ela_dim=args.dim,
                        pop_size=10,
                        n_iter=5,
                        m=5,
                        experiment_config=config,
                        executor=executor,
                    )
                    experiment = await eoh.run()
                    experiment.save_to_dir()
                    break
                elif args.method == "llamea":
                    llamaea = LLaMEA(
                        target_problem=target_problem,
                        target_ela_features=target_ela_features,
                        generate_function=generate_function_wrapped,
                        ela_dim=args.dim,
                        experiment_config=config,
                        executor=executor,
                        pop_size=5,
                        n_iter=50,
                        n_offspring=5,
                        elitism=True,
                    )
                    experiment = await llamaea.run()
                    experiment.save_to_dir()
                    break
                elif args.method == "zero_shot":
                    zero_shot = ZeroShot(
                        target_problem=target_problem,
                        target_ela_features=target_ela_features,
                        generate_function=generate_function_wrapped,
                        experiment_config=config,
                        executor=executor,
                        n_evaluations=100,
                    )
                    experiment = await zero_shot.run()
                    experiment.save_to_dir()
                    break
            except Exception as e:
                print(f"Error running {args.method} for function {fid}: {e}")
                attempts += 1
                if attempts == 3:
                    raise e


if __name__ == "__main__":
    asyncio.run(main())
