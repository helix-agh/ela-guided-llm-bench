import asyncio
import json
import os

from dotenv import load_dotenv
from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.executor import NaiveExecutor
from ela_guided_llm_bench.gemini import gemini_key_rotator, generate_function
from ela_guided_llm_bench.llamea.llamea import LLaMEA
from ioh import ProblemClass, get_problem

load_dotenv()
MODEL = "gemini-2.0-flash"


async def main():
    async def generate_function_wrapped(prompt: str) -> str:
        return await generate_function(prompt=prompt, model=MODEL, temperature=1.0)

    for fid in range(6, 21):
        attempts = 0
        while attempts < 3:
            try:
                IID = 1
                DIM = 2
                target_problem = get_problem(fid, IID, DIM, problem_class=ProblemClass.BBOB)
                target_ela_features = get_ela_features(target_problem, DIM)
                dir_name = f"./results_24_07_2.0_flash/llm_llamea_2.0_flash_f{fid}_iid{IID}_dim{DIM}"
                os.makedirs(dir_name, exist_ok=True)
                with open(f"{dir_name}/target_ela_features.json", "w") as f:
                    json.dump(target_ela_features, f)
                executor = NaiveExecutor(
                    gemini_key_rotator=gemini_key_rotator,
                    delay_in_seconds=0.5,
                )
                # executor = Executor(
                #     gemini_key_rotator=gemini_key_rotator,
                #     delay_in_seconds=15.0,
                #     batch_size=5,
                #     adaptive_batching=True,
                # )
                llamaea = LLaMEA(
                    target_problem=target_problem,
                    target_ela_features=target_ela_features,
                    generate_function=generate_function_wrapped,
                    ela_dim=2,
                    dir_name=dir_name,
                    executor=executor,
                    pop_size=5,
                    n_iter=50,
                    n_offspring=5,
                    elitism=True,
                )
                await llamaea.run()
                break
            except Exception as e:
                print(f"Error running LLaMEA for function {fid}: {e}")
                attempts += 1
                if attempts == 3:
                    raise e


if __name__ == "__main__":
    asyncio.run(main())
