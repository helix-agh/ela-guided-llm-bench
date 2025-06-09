import asyncio
import os

from dotenv import load_dotenv
from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.eoh.eoh import EOH
from ela_guided_llm_bench.openai import generate_function
from ioh import ProblemClass, get_problem

load_dotenv()
MODEL = "gpt-4.1-mini"


async def main():
    async def generate_function_wrapped(prompt: str) -> str:
        return await generate_function(prompt=prompt, model=MODEL, temperature=1.0)

    fid = 16
    IID = 1
    DIM = 2
    target_problem = get_problem(fid, IID, DIM, problem_class=ProblemClass.BBOB)
    target_ela_features = get_ela_features(target_problem, DIM)
    dir_name = f"eoh_results_{fid}_{IID}_{DIM}"
    os.makedirs(dir_name, exist_ok=True)
    eoh = EOH(
        target_problem=target_problem,
        target_ela_features=target_ela_features,
        generate_function=generate_function_wrapped,
        ela_dim=2,
        pop_size=4,
        n_iter=2,
        m=3,
        dir_name=dir_name,
    )
    await eoh.run()


if __name__ == "__main__":
    asyncio.run(main())
