import asyncio
import json
import os

from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.experiment_loader import save_to_df
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt
from ela_guided_llm_bench.llm.gemini import generate_function
from ela_guided_llm_bench.prompt import LLM_SR_PROMPT
from ela_guided_llm_bench.selection import select_examples_by_roulette
from ela_guided_llm_bench.visualization import compare_contours, compare_ela_features, plot_target_values
from ioh import ProblemClass, get_problem

IID = 2
DIM = 2

DIR_NAME = "results_26_06_gemini_2.5_flash"
MODEL = "gemini-2.5-flash"  # "gemini-2.0-flash"
MODEL_TYPE = "flash"  # "flash" if "flash" in MODEL else "pro"
MODEL_VERSION = "2.5"  # "2.0" if "2.0" in MODEL else "2.5"


def format_examples(examples: list[FunctionInfo]) -> str:
    return "\n".join(str(example) for example in examples)


async def main():
    for fid in range(20, 25):
        experiment_name = f"llm_sr_{MODEL_VERSION}_{MODEL_TYPE}_f{fid}_iid{IID}_dim{DIM}"
        os.makedirs(f"./{DIR_NAME}/{experiment_name}", exist_ok=True)
        target_problem = get_problem(fid, IID, DIM, problem_class=ProblemClass.BBOB)
        target_ela_features = get_ela_features(target_problem, DIM)
        with open(f"./{DIR_NAME}/{experiment_name}/target_ela_features.json", "w") as f:
            json.dump(target_ela_features, f)
        generated_functions_info = []
        for epoch in range(250):
            try:
                examples = (
                    select_examples_by_roulette(generated_functions_info, k=min(epoch, 10))
                    if generated_functions_info
                    else []
                )
                prompt = LLM_SR_PROMPT.format(
                    ela_features=features_to_prompt(target_ela_features),
                    context=format_examples(examples),
                )
                response = await generate_function(prompt=prompt, model=MODEL, temperature=1.0)
                print(response)
                function_parser = FunctionParser(
                    ela_dim=2,
                    target_ela_features=target_ela_features,
                    problem_with_params=True,
                    max_evals=100,
                )
                function_info = function_parser.parse(response)
                if DIM == 2:
                    compare_contours(
                        problem1=function_info.function,
                        problem2=target_problem,
                        ela_features1=function_info.ela_features,
                        ela_features2=target_ela_features,
                        save_path=f"./{DIR_NAME}/{experiment_name}/epoch_{epoch}.png",
                    )
                else:
                    compare_ela_features(
                        ela_features1=function_info.ela_features,
                        ela_features2=target_ela_features,
                        save_path=f"./{DIR_NAME}/{experiment_name}/epoch_{epoch}.png",
                    )

                generated_functions_info.append(function_info)
                save_to_df(
                    generated_functions_info,
                    f"./{DIR_NAME}/{experiment_name}/generated_functions_info.csv",
                )
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                continue
        best_function_info = min(generated_functions_info, key=lambda x: x.distance_to_target)
        best_function_info.optimize_params(target_ela_features, max_evals=1000, algorithm="CMA-ES")
        generated_functions_info.append(function_info)
        save_to_df(
            generated_functions_info,
            f"./{DIR_NAME}/{experiment_name}/generated_functions_info.csv",
        )
        plot_target_values(
            [info.distance_to_target for info in generated_functions_info],
            save_path=f"./{DIR_NAME}/{experiment_name}/target_values.png",
        )


if __name__ == "__main__":
    asyncio.run(main())
