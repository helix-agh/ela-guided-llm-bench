import asyncio
import os

from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt
from ela_guided_llm_bench.visualization import compare_contours, plot_target_values, save_to_df
from ioh import ProblemClass, get_problem

from .gemini import generate_function
from .prompt import LLM_SR_PROMPT
from .selection import select_examples_by_roulette

IID = 2
DIM = 2

DIR_NAME = "results"
MODEL = "gemini-2.5-flash-preview-04-17"  # "gemini-2.0-flash"
MODEL_TYPE = "flash" if "flash" in MODEL else "pro"


def format_examples(examples: list[FunctionInfo]) -> str:
    return "\n".join(str(example) for example in examples)


async def main():
    for fid in range(19, 25):
        experiment_name = f"llm_sr_{MODEL_TYPE}_f{fid}_iid{IID}_dim{DIM}"
        os.makedirs(f"./{DIR_NAME}/{experiment_name}", exist_ok=True)
        target_problem = get_problem(fid, IID, DIM, problem_class=ProblemClass.BBOB)
        target_ela_features = get_ela_features(target_problem, DIM)
        generated_functions_info = []
        for epoch in range(50):
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
                print(prompt)
                response = await generate_function(
                    prompt,
                    model=MODEL,
                )
                print(response)
                function_parser = FunctionParser(
                    ela_dim=2,
                    target_ela_features=target_ela_features,
                    problem_with_params=True,
                    max_evals=100,
                )
                function_info = function_parser.parse(response)

                compare_contours(
                    problem1=function_info.function,
                    problem2=target_problem,
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
        plot_target_values(
            [info.distance_to_target for info in generated_functions_info],
            save_path=f"./{DIR_NAME}/{experiment_name}/target_values.png",
        )


if __name__ == "__main__":
    asyncio.run(main())
