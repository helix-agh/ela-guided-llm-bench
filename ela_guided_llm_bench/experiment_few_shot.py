import asyncio
import os

from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt
from ela_guided_llm_bench.visualization import compare_contours, plot_target_values, save_to_df
from ioh import ProblemClass, get_problem

from .gemini import generate_function
from .prompt import PROMPT_FEW_SHOT
from .selection import select_examples_by_roulette

FID = 2
IID = 2
DIM = 2

MODEL = "gemini-2.5-flash-preview-04-17"  # "gemini-2.0-flash"
MODEL_TYPE = "flash" if "flash" in MODEL else "pro"
EXPERIMENT_NAME = f"few_shot_{MODEL_TYPE}_f{FID}"


def format_examples(examples: list[FunctionInfo]) -> str:
    return "\n".join(str(example) for example in examples)


async def main():
    os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)
    target_problem = get_problem(FID, IID, DIM, problem_class=ProblemClass.BBOB)
    target_ela_features = get_ela_features(target_problem, DIM)
    generated_functions_info = []
    for epoch in range(200):
        try:
            examples = (
                select_examples_by_roulette(generated_functions_info, k=min(epoch, 10))
                if generated_functions_info
                else []
            )
            prompt = PROMPT_FEW_SHOT.format(
                ela_features=features_to_prompt(target_ela_features),
                context=format_examples(examples),
            )
            print(prompt)
            response = await generate_function(
                prompt,
                model=MODEL,
            )
            print(response)
            function_parser = FunctionParser(ela_dim=2, target_ela_features=target_ela_features)
            function_info = function_parser.parse(response)

            compare_contours(
                problem1=function_info.function,
                problem2=target_problem,
                ela_features1=function_info.ela_features,
                ela_features2=target_ela_features,
                save_path=f"./results/{EXPERIMENT_NAME}/epoch_{epoch}.png",
            )

            generated_functions_info.append(function_info)
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            continue
    plot_target_values(
        [info.distance_to_target for info in generated_functions_info],
        save_path=f"./results/{EXPERIMENT_NAME}/target_values.png",
    )
    save_to_df(
        generated_functions_info,
        f"./results/{EXPERIMENT_NAME}/generated_functions_info.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
