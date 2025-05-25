import asyncio
import os

from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.function import FunctionParser, features_to_prompt, save_to_df
from ela_guided_llm_bench.visualization import compare_contours, plot_target_values
from ioh import ProblemClass, get_problem

from .gemini import generate_function
from .prompt import PROMPT_ZERO_SHOT

FID = 22
IID = 2
DIM = 2

EXPERIMENT_NAME = "zero_shot"


async def main():
    os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)
    target_problem = get_problem(FID, IID, DIM, problem_class=ProblemClass.BBOB)
    target_ela_features = get_ela_features(target_problem, DIM)
    generated_functions_info = []
    for epoch in range(1):
        response = await generate_function(
            PROMPT_ZERO_SHOT.format(ela_features=features_to_prompt(target_ela_features)),
            model="gemini-2.5-pro-exp-03-25",
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
