import asyncio
import os

import numpy as np
from ela_guided_llm_bench.ela import features_to_prompt, get_ela_features
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser
from ela_guided_llm_bench.visualization import compare_contours, plot_target_values, save_to_df
from ioh import ProblemClass, get_problem

from .gemini import generate_function
from .prompt import PROMPT_FEW_SHOT

FID = 22
IID = 2
DIM = 2

EXPERIMENT_NAME = "few_shot_flash"

EXAMPLE_PROMPT = """
    <previous_attempt>
    * **Previously Generated Function:**
    ```python
    {example_source_code}
    ```
    * **Its ELA Features:**
    {example_ela_features}
    * **Error (Previous ELA - Target ELA):**
    {example_error}
    </previous_attempt>
    """


def format_examples(examples: list[FunctionInfo]) -> str:
    formatted_examples = []
    for i, example in enumerate(examples):
        formatted_examples.append(
            EXAMPLE_PROMPT.format(
                example_source_code=example.source_code,
                example_ela_features=features_to_prompt(example.ela_features),
                example_error=round(example.distance_to_target, 2),
            )
        )
    return "\n".join(formatted_examples)


def select_examples(examples: list[FunctionInfo], k: int = 3) -> list[FunctionInfo]:
    distances_to_target = [example.distance_to_target for example in examples]
    weights = 1.0 / (np.array(distances_to_target) + 1e-10)
    weights = weights / np.sum(weights)
    sampled_indices = np.random.choice(range(len(distances_to_target)), size=k, replace=False, p=weights)
    return [examples[i] for i in sampled_indices]


async def main():
    os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)
    target_problem = get_problem(FID, IID, DIM, problem_class=ProblemClass.BBOB)
    target_ela_features = get_ela_features(target_problem, DIM)
    generated_functions_info = []
    for epoch in range(5):
        try:
            examples = select_examples(generated_functions_info, k=min(epoch, 10)) if generated_functions_info else []
            context = format_examples(examples)
            prompt = PROMPT_FEW_SHOT.format(
                ela_features=features_to_prompt(target_ela_features),
                context=context,
            )
            print(prompt)
            response = await generate_function(
                prompt,
                model="gemini-2.0-flash",
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
