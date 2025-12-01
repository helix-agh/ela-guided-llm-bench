import asyncio
import os

from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.experiment import save_to_df
from ela_guided_llm_bench.function import FunctionParser, features_to_prompt
from ela_guided_llm_bench.llm.gemini import generate_function
from ela_guided_llm_bench.llm_sr.buffer import ExperienceBuffer
from ela_guided_llm_bench.naive.prompt import FEW_SHOT_PROMPT
from ela_guided_llm_bench.visualization import compare_contours, plot_target_values
from ioh import ProblemClass, get_problem

FID = 16
IID = 2
DIM = 2

MODEL = "gemini-2.0-flash"
MODEL_TYPE = "flash" if "flash" in MODEL else "pro"
EXPERIMENT_NAME = f"llmsr_style_{MODEL_TYPE}_f{FID}"


async def main():
    os.makedirs(f"./results/{EXPERIMENT_NAME}", exist_ok=True)
    target_problem = get_problem(FID, IID, DIM, problem_class=ProblemClass.BBOB)
    target_ela_features = get_ela_features(target_problem, DIM)
    buffer = ExperienceBuffer()
    for epoch in range(200):
        try:
            context, island_id = buffer.get_prompt_context()
            print(f"Island {island_id}")
            prompt = FEW_SHOT_PROMPT.format(
                ela_features=features_to_prompt(target_ela_features),
                context=context,
            )
            response = await generate_function(
                prompt,
                model=MODEL,
            )
            function_parser = FunctionParser(ela_dim=2, target_ela_features=target_ela_features)
            function_info = function_parser.parse(response)
            buffer.register_function(function_info, island_id)
            compare_contours(
                problem1=function_info.function,
                problem2=target_problem,
                ela_features1=function_info.ela_features,
                ela_features2=target_ela_features,
                save_path=f"./results/{EXPERIMENT_NAME}/epoch_{epoch}.png",
                dim=DIM,
            )
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            continue
    plot_target_values(
        [info.distance_to_target for info in buffer._all_functions],
        save_path=f"./results/{EXPERIMENT_NAME}/target_values.png",
    )
    save_to_df(
        buffer._all_functions,
        f"./results/{EXPERIMENT_NAME}/generated_functions_info.csv",
    )


if __name__ == "__main__":
    asyncio.run(main())
