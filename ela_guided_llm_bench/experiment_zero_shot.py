import asyncio
import os
import time

from ela_guided_llm_bench.ela import get_ela_features
from ela_guided_llm_bench.function import FunctionParser, features_to_prompt, save_to_df
from ela_guided_llm_bench.visualization import compare_contours, plot_target_values
from ioh import ProblemClass, get_problem

from .openai import generate_function
from .prompt import PROMPT_ZERO_SHOT

# Experiment configuration
FID = 22
IID = 2
DIM = 2
EXPERIMENT_NAME = "zero_shot_20250615"

N_FUNCTIONS = 100
BATCH_SIZE = 10
MODEL = "gpt-4.1-mini"
TEMPERATURE = 1.0


async def generate_single_function(
    prompt: str,
    model: str,
    temperature: float,
    function_id: int,
    target_ela_features: dict,
) -> tuple[int, dict]:
    try:
        start_time = time.time()
        response = await generate_function(prompt, model=model, temperature=temperature)

        function_parser = FunctionParser(ela_dim=DIM, target_ela_features=target_ela_features)
        function_info = function_parser.parse(response)

        elapsed_time = time.time() - start_time

        return function_id, {
            "function_info": function_info,
            "response": response,
            "elapsed_time": elapsed_time,
            "success": True,
            "error": None,
        }
    except Exception as e:
        print(f"Error generating function {function_id}: {e}")
        return function_id, {
            "function_info": None,
            "response": None,
            "elapsed_time": 0,
            "success": False,
            "error": str(e),
        }


async def generate_batch(
    prompt: str,
    model: str,
    temperature: float,
    target_ela_features: dict,
    batch_start_id: int,
    batch_size: int,
) -> list[tuple[int, dict]]:
    tasks = []
    for i in range(batch_size):
        function_id = batch_start_id + i
        task = generate_single_function(prompt, model, temperature, function_id, target_ela_features)
        tasks.append(task)

    print(f"Starting batch generation for functions {batch_start_id} to {batch_start_id + batch_size - 1}")
    batch_start_time = time.time()

    results = await asyncio.gather(*tasks)

    batch_elapsed_time = time.time() - batch_start_time
    successful_results = [r for r in results if r[1]["success"]]

    print(f"Batch completed in {batch_elapsed_time:.2f}s. " f"Success rate: {len(successful_results)}/{len(results)}")

    return results


def save_visualization(
    function_info,
    target_problem,
    target_ela_features,
    function_id: int,
    experiment_dir: str,
) -> None:
    try:
        if function_info.function is not None:
            compare_contours(
                problem1=function_info.function,
                problem2=target_problem,
                ela_features1=function_info.ela_features,
                ela_features2=target_ela_features,
                save_path=f"{experiment_dir}/function_{function_id}.png",
            )
    except Exception as e:
        print(f"Error saving visualization for function {function_id}: {e}")


async def main():
    experiment_dir = f"./results/{EXPERIMENT_NAME}"
    os.makedirs(experiment_dir, exist_ok=True)

    target_problem = get_problem(FID, IID, DIM, problem_class=ProblemClass.BBOB)
    target_ela_features = get_ela_features(target_problem, DIM)

    prompt = PROMPT_ZERO_SHOT.format(ela_features=features_to_prompt(target_ela_features))

    print("Starting zero-shot experiment:")
    print(f"- Target: F{FID}, IID{IID}, DIM{DIM}")
    print(f"- Total functions: {N_FUNCTIONS}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Model: {MODEL}")
    print(f"- Temperature: {TEMPERATURE}")
    print("-" * 50)

    generated_functions_info = []
    successful_generations = 0
    total_start_time = time.time()

    # Process in batches
    for batch_start in range(0, N_FUNCTIONS, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, N_FUNCTIONS - batch_start)
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (N_FUNCTIONS + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n=== Batch {batch_num}/{total_batches} ===")

        batch_results = await generate_batch(prompt, MODEL, TEMPERATURE, target_ela_features, batch_start, batch_size)

        for function_id, result in batch_results:
            if result["success"]:
                function_info = result["function_info"]
                generated_functions_info.append(function_info)
                successful_generations += 1

                save_visualization(
                    function_info,
                    target_problem,
                    target_ela_features,
                    function_id,
                    experiment_dir,
                )

                print(f"  ✓ Function {function_id}: distance = {function_info.distance_to_target:.4f}")
            else:
                print(f"  ✗ Function {function_id}: {result['error']}")

        if generated_functions_info:
            save_to_df(
                generated_functions_info,
                f"{experiment_dir}/generated_functions_info.csv",
            )

            plot_target_values(
                [info.distance_to_target for info in generated_functions_info],
                save_path=f"{experiment_dir}/target_values.png",
            )

        progress = (batch_start + batch_size) / N_FUNCTIONS * 100
        print(f"Progress: {progress:.1f}% ({successful_generations}/{batch_start + batch_size} successful)")

    total_elapsed_time = time.time() - total_start_time

    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETED")
    print("=" * 50)
    print(f"Total time: {total_elapsed_time:.2f}s")
    print(f"Successful generations: {successful_generations}/{N_FUNCTIONS}")
    print(f"Success rate: {successful_generations/N_FUNCTIONS*100:.1f}%")

    if generated_functions_info:
        distances = [info.distance_to_target for info in generated_functions_info]
        print(f"Best distance: {min(distances):.4f}")
        print(f"Average distance: {sum(distances)/len(distances):.4f}")
        print(f"Worst distance: {max(distances):.4f}")

        save_to_df(
            generated_functions_info,
            f"{experiment_dir}/generated_functions_info.csv",
        )

        plot_target_values(
            distances,
            save_path=f"{experiment_dir}/target_values.png",
        )

        print(f"\nResults saved to: {experiment_dir}")
    else:
        print("No successful function generations!")


if __name__ == "__main__":
    asyncio.run(main())
