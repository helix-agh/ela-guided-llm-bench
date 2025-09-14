from typing import Callable

from ela_guided_llm_bench.experiment_loader import save_to_df
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt
from ela_guided_llm_bench.llm.executor import Executor
from ela_guided_llm_bench.naive.prompt import ZERO_SHOT_PROMPT
from ela_guided_llm_bench.visualization import compare_contours


class ZeroShot:
    def __init__(
        self,
        target_problem: Callable,
        target_ela_features: dict,
        generate_function: Callable,
        executor: Executor,
        ela_dim: int,
        dir_name: str,
        n_evaluations: int,
    ):
        self.target_problem = target_problem
        self.target_ela_features = target_ela_features
        self.target_ela_features_formatted: str = features_to_prompt(target_ela_features)
        self.parser = FunctionParser(
            ela_dim=ela_dim,
            target_ela_features=target_ela_features,
            problem_with_params=False,
        )
        self.dir_name = dir_name
        self.n_evaluations = n_evaluations
        self.generate_function = generate_function
        self.executor = executor
        self.prompt = ZERO_SHOT_PROMPT.format(ela_features=self.target_ela_features_formatted)
        self.population: list[FunctionInfo] = []

    async def get_function_info(self, prompt: str) -> FunctionInfo:
        raw_function = await self.generate_function(prompt)
        return self.parser.parse(raw_function)

    async def run(self):
        tasks = [self.get_function_info(self.prompt) for _ in range(self.n_evaluations)]
        population = await self.executor.process_tasks_in_batches(tasks)

        for iteration, function_info in enumerate(population):
            if function_info is None:
                print(f"Iter: {iteration}, Offspring is None")
            else:
                compare_contours(
                    problem1=function_info.function,
                    problem2=self.target_problem,
                    ela_features1=function_info.ela_features,
                    ela_features2=self.target_ela_features,
                    save_path=f"./{self.dir_name}/epoch_{iteration}.png",
                )
        save_to_df(population, f"./{self.dir_name}/generated_functions_info.csv")
