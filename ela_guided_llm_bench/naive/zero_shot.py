from typing import Callable

from ela_guided_llm_bench.experiment import Experiment
from ela_guided_llm_bench.experiment_config import ExperimentConfig
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt
from ela_guided_llm_bench.llm.executor import BaseExecutor
from ela_guided_llm_bench.naive.prompt import ZERO_SHOT_PROMPT
from ela_guided_llm_bench.visualization import compare_contours


class ZeroShot:
    def __init__(
        self,
        target_problem: Callable,
        target_ela_features: dict,
        generate_function: Callable,
        executor: BaseExecutor,
        experiment_config: ExperimentConfig,
        n_evaluations: int,
        log_contours: bool = False,
        batch_size: int = 25,
    ):
        self.target_problem = target_problem
        self.target_ela_features = target_ela_features
        self.target_ela_features_formatted: str = features_to_prompt(target_ela_features)
        self.parser = FunctionParser(
            ela_dim=experiment_config.dim,
            target_ela_features=target_ela_features,
            problem_with_params=False,
        )
        self.n_evaluations = n_evaluations
        self.generate_function = generate_function
        self.executor = executor
        self.prompt = ZERO_SHOT_PROMPT.format(ela_features=self.target_ela_features_formatted)
        self.population: list[FunctionInfo] = []
        self.experiment_config = experiment_config
        self.log_contours = log_contours
        self.batch_size = batch_size

    async def get_function_info(self, prompt: str) -> FunctionInfo:
        raw_function = await self.generate_function(prompt)
        return self.parser.parse(raw_function)

    async def run(self) -> Experiment:
        tasks = [self.get_function_info(self.prompt) for _ in range(self.n_evaluations)]
        batched_tasks = [tasks[i : i + self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        population = []
        for batch in batched_tasks:
            batch_results = await self.executor.process_tasks_in_batches(batch)
            population.extend(batch_results)

        for iteration, function_info in enumerate(population):
            if function_info is None:
                print(f"Iter: {iteration}, Offspring is None")
            elif self.log_contours:
                compare_contours(
                    problem1=function_info.function,
                    problem2=self.target_problem,
                    ela_features1=function_info.ela_features,
                    ela_features2=self.target_ela_features,
                    save_path=f"./{self.experiment_config.dir_name}/epoch_{iteration}.png",
                    dim=self.experiment_config.dim,
                )
        return Experiment(
            function_infos=[population],
            target_ela_features=self.target_ela_features,
            config=self.experiment_config,
        )
