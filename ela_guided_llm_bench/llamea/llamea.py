import random
from typing import Callable

from ela_guided_llm_bench.executor import Executor
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt, save_to_df
from ela_guided_llm_bench.llamea.prompt import EVOLUTION_PROMPT, INITIAL_PROMPT
from ela_guided_llm_bench.visualization import compare_contours


class LLaMEA:
    def __init__(
        self,
        target_problem: Callable,
        target_ela_features: dict,
        generate_function: Callable,
        ela_dim: int,
        dir_name: str,
        executor: Executor,
        pop_size: int,
        n_iter: int,
        n_offspring: int = 10,
        elitism: bool = False,
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
        self.executor = executor
        self.generate_function = generate_function
        self.mutation_prompts = [
            "Refine the function to improve it.",
            "The new function should be as diverse as possible from the previous ones on every aspect.",
            "Simplify the function to make it easier to understand.",
            "Add completely new mathematical operation that was not tested before.",
        ]
        self.pop_size = pop_size
        self.n_offspring = n_offspring
        self.n_iter = n_iter
        self.elitism = elitism
        self.population: list[FunctionInfo] = []
        self.history: list[list[FunctionInfo]] = []

    async def get_function_info(self, prompt: str) -> FunctionInfo:
        raw_function = await self.generate_function(prompt)
        return self.parser.parse(raw_function)

    async def initialize_single(self):
        prompt = INITIAL_PROMPT.format(ela_features=self.target_ela_features_formatted)
        new_individual = await self.get_function_info(prompt)
        return new_individual

    async def population_generation(self) -> list[FunctionInfo]:
        tasks = [self.initialize_single() for _ in range(self.pop_size)]
        return await self.executor.process_tasks_in_batches(tasks)

    def selection(self, parents: list[FunctionInfo], offsprings: list[FunctionInfo]):
        if self.elitism:
            combined_population = parents + offsprings
            combined_population.sort(key=lambda info: (info.distance_to_target if info is not None else float("inf")))
            new_population = combined_population[: self.pop_size]
        else:
            offsprings.sort(key=lambda info: (info.distance_to_target if info is not None else float("inf")))
            new_population = offsprings[: self.pop_size]

        return new_population

    async def evolve_solution(self, individual: FunctionInfo) -> FunctionInfo:
        new_prompt = EVOLUTION_PROMPT.format(
            ela_features=self.target_ela_features_formatted,
            population_summary="\n".join([ind.summary for ind in self.population]),
            description=individual.description,
            source_code=individual.source_code,
            mutation_operator=random.choice(self.mutation_prompts),
        )
        return await self.get_function_info(new_prompt)

    async def run(self):
        self.population = await self.population_generation()
        self.history = [self.population.copy()]
        for iteration in range(1, self.n_iter + 1):
            new_offspring_population = random.choices(self.population, k=self.n_offspring)
            print(f"Iteration {iteration}")
            for offspring in new_offspring_population:
                if offspring:
                    print(offspring.summary)
            new_population = await self.executor.process_tasks_in_batches(
                [self.evolve_solution(individual) for individual in new_offspring_population]
            )
            self.log_new_solutions(new_population, iteration)
            self.population = self.selection(self.population, new_population)
            self.history.append(new_population)
        flattened_history = [
            individual for generation in self.history for individual in generation if individual is not None
        ]
        save_to_df(flattened_history, f"./{self.dir_name}/generated_functions_info.csv")

    def log_new_solutions(self, offsprings: list[FunctionInfo], iter: int) -> None:
        for offspring_idx, function_info in enumerate(offsprings):
            if function_info is None:
                print(f"Iter: {iter}, Offspring {offspring_idx} is None")
            else:
                compare_contours(
                    problem1=function_info.function,
                    problem2=self.target_problem,
                    ela_features1=function_info.ela_features,
                    ela_features2=self.target_ela_features,
                    save_path=f"./{self.dir_name}/epoch_{iter}_offspring_{offspring_idx}.png",
                )
