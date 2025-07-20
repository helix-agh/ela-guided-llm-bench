import asyncio
import random
from typing import Callable, Literal

from ela_guided_llm_bench.executor import Executor
from ela_guided_llm_bench.function import FunctionInfo, FunctionParser, features_to_prompt, save_to_df
from ela_guided_llm_bench.prompt import E1_PROMPT, E2_PROMPT, I1_PROMPT, M1_PROMPT, M2_PROMPT, M3_PROMPT
from ela_guided_llm_bench.visualization import compare_contours


def parent_selection(pop: list[FunctionInfo], m: int) -> list[FunctionInfo]:
    sorted_pop = sorted([x for x in pop if x is not None], key=lambda x: x.distance_to_target)
    ranks = [i for i in range(len(sorted_pop))]
    probs = [1 / (rank + 1 + len(sorted_pop)) for rank in ranks]
    parents = random.choices(sorted_pop, weights=probs, k=min(m, len(sorted_pop)))
    return parents


class EOH:
    def __init__(
        self,
        target_problem: Callable,
        target_ela_features: dict,
        generate_function: Callable,
        ela_dim: int,
        pop_size: int,
        n_iter: int,
        m: int,
        dir_name: str,
        executor: Executor,
    ) -> None:
        self.target_problem = target_problem
        self.target_ela_features = target_ela_features
        self.target_ela_features_formatted: str = features_to_prompt(target_ela_features)
        self.generate_function = generate_function
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.m = m
        self.ela_dim = ela_dim
        self.executor = executor
        self.parser = FunctionParser(
            ela_dim=ela_dim,
            target_ela_features=target_ela_features,
            problem_with_params=False,
        )
        self.history: list[FunctionInfo] = []
        self.dir_name = dir_name

    async def run(self):
        print("Creating initial population:")
        await self.executor.log_key_usage_stats()
        population = await self.population_generation()
        self.log_new_solutions(population, 0)

        for iteration in range(1, self.n_iter + 1):
            print(f"\nIteration {iteration}/{self.n_iter}")
            await self._log_key_usage_stats()

            for operator in ["e1", "e2", "m1", "m2", "m3"]:
                print(f"Processing operator: {operator}")
                offspring_tasks = [self.get_offspring(population, operator) for _ in range(self.pop_size)]
                offsprings = await self.executor.process_tasks_in_batches(offspring_tasks)
                population = population + offsprings
                self.history.extend([offspring for offspring in offsprings if offspring is not None])
                self.log_new_solutions(offsprings, len(self.history) // self.pop_size)
                population = self.select(population)
                print(f"Waiting {self.executor.delay_in_seconds} seconds before next operator...")
                await asyncio.sleep(self.executor.delay_in_seconds)

        save_to_df(self.history, f"./{self.dir_name}/generated_functions_info.csv")

    async def population_generation(self) -> list[FunctionInfo]:
        tasks = [self.i1() for _ in range(self.pop_size)]
        return await self.executor.process_tasks_in_batches(tasks)

    async def get_function_info(self, prompt: str) -> FunctionInfo:
        raw_function = await self.generate_function(prompt)
        return self.parser.parse(raw_function)

    async def get_offspring(
        self,
        pop: list[FunctionInfo] | None = None,
        operator: Literal["e1", "e2", "m1", "m2", "m3"] = "e1",
    ) -> FunctionInfo:
        parents = parent_selection(pop, m=self.m)
        if operator == "e1":
            return await self.e1(parents)
        elif operator == "e2":
            return await self.e2(parents)
        elif operator == "m1":
            return await self.m1(parents[0])
        elif operator == "m2":
            return await self.m2(parents[0])
        elif operator == "m3":
            return await self.m3(parents[0])

    async def i1(self) -> FunctionInfo:
        prompt = I1_PROMPT.format(ela_features=self.target_ela_features_formatted)
        return await self.get_function_info(prompt)

    async def e1(self, parents: list[FunctionInfo]) -> FunctionInfo:
        parents_prompt = "\n".join(str(parent) for parent in parents)
        prompt = E1_PROMPT.format(context=parents_prompt, ela_features=self.target_ela_features_formatted)
        return await self.get_function_info(prompt)

    async def e2(self, parents: list[FunctionInfo]) -> FunctionInfo:
        parents_prompt = "\n".join(str(parent) for parent in parents)
        prompt = E2_PROMPT.format(context=parents_prompt, ela_features=self.target_ela_features_formatted)
        return await self.get_function_info(prompt)

    async def m1(self, parent: FunctionInfo) -> FunctionInfo:
        prompt = M1_PROMPT.format(context=str(parent), ela_features=self.target_ela_features_formatted)
        return await self.get_function_info(prompt)

    async def m2(self, parent: FunctionInfo) -> FunctionInfo:
        prompt = M2_PROMPT.format(context=str(parent), ela_features=self.target_ela_features_formatted)
        return await self.get_function_info(prompt)

    async def m3(self, parent: FunctionInfo) -> FunctionInfo:
        prompt = M3_PROMPT.format(context=str(parent), ela_features=self.target_ela_features_formatted)
        return await self.get_function_info(prompt)

    def select(self, population: list[FunctionInfo]) -> list[FunctionInfo]:
        sorted_pop = sorted([x for x in population if x is not None], key=lambda x: x.distance_to_target)
        return sorted_pop[: self.pop_size]

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
