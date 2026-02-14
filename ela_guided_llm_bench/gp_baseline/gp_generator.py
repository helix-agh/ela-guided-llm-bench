import operator
import random
from functools import partial

import numpy as np
from deap import base, creator, gp, tools
from ela_guided_llm_bench.ela import get_distance, get_ela_features
from ela_guided_llm_bench.experiment import Experiment
from ela_guided_llm_bench.experiment_config import ExperimentConfig
from ela_guided_llm_bench.function import FunctionInfo

from .primitives import create_pset
from .tree_init import genHalfAndHalf_

PENALTY_FITNESS = 1e4

_PRIMITIVES_SOURCE = """\
import numpy as np

def first_dv(x):
    if np.isscalar(x):
        return x
    return x[0]

def trans_dv(x):
    if np.isscalar(x):
        return x
    return np.hstack((x[1:].ravel(), np.zeros((1, 1)).ravel()))

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

def div(x, y):
    return np.where(np.abs(y) > 1e-20, np.divide(x, y), 1.)

def neg(x):
    return -1 * x

def reciprocal(x):
    return np.where(np.abs(x) > 1e-20, np.divide(1, x), 1.)

def mul10(x):
    return 10 * x

def square(x):
    return np.square(x)

def sqrt(x):
    return np.sqrt(abs(x))

def abs_(x):
    return abs(x)

def roundoff(x):
    return np.round(x)

def sin(x):
    return np.sin(2 * np.pi * x)

def cos(x):
    return np.cos(2 * np.pi * x)

def ln(x):
    return np.where(np.abs(x) > 1e-20, np.log(abs(x)), 1.)

def exp(x):
    return np.exp(x)

def sum_vec(x):
    return np.sum(x)

def mean_vec(x):
    return np.mean(x)

def cumsum_vec(x):
    return np.cumsum(x)

def prod_vec(x):
    return np.prod(x)

def amax_vec(x):
    return np.amax(x)
"""


def _expr_to_source_code(expr_str: str) -> str:
    return _PRIMITIVES_SOURCE + f"\ndef problem(x):\n    return float(np.mean({expr_str}))\n"


def _make_problem(func_):
    def problem(x):
        result = func_(x)
        return float(np.mean(result))

    return problem


class GPBaseline:
    def __init__(
        self,
        target_ela_features: dict,
        dim: int,
        experiment_config: ExperimentConfig,
        population: int = 50,
        ngen: int = 5,
        cxpb: float = 0.5,
        mutpb: float = 0.1,
        tree_size: tuple = (8, 12),
        seed: int = 1,
        problem_type: str = "bbob",
        verbose: bool = True,
    ):
        self.target_ela_features = target_ela_features
        self.dim = dim
        self.experiment_config = experiment_config
        self.population_size = population
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tree_size = tree_size
        self.seed = seed
        self.problem_type = problem_type
        self.verbose = verbose

    def _evaluate(self, individual, pset, doe_x):
        try:
            func_ = gp.compile(expr=individual, pset=pset)
        except Exception:
            return (PENALTY_FITNESS,)

        problem = _make_problem(func_)

        try:
            test_outputs = []
            for i in range(min(10, len(doe_x))):
                val = problem(doe_x[i])
                test_outputs.append(val)
            test_outputs = np.array(test_outputs)
            if np.isnan(test_outputs).any() or np.isinf(test_outputs).any():
                return (PENALTY_FITNESS,)
            if np.var(test_outputs) < 1e-20:
                return (PENALTY_FITNESS,)
        except Exception:
            return (PENALTY_FITNESS,)

        try:
            ela_features = get_ela_features(problem, self.dim, problem_type=self.problem_type)
            distance = get_distance(ela_features, self.target_ela_features)
            return (distance,)
        except Exception:
            return (PENALTY_FITNESS,)

    def _individual_to_function_info(self, individual, pset):
        try:
            func_ = gp.compile(expr=individual, pset=pset)
            problem = _make_problem(func_)
            ela_features = get_ela_features(problem, self.dim, problem_type=self.problem_type)
            distance = get_distance(ela_features, self.target_ela_features)
            source_code = _expr_to_source_code(str(individual))
            return FunctionInfo(
                function=problem,
                source_code=source_code,
                description="GP-generated expression",
                ela_features=ela_features,
                distance_to_target=distance,
            )
        except Exception:
            return None

    def run(self) -> Experiment:
        np.random.seed(self.seed)
        random.seed(self.seed + 10)

        pset = create_pset()

        doe_x = np.random.uniform(-5, 5, (50, self.dim))

        genHalfAndHalf = partial(genHalfAndHalf_, doe_x)

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", genHalfAndHalf, pset=pset, min_=self.tree_size[0], max_=self.tree_size[1])
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self._evaluate, pset=pset, doe_x=doe_x)
        toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        all_function_infos: list[list[FunctionInfo]] = []

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        # Collect generation 0 (initial population)
        gen_infos = []
        for ind in pop:
            info = self._individual_to_function_info(ind, pset)
            if info is not None:
                gen_infos.append(info)
        all_function_infos.append(gen_infos)

        if self.verbose:
            best_fit = min(ind.fitness.values[0] for ind in pop)
            print(f"[GP] Gen 0: pop={len(pop)}, best_distance={best_fit:.4f}")

        for gen in range(1, self.ngen + 1):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            gen_infos = []
            for ind in pop:
                info = self._individual_to_function_info(ind, pset)
                if info is not None:
                    gen_infos.append(info)
            all_function_infos.append(gen_infos)

            if self.verbose:
                best_fit = min(ind.fitness.values[0] for ind in pop)
                print(f"[GP] Gen {gen}: pop={len(pop)}, best_distance={best_fit:.4f}")

        return Experiment(
            function_infos=all_function_infos,
            target_ela_features=self.target_ela_features,
            config=self.experiment_config,
        )
