from itertools import product
from typing import Any, Callable, Iterable, NamedTuple, cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.algorithm import Algorithm as PymooAlgorithm
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination

from src.cli.problem_cli import InstanceRunner, RunConfig
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.core.abstract import Problem, Solution


class PymooResult(NamedTuple):
    F: np.ndarray
    X: np.ndarray


class PymooProblemConfig(NamedTuple):
    problem_instance: Problem
    serialize_output: Callable[[PymooResult], list[Solution]]

    pymoo_problem: PymooProblem


class BasePymooInstanceRunner(InstanceRunner):
    def __init__(self, run_config: RunConfig, problem_config: PymooProblemConfig):
        super().__init__(run_config)
        self.problem_config = problem_config

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        return []

    def make_func(self, name: str, algorithm: PymooAlgorithm):
        def run(config: RunConfig):
            res = minimize(
                problem=self.problem_config.pymoo_problem,
                algorithm=algorithm,
                termination=TimeBasedTermination(config.run_time_seconds),
                verbose=True,
            )

            results = res.F
            if results is None:
                raise ValueError("Expected res.F to be non-null")

            solution_data = cast(None | np.ndarray, res.X)
            if solution_data is None:
                raise ValueError("Expected res.X to be non-null")

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=name,
                    version=5,
                    problem_name=self.problem_config.problem_instance.problem_name,
                    instance_name=config.instance_path.stem,
                ),
                solutions=[
                    SavedSolution(sol.objectives, sol.to_json_serializable())
                    for sol in self.problem_config.serialize_output(
                        PymooResult(results, solution_data)
                    )
                ],
            )

        return run


def get_nsga_variants(
    _config: PymooProblemConfig, sampling: Sampling
) -> Iterable[tuple[str, NSGA2]]:
    population_sizes = [50, 100, 150, 200, 300, 400, 500, 700, 1000]

    for (population,) in product(population_sizes):
        name = f"pymoo NSGA2 pop_{population}"
        yield (
            name,
            NSGA2(
                sampling=cast(Any, sampling),
                eliminate_duplicates=True,
                pop_size=population,
            ),
        )


def get_spea_variants(
    _config: PymooProblemConfig, sampling: Sampling
) -> Iterable[tuple[str, SPEA2]]:
    population_sizes = [50, 100, 150, 200, 300, 400, 500, 700, 1000]

    for (population,) in product(population_sizes):
        name = f"pymoo SPEA2 pop_{population}"
        yield (
            name,
            SPEA2(
                sampling=cast(Any, sampling),
                eliminate_duplicates=True,
                pop_size=population,
            ),
        )
