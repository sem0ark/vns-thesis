from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.cli.cli import Metadata, SavedRun, SavedSolution
from src.examples.mokp.mokp_problem import MOKPProblem, MOKPPymoo


def run_pymoo(instance_path: str, run_seconds: float, name: str, algorithm: NSGA2 | SPEA2):
    problem_instance = MOKPProblem.load(instance_path)
    problem = MOKPPymoo(problem_instance)

    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=TimeBasedTermination(run_seconds),
        verbose=True,
    )

    results = res.F
    if results is not None:
        nd_sorting = NonDominatedSorting()
        non_dominated_indices = nd_sorting.do(results, only_non_dominated_front=True)

        final_solutions_F = results[non_dominated_indices]
    else:
        final_solutions_F = np.array([])

    solutions_data = [
        SavedSolution(cast(np.ndarray, sol).tolist()) for sol in final_solutions_F
    ]

    return SavedRun(
        metadata=Metadata(
            run_time_seconds=int(run_seconds),
            name=name,
            version=3,
            problem_name="mokp",
            instance_name=Path(instance_path).stem,
        ),
        solutions=solutions_data,
    )


def make_runner(name: str, algorithm: NSGA2):
    def start(instance_path: str, run_seconds: float):
        return run_pymoo(instance_path, run_seconds, name, algorithm)

    return start


def prepare_optimizers():
    algorithms = [
        ("NSGA2", NSGA2),
        ("SPEA2", SPEA2),
    ]
    population_sizes = [50, 100, 150, 200, 300]

    for (
        (algorithm_name, algorithm),
        population,
    ) in product(algorithms, population_sizes):
        name = f"{algorithm_name} pop_{population}"
        yield name, make_runner(name, algorithm(
            eliminate_duplicates=True,
            pop_size=population,
        ))


def register_cli(cli: Any) -> None:
    cli.register_runner("mokp", list(prepare_optimizers()))
