from pathlib import Path
from typing import Any, cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.cli.cli import Metadata, SavedRun, SavedSolution

from src.examples.mokp.mokp_problem import MOKPProblem, MOKPPymoo


def solve_mokp_nsga2(instance_path: str, run_seconds: float):
    problem_instance = MOKPProblem.load(instance_path)
    problem = MOKPPymoo(problem_instance)

    algorithm = NSGA2(eliminate_duplicates=True)
    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=TimeBasedTermination(run_seconds),
        verbose=True,
    )

    if res.X is not None:
        nd_sorting = NonDominatedSorting()
        non_dominated_indices = nd_sorting.do(res.F, only_non_dominated_front=True)
        final_solutions_F = res.F[non_dominated_indices]
    else:
        final_solutions_F = np.array([])

    solutions_data = [
        SavedSolution(cast(np.ndarray, sol).tolist()) for sol in final_solutions_F
    ]

    return SavedRun(
        metadata=Metadata(
            run_time_seconds=int(run_seconds),
            name="NSGA2",
            version=3,
            problem_name="mokp",
            instance_name=Path(instance_path).stem,
        ),
        solutions=solutions_data,
    )


def register_cli(cli: Any) -> None:
    cli.register_runner("mokp", [("NSGA2", solve_mokp_nsga2)])
