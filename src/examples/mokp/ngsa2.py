import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.cli.cli import Metadata, SavedRun, SavedSolution

from src.examples.mokp.mokp_problem import MOKPProblem

BASE = Path(__file__).parent.parent.parent / "runs"


class MOKP(ElementwiseProblem):
    """
    Multi-Objective Knapsack Problem.

    A problem is defined by inheriting from the Problem class and
    implementing the _evaluate method.
    """

    def __init__(self, problem: MOKPProblem):
        super().__init__(
            n_var=problem.num_items,
            n_obj=problem.num_objectives,
            n_constr=problem.num_limits,
            xl=0.0,
            xu=1.0,
            vtype=bool,
        )
        self.problem_instance = problem

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution `x`.
        `x` is a NumPy array representing a single solution (a vector of booleans).
        """
        # Calculate profits for all objectives
        total_profits = np.sum(x * self.problem_instance.profits, axis=1)

        # Objectives: We minimize the negative profits
        out["F"] = -total_profits

        # Constraints: A solution is feasible if its weight is within capacity for all limits.
        # This will be an array of values, one for each constraint.
        total_weights = np.sum(x * self.problem_instance.weights, axis=1)
        out["G"] = total_weights - self.problem_instance.capacity


def solve_mokp_ngsa2(instance_path: str, run_seconds: float):
    problem_instance = MOKPProblem.load(instance_path)
    problem = MOKP(problem_instance)

    algorithm = NSGA2(pop_size=200, eliminate_duplicates=True)

    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=TimeBasedTermination(run_seconds),
        verbose=True,
    )

    # Apply non-dominated sorting to the final population
    # Note: pymoo already does this internally, but it's good practice to ensure.
    if res.X is not None:
        nd_sorting = NonDominatedSorting()
        negated_F = -res.F
        non_dominated_indices = nd_sorting.do(negated_F, only_non_dominated_front=True)
        final_solutions_F = negated_F[non_dominated_indices]
    else:
        final_solutions_F = np.array([])

    solutions_data = [SavedSolution(cast(np.ndarray, -sol).tolist()) for sol in final_solutions_F]

    BASE.mkdir(parents=True, exist_ok=True)

    return SavedRun(
        metadata=Metadata(
            run_time_seconds=int(run_seconds),
            name="NGSA2",
            version=1,
            problem_name="mokp",
            instance_name=Path(instance_path).stem,
        ),
        solutions=solutions_data,
    )


def register_cli(cli: Any) -> None:
    cli.register_runner("mokp", [("NGSA2", solve_mokp_ngsa2)])
