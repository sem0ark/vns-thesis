import random
from typing import Iterable, cast

import numpy as np

from src.core.abstract import Delta
from src.problems.moscp.problem import MOSCPProblem, MOSCPSolution, _MOSCPSolution


class FlipDelta(Delta[np.ndarray]):
    def __init__(self, flip_index) -> None:
        super().__init__()
        self.flip_index = flip_index

    def apply(self, data: np.ndarray):
        data[self.flip_index] = np.logical_not(data[self.flip_index])

    def revert(self, data: np.ndarray):
        data[self.flip_index] = np.logical_not(data[self.flip_index])


def flip_op(solution: MOSCPSolution) -> Iterable[MOSCPSolution]:
    for i in range(len(solution.data)):
        delta = FlipDelta(i)

        delta.apply(solution.data)
        objectives = solution.problem.calculate_objectives(solution.data)
        delta.revert(solution.data)

        yield _MOSCPSolution(solution.data, solution.problem, objectives, delta)


def flip_op_v2(solution: MOSCPSolution) -> Iterable[MOSCPSolution]:
    problem = cast(MOSCPProblem, solution.problem)

    current_set_selection = solution.data
    current_objectives = np.array(solution.objectives)

    for i in range(problem.num_variables):
        delta = FlipDelta(i)

        delta.apply(solution.data)
        is_feasible = problem.satisfies_constraints(solution.data)
        delta.revert(solution.data)

        if not is_feasible:
            continue

        is_selected = current_set_selection[i]
        set_costs = problem.costs[:, i]
        if is_selected:
            new_objectives = current_objectives - set_costs
        else:
            new_objectives = current_objectives + set_costs

        yield _MOSCPSolution(
            solution.data,
            problem,
            objectives=tuple(new_objectives.tolist()),
            delta=delta,
        )


def shake_flip(solution: MOSCPSolution, k: int) -> MOSCPSolution:
    new_data = solution.get_data_copy()
    flip_indices = random.sample(range(len(new_data)), k)
    new_data[flip_indices] = np.logical_not(new_data[flip_indices])
    return _MOSCPSolution(
        new_data, solution.problem, solution.problem.calculate_objectives(new_data)
    )
