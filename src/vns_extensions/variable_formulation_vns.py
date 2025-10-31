from typing import Any, Callable

from src.core.abstract import Solution
from src.vns.acceptance import (
    AcceptBatch,
    AcceptBeam,
    ComparisonResult,
    is_dominating_min,
)

ObjectiveFunction = Callable[[Any], tuple[float, ...]]


def make_comparator(additional_objective_functions: list[ObjectiveFunction]):
    def get_objectives(solution: Solution):
        obj = solution.objectives
        for func in additional_objective_functions:
            obj += func(solution.data)
        return obj

    def compare_solutions_better_stacked(
        new_solution: Solution, current_solution: Solution
    ) -> ComparisonResult:
        return is_dominating_min(
            get_objectives(new_solution), get_objectives(current_solution)
        )

    return compare_solutions_better_stacked


class AcceptBeamVFS(AcceptBeam):
    def __init__(self, additional_objective_functions: list[ObjectiveFunction]):
        super().__init__(make_comparator(additional_objective_functions))


class AcceptBatchVFS(AcceptBatch):
    def __init__(self, *additional_objective_functions: ObjectiveFunction):
        super().__init__(make_comparator(list(additional_objective_functions)))
