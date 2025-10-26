from functools import lru_cache
from typing import Any, Callable
from src.core.abstract import Solution
from src.vns.acceptance import (
    AcceptBatchWrapped,
    AcceptBeamWrapped,
    ComparisonResult,
    is_dominating_min,
)


ObjectiveFunction = Callable[[Any], tuple[float, ...]]


def make_comparator(additional_objective_functions: list[ObjectiveFunction]):
    funcs = [lru_cache(maxsize=300)(func) for func in additional_objective_functions]

    def compare_solutions_better_stacked(
        new_solution: Solution, current_solution: Solution
    ) -> ComparisonResult:
        result = is_dominating_min(new_solution.objectives, current_solution.objectives)
        if result != ComparisonResult.NON_DOMINATED:
            return result

        for func in funcs:
            result = is_dominating_min(
                func(new_solution.data), func(current_solution.data)
            )
            if result != ComparisonResult.NON_DOMINATED:
                return result

        return result

    return compare_solutions_better_stacked


class AcceptBeamVFS(AcceptBeamWrapped):
    def __init__(self, additional_objective_functions: list[ObjectiveFunction]):
        super().__init__(make_comparator(additional_objective_functions))


class AcceptBatchVFS(AcceptBatchWrapped):
    def __init__(self, additional_objective_functions: list[ObjectiveFunction]):
        super().__init__(make_comparator(additional_objective_functions))
