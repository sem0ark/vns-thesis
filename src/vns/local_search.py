from ast import TypeVar
from typing import Callable, Iterable

from src.vns.abstract import NeighborhoodOperator, Solution, VNSOptimizerAbstract
from src.vns.acceptance import ComparisonResult, compare

T = TypeVar("T")
SearchFunction = Callable[
    [Solution[T], VNSOptimizerAbstract[T]], Iterable[Solution[T] | None]
]


def noop(*_args):
    """Skip search phase, used for RVNS."""

    def search(
        initial: Solution, _config: VNSOptimizerAbstract
    ) -> Iterable[Solution | None]:
        yield initial

    return search


def best_improvement(
    operator: NeighborhoodOperator, objective_index: int | None = None
):
    """Go as far as possible, each time moving to the best near neighbor."""

    if objective_index is not None:
        is_better = lambda a, b: a[objective_index] < b[objective_index]
    else:
        is_better = lambda a, b: compare(a, b) == ComparisonResult.STRICTLY_BETTER

    def search(
        initial: Solution, config: VNSOptimizerAbstract
    ) -> Iterable[Solution | None]:
        current = initial

        while True:
            improved = False
            best_found_in_neighborhood = current

            for neighbor in operator(current, config):
                if is_better(
                    neighbor.objectives, best_found_in_neighborhood.objectives
                ):
                    best_found_in_neighborhood = neighbor
                    improved = True

            if improved:
                current = best_found_in_neighborhood
                yield None
            else:
                break

        yield current

    return search


def first_improvement(
    operator: NeighborhoodOperator, objective_index: int | None = None
):
    """Go as far as possible, each time moving to the first better near neighbor."""

    if objective_index is not None:
        is_better = lambda a, b: a[objective_index] < b[objective_index]
    else:
        is_better = lambda a, b: compare(a, b) == ComparisonResult.STRICTLY_BETTER

    def search(
        initial: Solution, config: VNSOptimizerAbstract
    ) -> Iterable[Solution | None]:
        current = initial

        while True:
            improvement_found = False

            for neighbor in operator(current, config):
                if is_better(neighbor.objectives, current.objectives):
                    current = neighbor
                    improvement_found = True

                    yield None
                    break

            if not improvement_found:
                break

        yield current

    return search


def first_improvement_quick(
    operator: NeighborhoodOperator, objective_index: int | None = None
):
    """Move to the first better near neighbor once."""

    if objective_index is not None:
        is_better = lambda a, b: a[objective_index] < b[objective_index]
    else:
        is_better = lambda a, b: compare(a, b) == ComparisonResult.STRICTLY_BETTER

    def search(
        initial: Solution, config: VNSOptimizerAbstract
    ) -> Iterable[Solution | None]:
        for neighbor in operator(initial, config):
            if is_better(neighbor.objectives, initial.objectives):
                yield neighbor
                return

        yield initial

    return search


def composite(search_functions: list[SearchFunction]):
    def search(
        initial: Solution, config: VNSOptimizerAbstract
    ) -> Iterable[Solution | None]:
        current = initial

        vnd_level = 0
        while vnd_level < len(search_functions):
            improved = False
            local_best_solution = current
            for local_solution in search_functions[vnd_level](current, config):
                if not local_solution:
                    yield None
                    continue

                if (
                    compare(local_solution.objectives, local_best_solution.objectives)
                    == ComparisonResult.STRICTLY_BETTER
                ):
                    improved = True
                    local_best_solution = local_solution

            if improved:
                vnd_level = 0
                current = local_best_solution
            else:
                vnd_level += 1

        yield current

    return search
