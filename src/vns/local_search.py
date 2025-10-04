from ast import TypeVar
from typing import Callable, Iterable

from src.vns.abstract import Solution
from src.vns.acceptance import ComparisonResult, is_dominating_min

T = TypeVar("T")
SearchFunction = Callable[[Solution], Iterable[Solution | None]]
NeighborhoodOperator = Callable[[Solution], Iterable[Solution]]


def noop(*_args):
    """Skip search phase, used for RVNS."""

    def search(initial: Solution) -> Iterable[Solution | None]:
        yield initial

    return search


def best_improvement(
    operator: NeighborhoodOperator, objective_index: int | None = None
) -> SearchFunction:
    """Go as far as possible, each time moving to the best near neighbor."""

    if objective_index is not None:
        is_better = lambda a, b: a[objective_index] < b[objective_index]  # noqa: E731
    else:
        is_better = (  # noqa: E731
            lambda a, b: is_dominating_min(a, b) == ComparisonResult.STRICTLY_BETTER
        )

    def search(initial: Solution) -> Iterable[Solution | None]:
        current = initial

        while True:
            improved = False
            best_found_in_neighborhood = current

            for neighbor in operator(current):
                if is_better(
                    neighbor.objectives, best_found_in_neighborhood.objectives
                ):
                    best_found_in_neighborhood = neighbor
                    improved = True

            if improved:
                current = best_found_in_neighborhood
            else:
                break

            yield None

        yield current

    return search


def first_improvement(
    operator: NeighborhoodOperator, objective_index: int | None = None
):
    """Go as far as possible, each time moving to the first better near neighbor."""

    if objective_index is not None:
        is_better = lambda a, b: a[objective_index] < b[objective_index]  # noqa: E731
    else:
        is_better = (  # noqa: E731
            lambda a, b: is_dominating_min(a, b) == ComparisonResult.STRICTLY_BETTER
        )

    def search(initial: Solution) -> Iterable[Solution | None]:
        current = initial

        while True:
            for neighbor in operator(current):
                if is_better(neighbor.objectives, current.objectives):
                    current = neighbor
                    break
            else:
                break

            yield None

        yield current

    return search


def first_improvement_quick(
    operator: NeighborhoodOperator, objective_index: int | None = None
):
    """Move to the first better near neighbor once."""

    if objective_index is not None:
        is_better = lambda a, b: a[objective_index] < b[objective_index]  # noqa: E731
    else:
        is_better = (  # noqa: E731
            lambda a, b: is_dominating_min(a, b) == ComparisonResult.STRICTLY_BETTER
        )

    def search(initial: Solution) -> Iterable[Solution | None]:
        for neighbor in operator(initial):
            if is_better(neighbor.objectives, initial.objectives):
                yield neighbor
                return

        yield initial

    return search


def composite(search_functions: list[SearchFunction]):
    def search(initial: Solution) -> Iterable[Solution | None]:
        current = initial

        vnd_level = 0
        while vnd_level < len(search_functions):
            search_generator = search_functions[vnd_level](current)
            local_optimum = None

            # The search generator yields None for intermediate steps,
            # and the final solution at the end.
            # Used to ensure termination on time,
            # when running optimization session
            # with large search neighborhoods.
            for solution in search_generator:
                if solution is None:
                    yield None
                    continue

                local_optimum = solution

            if (
                local_optimum
                and is_dominating_min(local_optimum.objectives, current.objectives)
                == ComparisonResult.STRICTLY_BETTER
            ):
                current = local_optimum
                vnd_level = 0
            else:
                vnd_level += 1

            yield None

        yield current

    return search
