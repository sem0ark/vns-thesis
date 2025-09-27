from ast import TypeVar
from typing import Callable, Iterable

from src.vns.abstract import (NeighborhoodOperator, Solution,
                              VNSOptimizerAbstract)
from src.vns.acceptance import AcceptBeam, ComparisonResult, compare

T = TypeVar("T")
SingleSearchFunction = Callable[[Solution[T], VNSOptimizerAbstract[T]], Iterable[Solution[T]]]


def noop(*_args):
    def search(initial: Solution, _config: VNSOptimizerAbstract) -> Iterable[Solution]:
        yield initial

    return search


def best_improvement(operator: NeighborhoodOperator):
    """Go as far as possible, each time moving to the best near neighbor."""

    def search(initial: Solution, config: VNSOptimizerAbstract) -> Iterable[Solution]:
        current = initial

        while True:
            best_found_in_neighborhood = current

            for neighbor in operator(current, config):
                if (
                    compare(neighbor.objectives, best_found_in_neighborhood.objectives)
                    == ComparisonResult.STRICTLY_BETTER
                ):
                    best_found_in_neighborhood = neighbor

            if (
                compare(current.objectives, best_found_in_neighborhood.objectives)
                == ComparisonResult.STRICTLY_BETTER
            ):
                current = best_found_in_neighborhood
            else:
                break

        yield current

    return search


def first_improvement(operator: NeighborhoodOperator):
    """Go as far as possible, each time moving to the first better near neighbor."""

    def search(initial: Solution, config: VNSOptimizerAbstract) -> Iterable[Solution]:
        current = initial

        while True:
            improvement_found = False

            for neighbor in operator(current, config):
                if (
                    compare(neighbor.objectives, current.objectives)
                    == ComparisonResult.STRICTLY_BETTER
                ):
                    current = neighbor
                    improvement_found = True
                    break

            if not improvement_found:
                break

        yield current

    return search


def first_improvement_quick(operator: NeighborhoodOperator):
    """Move to the first better near neighbor once."""

    def search(initial: Solution, config: VNSOptimizerAbstract) -> Iterable[Solution]:
        current = initial

        for neighbor in operator(current, config):
            if (
                compare(neighbor.objectives, current.objectives)
                == ComparisonResult.STRICTLY_BETTER
            ):
                return neighbor

        yield current

    return search


def composite(search_functions: list[SingleSearchFunction]):
    def search(initial: Solution, config: VNSOptimizerAbstract) -> Iterable[Solution]:
        current = initial

        vnd_level = 0
        while vnd_level < len(search_functions):
            improved = False
            local_best_solution = current
            for local_solution in search_functions[vnd_level](current, config):
                if compare(local_solution.objectives, local_best_solution.objectives) == ComparisonResult.STRICTLY_BETTER:
                    improved = True
                    local_best_solution = local_solution
                    yield local_best_solution

            if improved:
                vnd_level = 0
                current = local_best_solution
                yield local_best_solution
            else:
                vnd_level += 1

        yield current

    return search


def vnd_i(
    initial: Solution,
    operators: list[NeighborhoodOperator],
    objective_index: int,
    optimizer: VNSOptimizerAbstract,
) -> set[Solution]:
    """
    Implements the VND-i local search procedure (Algorithm 5).

    This function performs a single-objective local search and returns the set
    of efficient points found.
    """
    k = 0
    k_max = len(operators)
    x = initial
    acceptance = AcceptBeam()
    acceptance.accept(x)

    while k < k_max:
        # Step 5: Find the best neighbor x' with respect to objective z_i
        x_prime = x
        for neighbor in operators[k](initial, optimizer):
            if (
                neighbor.objectives[objective_index]
                < x_prime.objectives[objective_index]
            ):
                x_prime = neighbor

        # Step 6: Update the set of efficient points E with x'
        acceptance.accept(x_prime)

        # Step 7-12: Check for improvement and update k
        # Improvement means x' is better than x for objective i
        if x_prime.objectives[objective_index] < x.objectives[objective_index]:
            x = x_prime
            k = 0
        else:
            k += 1

    return set(acceptance.get_all_solutions())


def mo_vnd(operators: list[NeighborhoodOperator]):
    """
    Implements the MO-VND strategy (Algorithm 6).

    This search function manages the overall multi-objective VND process,
    alternating between single-objective VND-i procedures.
    """

    def search(
        initial_solutions: Iterable[Solution], config: VNSOptimizerAbstract
    ) -> Iterable[Solution]:
        r = len(list(initial_solutions)[0].objectives)

        i = 0
        acceptance = AcceptBeam()
        for sol in config.acceptance_criterion.get_all_solutions():
            acceptance.accept(sol)

        exploited_sets: list[set[Solution]] = [set() for _ in range(r)]

        while i < r:
            accepted = False

            for x_prime in list(acceptance.get_all_solutions()):
                if x_prime in exploited_sets[i]:
                    continue

                exploited_sets[i].add(x_prime)
                Ei = vnd_i(x_prime, operators, i, config)
                for sol in Ei:
                    accepted = accepted or acceptance.accept(sol)
                    exploited_sets[i].add(sol)

            if accepted:
                i = 0
            else:
                i += 1

        return acceptance.get_all_solutions()

    return search
