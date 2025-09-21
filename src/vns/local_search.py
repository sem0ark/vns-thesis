from ast import TypeVar
from typing import Callable
from src.vns.abstract import NeighborhoodOperator, Solution, VNSOptimizerAbstract
from src.vns.acceptance import ComparisonResult, compare

T = TypeVar("T")
SingleSearchFunction = Callable[[Solution[T], VNSOptimizerAbstract[T]], Solution[T]]


def noop(*_args):
    def search(initial: Solution, _config: VNSOptimizerAbstract) -> Solution:
        return initial

    return search


def best_improvement(operator: NeighborhoodOperator):
    """Go as far as possible, each time moving to the best near neighbor."""

    def search(initial: Solution, config: VNSOptimizerAbstract) -> Solution:
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

        return current

    return search


def first_improvement(operator: NeighborhoodOperator):
    """Go as far as possible, each time moving to the first better near neighbor."""

    def search(initial: Solution, config: VNSOptimizerAbstract) -> Solution:
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

        return current

    return search


def first_improvement_quick(operator: NeighborhoodOperator):
    """Move to the first better near neighbor once."""

    def search(initial: Solution, config: VNSOptimizerAbstract) -> Solution:
        current = initial

        for neighbor in operator(current, config):
            if (
                compare(neighbor.objectives, current.objectives)
                == ComparisonResult.STRICTLY_BETTER
            ):
                return neighbor

        return current

    return search


def composite(search_functions: list[SingleSearchFunction]):
    def search(initial: Solution, config: VNSOptimizerAbstract) -> Solution:
        current = initial

        vnd_level = 0
        while vnd_level < len(search_functions):
            new_solution_from_ls = search_functions[vnd_level](current, config)

            if (
                compare(new_solution_from_ls.objectives, current.objectives)
                == ComparisonResult.STRICTLY_BETTER
            ):
                vnd_level = 0
                current = new_solution_from_ls
            else:
                vnd_level += 1

        return current

    return search


# def _single_objective_search(
#     initial: Solution,
#     operator: NeighborhoodOperator,
#     config: VNSOptimizerAbstract,
#     objective_index: int,
# ) -> Solution:
#     """
#     Performs a local search that aims to improve a single objective `i`.
#     Returns the first improving neighbor found.
#     """
#     initial_objective = initial.objectives[objective_index]

#     for neighbor in operator(initial, config):
#         # A new solution is "better" if it has a better value for the
#         # specified objective, ignoring all others.
#         is_better_objective = (
#             neighbor.objectives[objective_index] > initial.objectives[objective_index]
#         )
#         if is_better_objective:
#             return neighbor
#     return initial


# def mo_vnd(operators: list[NeighborhoodOperator]):
#     """
#     Implements the MO-VND strategy (Algorithm 6) from the paper.

#     This search function is designed to be used with an acceptance criterion
#     that manages a set of solutions (the Pareto front). It takes a single
#     solution from this front and tries to improve it using objective-specific
#     local searches.
#     """

#     def search(initial: Solution, config: VNSOptimizerAbstract) -> Solution:
#         # Get the number of objectives from the solution's objectives property
#         num_objectives = len(initial.objectives)
#         current = initial

#         # The VND-i loop from the paper (Algorithm 5)
#         for objective_index in range(num_objectives):
#             # Try each neighborhood operator for the current objective
#             for operator in operators:
#                 new_solution = _single_objective_search(
#                     current, operator, config, objective_index
#                 )

#                 # Check if the new solution is not dominated by the current front
#                 # The acceptance criterion's 'accept' method handles this check
#                 # and updates the front if the solution is non-dominated.
#                 if config.acceptance_criterion.accept(new_solution):
#                     # If accepted, we have found an improvement. Return it.
#                     return new_solution

#         # If no non-dominated improving solution is found after all searches,
#         # return the initial solution.
#         return initial

#     return search
