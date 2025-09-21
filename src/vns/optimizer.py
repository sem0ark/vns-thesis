import logging
from typing import Callable, Iterable, TypeVar

from src.vns.abstract import (
    AcceptanceCriterion,
    Problem,
    VNSOptimizerAbstract,
    Solution,
)


T = TypeVar("T")


class ElementwiseVNSOptimizer[T](VNSOptimizerAbstract[T]):
    def __init__(
        self,
        name: str,
        version: int,
        problem: Problem[T],
        search_functions: list[
            Callable[[Solution[T], VNSOptimizerAbstract], Solution[T]]
        ],
        shake_function: Callable[[Solution[T], int, VNSOptimizerAbstract], Solution[T]],
        acceptance_criterion: AcceptanceCriterion[T],
    ):
        super().__init__(name, version, problem)

        self.search_functions = search_functions
        self.shake_function = shake_function
        self.acceptance_criterion = acceptance_criterion

        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self) -> Iterable[bool]:
        """
        Runs the VNS optimization process.
        Returns the best solution found (for single-obj) or the Pareto front (for multi-obj).
        """
        for sol in self.problem.get_initial_solutions():
            self.acceptance_criterion.accept(sol)

        while True:
            improved = False
            current_solution = self.acceptance_criterion.get_one_current_solution()

            for k, search_function in enumerate(self.search_functions, 1):
                shaken_solution = self.shake_function(current_solution, k, self)
                local_optimum = search_function(shaken_solution, self)

                accepted = self.acceptance_criterion.accept(local_optimum)

                if accepted:
                    improved = True
                    break

            yield improved

    def get_solutions(self) -> Iterable[Solution]:
        return self.acceptance_criterion.get_all_solutions()


class FrontwiseVNSOptimizer[T](VNSOptimizerAbstract[T]):
    def __init__(
        self,
        name: str,
        version: int,
        problem: Problem[T],
        search_functions: list[
            Callable[
                [Iterable[Solution[T]], VNSOptimizerAbstract], Iterable[Solution[T]]
            ]
        ],
        shake_function: Callable[[Solution[T], int, VNSOptimizerAbstract], Solution[T]],
        acceptance_criterion: AcceptanceCriterion[T],
    ):
        super().__init__(name, version, problem)

        self.search_functions = search_functions
        self.shake_function = shake_function
        self.acceptance_criterion = acceptance_criterion

        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self) -> Iterable[bool]:
        """
        Runs the VNS optimization process.
        Returns the best solution found (for single-obj) or the Pareto front (for multi-obj).
        """
        for sol in self.problem.get_initial_solutions():
            self.acceptance_criterion.accept(sol)

        while True:
            improved = False
            current_solutions = self.acceptance_criterion.get_all_solutions()

            for k, search_function in enumerate(self.search_functions, 1):
                shaken_solutions = [
                    self.shake_function(sol, k, self) for sol in current_solutions
                ]
                local_optimums = search_function(shaken_solutions, self)

                accepted = False
                for local_optimum in local_optimums:
                    accepted = accepted or self.acceptance_criterion.accept(
                        local_optimum
                    )

                if accepted:
                    improved = True
                    break

            yield improved

    def get_solutions(self) -> Iterable[Solution]:
        return self.acceptance_criterion.get_all_solutions()
