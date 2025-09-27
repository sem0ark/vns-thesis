import logging
from typing import Callable, Iterable, TypeVar

from src.vns.abstract import (
    AcceptanceCriterion,
    Problem,
    Solution,
    VNSOptimizerAbstract,
)

T = TypeVar("T")


class ElementwiseVNSOptimizer[T](VNSOptimizerAbstract[T]):
    def __init__(
        self,
        name: str,
        version: int,
        problem: Problem[T],
        search_functions: list[
            Callable[[Solution[T], VNSOptimizerAbstract], Iterable[Solution[T]]]
        ],
        shake_function: Callable[[Solution[T], int, VNSOptimizerAbstract], Solution[T]],
        acceptance_criterion: AcceptanceCriterion[T],
    ):
        super().__init__(name, version, problem, acceptance_criterion)

        self.search_functions = search_functions
        self.shake_function = shake_function

        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self) -> Iterable[bool]:
        """
        Runs the VNS optimization process.
        Returns the best solution found (for single-obj) or the Pareto front (for multi-obj).
        """
        self.acceptance_criterion.clear()
        for sol in self.problem.get_initial_solutions():
            self.acceptance_criterion.accept(sol)

        while True:
            current_solution = self.acceptance_criterion.get_one_current_solution()
            k = 0

            while k < len(self.search_functions):
                shaken_solution = self.shake_function(current_solution, k + 1, self)
                accepted = False
                for intensified_solution in self.search_functions[k](
                    shaken_solution, self
                ):
                    if not intensified_solution:
                        yield False
                        continue

                    local_accepted = self.acceptance_criterion.accept(
                        intensified_solution
                    )
                    accepted = accepted or local_accepted

                k = 0 if accepted else k + 1
                yield accepted
