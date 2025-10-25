import time
from typing import Callable, Iterable

from src.core.abstract import OptimizerAbstract


TerminationCriterion = Callable[[Iterable[bool | None]], Iterable[bool | None]]


def terminate_noop() -> TerminationCriterion:
    def optimize(optimization_process: Iterable[bool | None]):
        for iteration_result in optimization_process:
            yield iteration_result

    return optimize


def terminate_time_based(max_time_seconds: float) -> TerminationCriterion:
    def optimize(optimization_process: Iterable[bool | None]):
        if max_time_seconds <= 0:
            raise ValueError("Maximum time must be a positive integer.")

        start_time = time.time()
        for iteration_result in optimization_process:
            yield iteration_result

            if time.time() - start_time >= max_time_seconds:
                break

    return optimize


def terminate_max_iterations(max_iterations: int) -> TerminationCriterion:
    def optimize(optimization_process: Iterable[bool | None]):
        if max_iterations <= 0:
            raise ValueError("Maximum iterations must be a positive integer.")

        current_iterations = 0
        for iteration_result in optimization_process:
            yield iteration_result

            if current_iterations >= max_iterations:
                break

            if iteration_result is not None:
                current_iterations += 1

    return optimize


def terminate_max_no_improvement(max_stagnant_cycles: int) -> TerminationCriterion:
    def optimize(optimization_process: Iterable[bool | None]):
        if max_stagnant_cycles <= 0:
            raise ValueError("Maximum stagnant cycles must be a positive integer.")

        stagnant_cycles = 0
        for iteration_result in optimization_process:
            yield iteration_result

            if iteration_result is None:
                continue

            stagnant_cycles = 0 if iteration_result else stagnant_cycles + 1
            if stagnant_cycles >= max_stagnant_cycles:
                break

    return optimize


class StoppableOptimizer[T](OptimizerAbstract[T]):
    def __init__(
        self,
        optimizer: OptimizerAbstract[T],
        termination_criterion: TerminationCriterion = terminate_noop(),
    ):
        super().__init__(
            optimizer.name,
            optimizer.version,
            optimizer.problem,
            optimizer.acceptance_criterion,
        )

        self.termination_criterion = termination_criterion
        self.optimizer = optimizer

    def reset(self) -> None:
        self.acceptance_criterion.clear()

    def initialize(self) -> None:
        for sol in self.problem.get_initial_solutions():
            self.acceptance_criterion.accept(sol)

    def optimize(self) -> Iterable[bool | None]:
        return self.termination_criterion(self.optimizer.optimize())


class ChainedOptimizer[T](OptimizerAbstract[T]):
    def __init__(
        self,
        optimizers: list[StoppableOptimizer[T]],
        termination_criterion: TerminationCriterion = terminate_noop(),
    ):
        optimizer = optimizers[0]
        super().__init__(
            optimizer.name,
            optimizer.version,
            optimizer.problem,
            optimizer.acceptance_criterion,
        )

        self.termination_criterion = termination_criterion
        self.optimizers = optimizers

    def reset(self) -> None:
        self.acceptance_criterion.clear()

    def initialize(self) -> None:
        for sol in self.problem.get_initial_solutions():
            self.acceptance_criterion.accept(sol)

    def _optimize(self) -> Iterable[bool | None]:
        for optimizer in self.optimizers:
            optimizer.reset()
            optimizer.acceptance_criterion = self.acceptance_criterion
            yield from optimizer.optimize()

    def optimize(self) -> Iterable[bool | None]:
        return self.termination_criterion(self._optimize())
