import time
from typing import Callable, Iterable, Literal, Optional

from src.core.abstract import OptimizerAbstract, Solution

TerminationCriterion = Callable[[Iterable[Optional[bool]]], Iterable[Optional[bool]]]


def terminate_noop() -> TerminationCriterion:
    """
    Returns a termination criterion that never stops the optimization process.

    This serves as a default or an infinite run wrapper.

    Returns:
        TerminationCriterion: A function that passes the optimization stream through unchanged.
    """

    def optimize(optimization_process: Iterable[Optional[bool]]):
        for iteration_result in optimization_process:
            yield iteration_result

    return optimize


def terminate_time_based(max_time_seconds: float) -> TerminationCriterion:
    """
    Returns a termination criterion that stops the process after a specified duration.

    Args:
        max_time_seconds: The maximum wall clock time (in seconds) the process is allowed to run.

    Returns:
        TerminationCriterion: A function that wraps the optimization stream and stops it by time.

    Raises:
        ValueError: If max_time_seconds is not positive.
    """
    if max_time_seconds <= 0:
        raise ValueError("Maximum time must be a positive number of seconds.")

    def optimize(optimization_process: Iterable[Optional[bool]]):
        start_time = time.time()
        for iteration_result in optimization_process:
            yield iteration_result

            if time.time() - start_time >= max_time_seconds:
                break

    return optimize


def terminate_max_iterations(max_iterations: int) -> TerminationCriterion:
    """
    Returns a termination criterion that stops the process after a maximum number
    of main VNS cycles (iterations) have completed.

    Note: Only yields where the iteration result is not None count as a VNS cycle.

    Args:
        max_iterations: The maximum number of VNS main cycles allowed.

    Returns:
        TerminationCriterion: A function that wraps and stops the stream by iteration count.

    Raises:
        ValueError: If max_iterations is not positive.
    """
    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be a positive integer.")

    def optimize(optimization_process: Iterable[Optional[bool]]):
        current_iterations = 0
        for iteration_result in optimization_process:
            if current_iterations >= max_iterations:
                break

            yield iteration_result

            if iteration_result is not None:
                current_iterations += 1

    return optimize


def terminate_max_no_improvement(max_stagnant_cycles: int) -> TerminationCriterion:
    """
    Returns a termination criterion that stops the process if no new improvement
    (an accepted solution, indicated by iteration_result=True) is observed for a
    specified number of consecutive VNS cycles.

    Args:
        max_stagnant_cycles: The maximum number of consecutive cycles without improvement
                             before stopping (patience).

    Returns:
        TerminationCriterion: A function that wraps and stops the stream based on stagnation.

    Raises:
        ValueError: If max_stagnant_cycles is not positive.
    """
    if max_stagnant_cycles <= 0:
        raise ValueError("Maximum stagnant cycles must be a positive integer.")

    def optimize(optimization_process: Iterable[Optional[bool]]):
        stagnant_cycles = 0
        for iteration_result in optimization_process:
            yield iteration_result

            if iteration_result is None:
                continue

            stagnant_cycles = 0 if iteration_result else stagnant_cycles + 1
            if stagnant_cycles >= max_stagnant_cycles:
                break

    return optimize


def terminate_early_stop(
    metric_function: Callable[[], float],
    patience: int,
    min_delta: float = 1e-4,
    mode: Literal["min", "max"] = "min",
    skip_cycles=100,
) -> TerminationCriterion:
    """
    Args:
        patience: Number of cycles with no improvement before stopping.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        mode: One of {'min', 'max'}. In 'min' mode, the goal is to minimize the metric.
    """
    if patience <= 0:
        raise ValueError("Patience must be a positive integer.")

    delta_sign = 1.0 if mode == "min" else -1.0

    def optimize(optimization_process: Iterable[Optional[bool]]):
        stagnant_cycles = 0
        best_metric = float("inf") if mode == "min" else -float("inf")
        count = 0

        for iteration_result in optimization_process:
            yield iteration_result

            if iteration_result is None or count % skip_cycles != 0:
                continue

            count += 1

            current_metric = metric_function() + delta_sign * min_delta
            current_metric_padded = current_metric + delta_sign * min_delta
            is_improved = (
                current_metric_padded < best_metric
                if mode == "min"
                else current_metric_padded > best_metric
            )

            if is_improved:
                best_metric = current_metric
                stagnant_cycles = 0
            else:
                stagnant_cycles += 1

            if stagnant_cycles >= patience:
                break

    return optimize


class StoppableOptimizer[T](OptimizerAbstract[T]):
    """
    A wrapper class that embeds a termination criterion into an existing optimizer.

    It delegates the core optimization to the inner optimizer and uses the
    termination criterion to wrap and control the resulting generator stream.

    The reset and initialize logic is delegated to the inner optimizer.
    """

    def __init__(
        self,
        optimizer: OptimizerAbstract[T],
        termination_criterion: TerminationCriterion = terminate_noop(),
    ):
        super().__init__(
            optimizer.name,
            optimizer.version,
            optimizer.problem,
        )

        self.termination_criterion = termination_criterion
        self.optimizer = optimizer

    def reset(self) -> None:
        """Resets the acceptance criterion and the inner optimizer."""
        self.optimizer.reset()

    def initialize(self) -> None:
        """Initializes the acceptance criterion with the problem's starting solutions."""
        self.optimizer.initialize()

    def get_solutions(self) -> list[Solution]:
        return self.optimizer.get_solutions()

    def add_solutions(self, solutions: list[Solution]) -> None:
        return self.optimizer.add_solutions(solutions)

    def optimize(self):
        """
        Wraps the inner optimizer's stream with the termination criterion.

        Returns:
            Generator[Optional[bool]]: The terminated optimization stream.
        """
        return self.termination_criterion(self.optimizer.optimize())


class ChainedOptimizer[T](OptimizerAbstract[T]):
    """
    An optimizer that runs a sequence of Optimizers one after the other.

    It ensures the acceptance criterion (solution set) is passed from the termination
    point of one optimizer to the start of the next. The overall run terminates
    based on the outer termination_criterion wrapping the entire sequence.
    """

    def __init__(
        self,
        optimizers: list[OptimizerAbstract[T]],
    ):
        if not optimizers:
            raise ValueError("ChainedOptimizer requires at least one optimizer.")

        optimizer = optimizers[0]
        super().__init__(optimizer.name, optimizer.version, optimizer.problem)

        self.optimizers = optimizers
        self.current_optimizer = optimizer

    def reset(self) -> None:
        """Resets the shared acceptance criterion."""
        for optimizer in self.optimizers:
            optimizer.reset()

    def initialize(self) -> None:
        """Initializes the shared acceptance criterion with the problem's starting solutions."""
        for optimizer in self.optimizers:
            optimizer.initialize()

    def get_solutions(self) -> list[Solution]:
        return self.current_optimizer.get_solutions()

    def add_solutions(self, solutions: list[Solution]) -> None:
        return self.current_optimizer.add_solutions(solutions)

    def optimize(self) -> Iterable[Optional[bool]]:
        for optimizer in self.optimizers:
            optimizer.add_solutions(self.current_optimizer.get_solutions())
            self.current_optimizer = optimizer
            yield from optimizer.optimize()
