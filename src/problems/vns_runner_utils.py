import logging
import time

from src.core.abstract import Solution
from src.vns.optimizer import OptimizerAbstract


LOGGING_INTERVAL = 5.0


def run_vns_optimizer(
    run_time_seconds: float,
    optimizer: OptimizerAbstract,
) -> list[Solution]:
    """
    Runs the VNS optimizer for a specified duration and returns the final
    list of non-dominated solutions.

    Args:
        run_time_seconds: The maximum duration in seconds for the optimization run.
        optimizer: The VNSOptimizer instance to run.

    Returns:
        A list of Solution objects representing the final Pareto front.
    """
    logger = logging.getLogger(optimizer.name)
    last_log_time = time.time()

    improved_in_cycle = False
    iteration_actual = 1
    optimizer.reset()

    start_time = time.time()
    for _, improved in enumerate(optimizer.optimize(), 1):
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > run_time_seconds:
            num_solutions = len(optimizer.acceptance_criterion.get_all_solutions())
            logger.info(
                "Timeout after %d iterations, ran for %.2f seconds. Total # solutions: %d",
                iteration_actual,
                elapsed_time,
                num_solutions,
            )
            break

        if improved is not None:
            iteration_actual += 1
            improved_in_cycle = improved or improved_in_cycle

        if current_time >= last_log_time + LOGGING_INTERVAL:
            front = optimizer.acceptance_criterion.get_all_solutions()
            last_log_time = current_time

            logger.info(
                "Iteration %d %s %s",
                iteration_actual,
                f"({len(front)} solutions in front)" if front is not None else "",
                improved_in_cycle and ": Improved!" or "",
            )
            improved_in_cycle = False

    return optimizer.acceptance_criterion.get_all_solutions()
