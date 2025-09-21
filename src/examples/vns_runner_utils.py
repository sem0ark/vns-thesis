import logging
import time

from src.vns.abstract import Solution
from src.vns.optimizer import VNSOptimizerAbstract


def run_vns_optimizer(
    run_time_seconds: float,
    optimizer: VNSOptimizerAbstract,
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
    start_time = time.time()

    for iteration, improved in enumerate(optimizer.optimize(), 1):
        elapsed_time = time.time() - start_time

        if elapsed_time > run_time_seconds:
            num_solutions = len(optimizer.get_solutions())
            logger.info(
                "Timeout after %d iterations, ran for %d seconds. Total # solutions: %d",
                iteration,
                elapsed_time,
                num_solutions,
            )
            break

        if improved:
            logger.info("Iteration %d: Improved!", iteration)

    return optimizer.get_solutions()
