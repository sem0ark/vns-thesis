import logging
import time

from src.core.abstract import Solution
from src.core.termination import TerminationCriterion
from src.vns.acceptance import ParetoFront
from src.vns.optimizer import OptimizerAbstract

LOGGING_INTERVAL = 5.0


def run_vns_optimizer(
    optimizer: OptimizerAbstract, termination_criterion: TerminationCriterion
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
    optimizer.initialize()

    start_time = time.time()
    for _, improved in enumerate(termination_criterion(optimizer.optimize()), 1):
        current_time = time.time()

        if improved is not None:
            iteration_actual += 1
            improved_in_cycle = improved or improved_in_cycle

        if current_time >= last_log_time + LOGGING_INTERVAL:
            front = optimizer.get_solutions()
            last_log_time = current_time

            logger.info(
                "Iteration %d %s %s",
                iteration_actual,
                f"({len(front)} solutions in front)",
                improved_in_cycle and ": Improved!" or "",
            )
            improved_in_cycle = False

    non_dominant_front = ParetoFront()
    for sol in optimizer.get_solutions():
        non_dominant_front.accept(sol)

    num_solutions = len(non_dominant_front.solutions)
    logger.info(
        "Ran for %.2f seconds, %d iterations. Total # solutions: %d",
        time.time() - start_time,
        iteration_actual,
        num_solutions,
    )
    return non_dominant_front.solutions
