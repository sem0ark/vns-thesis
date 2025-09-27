import itertools
import logging
import random
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Iterable

from src.cli.cli import CLI, Metadata, SavedRun, SavedSolution
from src.examples.moacbw.problem import MOACBWProblem, MOACBWSolution
from src.examples.vns_runner_utils import run_vns_optimizer
from src.vns.abstract import VNSOptimizerAbstract
from src.vns.acceptance import AcceptBatch, AcceptBatchSkewed
from src.vns.local_search import (
    best_improvement,
    first_improvement,
    first_improvement_quick,
    noop,
)
from src.vns.optimizer import ElementwiseVNSOptimizer

logger = logging.getLogger("moacbw-solver")


def shuffled(lst: Iterable) -> list[Any]:
    lst = list(lst)
    random.shuffle(lst)
    return lst


def swap_op(
    solution: MOACBWSolution, config: VNSOptimizerAbstract
) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    index_order = shuffled(range(solution_data.size))

    for i in index_order:
        for j in index_order:
            if i >= j:
                continue

            new_data = solution_data.copy()
            new_data[i], new_data[j] = new_data[j], new_data[i]
            yield solution.new(new_data)


def swap_limited_op(
    solution: MOACBWSolution, config: VNSOptimizerAbstract
) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    index_order = shuffled(range(solution_data.size - 1))

    for i in index_order:
        new_data = solution_data.copy()
        new_data[i], new_data[i + 1] = new_data[i + 1], new_data[i]
        yield solution.new(new_data)


def shake_swap(
    solution: MOACBWSolution, k: int, _config: VNSOptimizerAbstract
) -> MOACBWSolution:
    """
    Randomly swaps two vertcies 'k' times.
    """
    solution_data = solution.data.copy()
    n = solution_data.size

    for _ in range(k):
        i = random.randint(0, n - 1)
        j = (n + random.randint(1, k) * (random.randint(0, 1) * 2 - 1)) % n

        solution_data[i], solution_data[j] = solution_data[j], solution_data[i]

    return solution.new(solution_data)


def shake_swap_limited(
    solution: MOACBWSolution, k: int, _config: VNSOptimizerAbstract
) -> MOACBWSolution:
    """
    Randomly swaps two vertcies 'k' times limited to a range of 'k'.
    """
    solution_data = solution.data.copy()
    n = solution_data.size

    for _ in range(k):
        i = random.randint(0, n - 1)
        offset = k * (random.randint(0, 1) * 2 - 1)
        j = i + offset

        if not (0 <= j < n):
            j += -2 * offset

        solution_data[i], solution_data[j] = solution_data[j], solution_data[i]

    return solution.new(solution_data)


# ----------------------------------------------------------------------------------


def run_instance_with_config(
    run_time_seconds: float,
    instance_path: str,
    optimizer: VNSOptimizerAbstract,
) -> SavedRun:
    solutions = run_vns_optimizer(
        run_time_seconds,
        optimizer,
    )

    return SavedRun(
        metadata=Metadata(
            run_time_seconds=int(run_time_seconds),
            name=optimizer.name,
            version=optimizer.version,
            problem_name="moacbw",
            instance_name=Path(instance_path).stem,
        ),
        solutions=[
            SavedSolution(sol.objectives, sol.to_json_serializable())
            for sol in solutions
        ],
    )


@lru_cache
def prepare_optimizers(
    instance_path: str | None,
) -> dict[str, Callable[[float], SavedRun]]:
    """
    Automatically generates all possible optimizer configurations using itertools.product.
    """

    optimizers: dict[str, Callable[[float], SavedRun]] = {}
    problem: Any = None

    if instance_path:
        problem = MOACBWProblem.load(instance_path)

    acceptance_criteria = [
        ("batch", AcceptBatch),
        (
            "skewed",
            partial(
                AcceptBatchSkewed, [5.0, 5.0], MOACBWProblem.calculate_solution_distance
            ),
        ),
    ]
    local_search_functions = [
        ("noop", noop()),
        *[
            (f"{search_name}_{op_name}", search_func_factory(op_func))
            for (
                (search_name, search_func_factory),
                (op_name, op_func),
            ) in itertools.product(
                [
                    ("BI", best_improvement),
                    ("FI", first_improvement),
                    ("QFI", first_improvement_quick),
                ],
                [("op_swap", swap_op), ("op_short_swap", swap_limited_op)],
            )
        ],
    ]
    shake_functions = [
        ("shake_swap", shake_swap),
        ("shake_swap_limited", shake_swap_limited),
    ]

    for (
        (acc_name, make_acc_func),
        (search_name, search_func),
        (shake_name, shake_func),
        k,
    ) in itertools.product(
        acceptance_criteria, local_search_functions, shake_functions, range(1, 8)
    ):
        config_name = f"vns {acc_name} {search_name} k{k} {shake_name}"

        config = ElementwiseVNSOptimizer(
            problem=problem,
            search_functions=[search_func] * k,
            acceptance_criterion=make_acc_func(),
            shake_function=shake_func,
            name=config_name,
            version=3,
        )

        def runner_func(run_time, _config=config):
            return run_instance_with_config(run_time, str(instance_path), _config)

        optimizers[config_name] = runner_func

    return optimizers


def register_cli(cli: CLI) -> None:
    def make_runner(optimizer_name: str):
        def run(instance_path, run_time, _optimizer_name=optimizer_name):
            return {
                **prepare_optimizers(instance_path),
            }[_optimizer_name](run_time)

        return run

    cli.register_runner(
        "moacbw",
        [
            (optimizer_name, make_runner(optimizer_name))
            for optimizer_name in prepare_optimizers(None).keys()
        ],
    )
