import itertools
import logging
import random
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from src.cli.cli import CLI, Metadata, SavedRun, SavedSolution
from src.examples.mokp.problem import MOKPProblem, MOKPSolution
from src.examples.vns_runner_utils import run_vns_optimizer
from src.vns.abstract import VNSOptimizerAbstract
from src.vns.acceptance import AcceptBatch, AcceptBatchSkewed
from src.vns.local_search import (
    best_improvement,
    composite,
    first_improvement,
    first_improvement_quick,
    noop,
)
from src.vns.optimizer import ElementwiseVNSOptimizer

logger = logging.getLogger("mokp-solver")


def add_remove_op(
    solution: MOKPSolution, config: VNSOptimizerAbstract
) -> Iterable[MOKPSolution]:
    """Generates neighbors by adding or removing a single item."""
    solution_data = solution.data
    num_items = len(solution_data)

    for i in shuffled(range(num_items)):
        new_data = solution_data.copy()
        new_data[i] = 1 - new_data[i]
        yield solution.new(new_data)


def swap_op(
    solution: MOKPSolution, config: VNSOptimizerAbstract
) -> Iterable[MOKPSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    selected_items = np.where(solution_data == 1)[0]
    unselected_items = np.where(solution_data == 0)[0]

    for i in shuffled(selected_items):
        for j in shuffled(unselected_items):
            new_data = solution_data.copy()
            new_data[i] = 0
            new_data[j] = 1

            yield solution.new(new_data)


def shuffled(lst: Iterable) -> list[Any]:
    lst = list(lst)
    random.shuffle(lst)
    return lst


def shake_add_remove(
    solution: MOKPSolution, k: int, _config: VNSOptimizerAbstract
) -> MOKPSolution:
    """
    Randomly adds or removes 'k' items.
    """
    solution_data = solution.data.copy()

    for _ in range(k):
        is_add_operation = random.random() > 0.5

        if is_add_operation:
            unselected_items = np.where(solution_data == 0)[0]
            if unselected_items.size > 0:
                item_to_add = random.choice(unselected_items)
                solution_data[item_to_add] = 1
        else:
            selected_items = np.where(solution_data == 1)[0]
            if selected_items.size > 0:
                item_to_remove = random.choice(selected_items)
                solution_data[item_to_remove] = 0

    return solution.new(solution_data)


def shake_swap(
    solution: MOKPSolution, k: int, _config: VNSOptimizerAbstract
) -> MOKPSolution:
    """
    Randomly swaps a selected item with an unselected item 'k' times.
    """
    solution_data = solution.data.copy()

    for _ in range(k):
        selected_items = np.where(solution_data == 1)[0]
        unselected_items = np.where(solution_data == 0)[0]

        if selected_items.size > 0 and unselected_items.size > 0:
            item_to_swap_out = random.choice(selected_items)
            item_to_swap_in = random.choice(unselected_items)
            solution_data[item_to_swap_out] = 0
            solution_data[item_to_swap_in] = 1

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
            problem_name="mokp",
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
    alpha_weights = []
    num_objectives = 1
    problem: Any = None

    if instance_path:
        problem = MOKPProblem.load(instance_path)
        num_objectives = problem.num_objectives

        # Set alpha so that alpha * max-distance approximately one item from a set
        alpha_weights = np.mean(problem.profits, axis=1)

    # Ensure state is cleared in separate runs
    acceptance_criteria = [
        ("batch", AcceptBatch),
        (
            "skewed",
            partial(
                AcceptBatchSkewed,
                alpha_weights,
                MOKPProblem.calculate_solution_distance,
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
                [("op_ar", add_remove_op), ("op_swap", swap_op)],
            )
        ],
        *[
            (
                f"composite_{search_name}_ar_swap",
                composite(
                    [
                        composite(
                            [
                                search_func_factory(add_remove_op, obj_i)
                                for obj_i in range(num_objectives)
                            ]
                        ),
                        composite(
                            [
                                search_func_factory(swap_op, obj_i)
                                for obj_i in range(num_objectives)
                            ]
                        ),
                    ]
                ),
            )
            for (search_name, search_func_factory) in [
                ("BI", best_improvement),
                ("FI", first_improvement),
                ("QFI", first_improvement_quick),
            ]
        ],
    ]
    shake_functions = [
        ("shake_ar", shake_add_remove),
        ("shake_swap", shake_swap),
    ]

    for (
        (acc_name, make_acc),
        (search_name, search_func),
        (shake_name, shake_func),
        k,
    ) in itertools.product(
        acceptance_criteria, local_search_functions, shake_functions, range(1, 8)
    ):
        common_name = "BVNS"
        if "composite_" in search_name:
            common_name = "GVNS"
        elif "noop" in search_name:
            common_name = "RVNS"
        elif "skewed" in acc_name:
            common_name = "SVNS"

        config_name = f"{common_name} {acc_name} {search_name} k{k} {shake_name}"

        config = ElementwiseVNSOptimizer(
            problem=problem,
            search_functions=[search_func] * k,
            acceptance_criterion=make_acc(),
            shake_function=shake_func,
            name=config_name,
            version=10,
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
        "mokp",
        [
            (optimizer_name, make_runner(optimizer_name))
            for optimizer_name in prepare_optimizers(None).keys()
        ],
    )
