import itertools
import logging
from pathlib import Path
import random
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np

from src.examples.mokp.mokp_problem import MOKPProblem, MOKPSolution
from src.vns.abstract import VNSConfig
from src.vns.acceptance import AcceptBatchBigger, AcceptBatchSkewedBigger
from src.vns.local_search import best_improvement, first_improvement, noop
from src.vns.optimizer import VNSOptimizer

from src.examples.vns_runner_utils import run_vns_optimizer
from src.cli.cli import CLI, SavedRun, Metadata, SavedSolution

logger = logging.getLogger("mokp-solver")


def add_remove_op(solution: MOKPSolution, config: VNSConfig) -> Iterable[MOKPSolution]:
    """Generates neighbors by adding or removing a single item."""
    solution_data = solution.data
    num_items = len(solution_data)

    for i in shuffled(range(num_items)):
        new_data = solution_data.copy()
        new_data[i] = 1 - new_data[i]
        yield solution.new(new_data)


def swap_op(solution: MOKPSolution, config: VNSConfig) -> Iterable[MOKPSolution]:
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
    solution: MOKPSolution, k: int, _config: VNSConfig
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


def shake_swap(solution: MOKPSolution, k: int, _config: VNSConfig) -> MOKPSolution:
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
    optimizer_config: VNSConfig,
) -> SavedRun:
    solutions = run_vns_optimizer(
        run_time_seconds,
        VNSOptimizer(optimizer_config),
    )

    return SavedRun(
        metadata=Metadata(
            run_time_seconds=int(run_time_seconds),
            name=optimizer_config.name,
            version=optimizer_config.version,
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
    problem: Any = MOKPProblem.load(instance_path) if instance_path else None

    optimizers: dict[str, Callable[[float], SavedRun]] = {}

    acceptance_criteria = [
        ("BVNS", AcceptBatchBigger()),
        ("SVNS", AcceptBatchSkewedBigger(1, MOKPProblem.calculate_solution_distance)),
    ]
    local_search_approaches = [
        ("BI", best_improvement),
        ("FI", first_improvement),
        ("noop", noop),
    ]
    neighborhood_operations = [
        ("op_ar", add_remove_op),
        ("op_swap", swap_op),
    ]
    shake_functions = [
        ("shake_ar", shake_add_remove),
        ("shake_swap", shake_swap),
    ]

    for (
        (acc_name, acc_func),
        (search_name, search_func_factory),
        (op_name, op_func),
        (shake_name, shake_func),
    ) in itertools.product(
        acceptance_criteria,
        local_search_approaches,
        neighborhood_operations,
        shake_functions,
    ):
        if search_name == "noop":
            acc_name = "RVNS"

            if op_name != "add_remove":
                continue

        search_func_instance = (
            search_func_factory(op_func) if search_name != "noop" else noop()
        )

        for k in range(1, 8):
            config_name = f"{acc_name} {search_name} {op_name} k{k} {shake_name}"

            config = VNSConfig(
                problem=problem,
                search_functions=[search_func_instance] * k,
                acceptance_criterion=acc_func,
                shake_function=shake_func,
                name=config_name,
                version=2,
            )

            def runner_func(run_time, _config=config):
                return run_instance_with_config(run_time, str(instance_path), _config)

            optimizers[config_name] = runner_func

    return optimizers


def register_cli(cli: CLI) -> None:
    def make_runner(optimizer_name: str):
        def run(instance_path, run_time, _optimizer_name=optimizer_name):
            return prepare_optimizers(instance_path)[_optimizer_name](run_time)

        return run

    cli.register_runner(
        "mokp",
        [
            (optimizer_name, make_runner(optimizer_name))
            for optimizer_name in prepare_optimizers(None).keys()
        ],
    )
