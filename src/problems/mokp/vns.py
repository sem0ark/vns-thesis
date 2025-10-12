import random
from typing import Iterable, cast

import numpy as np

from src.core.abstract import Delta
from src.problems.mokp.problem import MOKPProblem, MOKPSolution, _MOKPSolution


class FlipDelta(Delta[np.ndarray]):
    def __init__(self, flip_index) -> None:
        super().__init__()
        self.flip_index = flip_index

    def apply(self, data: np.ndarray):
        data[self.flip_index] = 1 - data[self.flip_index]

    def revert(self, data: np.ndarray):
        data[self.flip_index] = 1 - data[self.flip_index]


class SwapDelta(Delta[np.ndarray]):
    def __init__(self, from_index, to_index) -> None:
        super().__init__()
        self.from_index = from_index
        self.to_index = to_index

    def apply(self, data: np.ndarray):
        data[self.from_index], data[self.to_index] = (
            data[self.to_index],
            data[self.from_index],
        )

    def revert(self, data: np.ndarray):
        data[self.from_index], data[self.to_index] = (
            data[self.to_index],
            data[self.from_index],
        )


def flip_op(solution: MOKPSolution) -> Iterable[MOKPSolution]:
    num_items = len(solution.data)

    problem = cast(MOKPProblem, solution.problem)
    total_profits = np.abs(np.array(solution.objectives))
    total_weights = np.sum(solution.data * problem.weights, axis=1)

    for i in range(num_items):
        will_add_item = solution.data[i] == 0

        if will_add_item:
            new_weight = total_weights + problem.weights[:, i]
        else:
            new_weight = total_weights - problem.weights[:, i]

        if not np.all(new_weight <= problem.capacity, axis=0):
            continue

        if will_add_item:
            new_objectives = total_profits + problem.profits[:, i]
        else:
            new_objectives = total_profits - problem.profits[:, i]

        yield _MOKPSolution(
            solution.data,
            problem,
            objectives=tuple((-new_objectives).tolist()),
            delta=FlipDelta(i),
        )


def swap_op(solution: MOKPSolution) -> Iterable[MOKPSolution]:
    """
    Generates neighbors by swapping one selected item with one unselected item,
    using incremental calculation for fast objective and feasibility checks.
    """
    problem = cast(MOKPProblem, solution.problem)
    current_profits = np.abs(np.array(solution.objectives))
    total_weights = np.sum(solution.data * problem.weights, axis=1)

    selected_items = np.where(solution.data == 1)[0]
    unselected_items = np.where(solution.data == 0)[0]

    for i in selected_items:
        for j in unselected_items:
            weight_change = problem.weights[:, j] - problem.weights[:, i]
            new_weight = total_weights + weight_change

            if not np.all(new_weight <= problem.capacity, axis=0):
                continue

            profit_change = problem.profits[:, j] - problem.profits[:, i]
            new_profits = current_profits + profit_change
            new_objectives = tuple((-new_profits).tolist())

            yield _MOKPSolution(
                solution.data,
                problem,
                new_objectives,
                SwapDelta(i, j),
            )


def shake_flip(solution: MOKPSolution, k: int) -> MOKPSolution:
    """
    Randomly adds or removes 'k' items.
    """
    new_data = solution.get_data_copy()
    flip_indices = random.sample(range(len(new_data)), k)
    new_data[flip_indices] = 1 - new_data[flip_indices]
    return _MOKPSolution(
        new_data, solution.problem, solution.problem.calculate_objectives(new_data)
    )


def shake_swap(solution: MOKPSolution, k: int) -> MOKPSolution:
    """
    Randomly swaps a selected item with an unselected item 'k' times.
    """
    new_data = solution.get_data_copy()

    selected_items = np.where(new_data == 1)[0]
    unselected_items = np.where(new_data == 0)[0]

    item_to_swap_out = random.sample(selected_items.tolist(), k)
    new_data[item_to_swap_out] = 0

    item_to_swap_in = random.sample(unselected_items.tolist(), k)
    new_data[item_to_swap_in] = 1

    return _MOKPSolution(
        new_data, solution.problem, solution.problem.calculate_objectives(new_data)
    )
