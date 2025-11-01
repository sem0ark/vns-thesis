import random
from typing import Iterable

from src.problems.moacbw.problem import MOACBWSolution, _MOACBWSolution


def swap_op(solution: MOACBWSolution) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    for i in range(solution_data.size):
        for j in range(i + 1, solution_data.size):
            new_data = solution_data.copy()
            new_data[i], new_data[j] = new_data[j], new_data[i]
            yield _MOACBWSolution(new_data, solution.problem)


def swap_limited_op(solution: MOACBWSolution) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    for i in range(solution_data.size - 1):
        new_data = solution_data.copy()
        new_data[i], new_data[i + 1] = new_data[i + 1], new_data[i]
        yield _MOACBWSolution(new_data, solution.problem)


def shake_swap(solution: MOACBWSolution, k: int) -> MOACBWSolution:
    """
    Randomly swaps two vertcies 'k' times.
    """
    new_data = solution.data.copy()
    n = new_data.size

    for _ in range(k):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        new_data[i], new_data[j] = new_data[j], new_data[i]

    return _MOACBWSolution(new_data, solution.problem)
