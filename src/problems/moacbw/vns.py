import random
from typing import Any, Iterable

from src.problems.moacbw.problem import _MOACBWSolution, MOACBWSolution


def shuffled(lst: Iterable) -> list[Any]:
    lst = list(lst)
    random.shuffle(lst)
    return lst


def swap_op(solution: MOACBWSolution) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    index_order = shuffled(range(solution_data.size))

    for i in index_order:
        for j in index_order:
            if i >= j:
                continue

            new_data = solution_data.copy()
            new_data[i], new_data[j] = new_data[j], new_data[i]
            yield _MOACBWSolution(new_data, solution.problem)


def swap_limited_op(solution: MOACBWSolution) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    index_order = shuffled(range(solution_data.size - 1))

    for i in index_order:
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
