import random
from typing import Any, Iterable

from src.problems.moacbw.problem import MOACBWSolution


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
            yield solution.new(new_data)


def swap_limited_op(solution: MOACBWSolution) -> Iterable[MOACBWSolution]:
    """Generates neighbors by swapping one selected item with one unselected item."""
    solution_data = solution.data

    index_order = shuffled(range(solution_data.size - 1))

    for i in index_order:
        new_data = solution_data.copy()
        new_data[i], new_data[i + 1] = new_data[i + 1], new_data[i]
        yield solution.new(new_data)


def shake_swap(solution: MOACBWSolution, k: int) -> MOACBWSolution:
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
