import random
from typing import Iterable

import numpy as np

from src.core.abstract import Delta
from src.problems.moscp.problem import MOSCPSolution, _MOSCPSolution


class FlipDelta(Delta[np.ndarray]):
    def __init__(self, flip_index) -> None:
        super().__init__()
        self.flip_index = flip_index

    def apply(self, data: np.ndarray):
        data[self.flip_index] = np.logical_not(data[self.flip_index])

    def revert(self, data: np.ndarray):
        data[self.flip_index] = np.logical_not(data[self.flip_index])


def flip_op(solution: MOSCPSolution) -> Iterable[MOSCPSolution]:
    for i in range(len(solution.data)):
        delta = FlipDelta(i)

        delta.apply(solution.data)
        objectives = solution.problem.calculate_objectives(solution.data)
        delta.revert(solution.data)

        yield _MOSCPSolution(solution.data, solution.problem, objectives, delta)


def shake_flip(solution: MOSCPSolution, k: int) -> MOSCPSolution:
    new_data = solution.get_data_copy()
    flip_indices = random.sample(range(len(new_data)), k)
    new_data[flip_indices] = np.logical_not(new_data[flip_indices])
    return _MOSCPSolution(
        new_data, solution.problem, solution.problem.calculate_objectives(new_data)
    )
