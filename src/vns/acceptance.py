import logging
import random
from collections import deque
from typing import Callable

from src.vns.abstract import AcceptanceCriterion, Solution


def dominates_minimize(
    new_objective: tuple[float, ...],
    current_objective: tuple[float, ...],
    buffer_value: float,
) -> bool:
    """
    Checks if objective vector new_objective dominates objective vector current_objective.
    Assumes minimization for all objectives.
    """
    if len(new_objective) != len(current_objective):
        raise ValueError("Objective vectors must have the same number of objectives.")

    at_least_one_strictly_better = False
    for i in range(len(new_objective)):
        if abs(new_objective[i] - current_objective[i]) < 1e-6:
            continue

        if new_objective[i] > current_objective[i]:
            buffer_value -= new_objective[i] - current_objective[i]
            if buffer_value > 0:
                continue
            return False
        elif new_objective[i] < current_objective[i]:
            at_least_one_strictly_better = True

    return at_least_one_strictly_better


def dominates_maximize(
    new_objective: tuple[float, ...],
    current_objective: tuple[float, ...],
    buffer_value: float,
) -> bool:
    """
    Checks if objective vector new_objective dominates objective vector current_objective.
    Assumes maximization for all objectives.
    """
    if len(new_objective) != len(current_objective):
        raise ValueError("Objective vectors must have the same number of objectives.")

    at_least_one_strictly_better = False
    for i in range(len(new_objective)):
        if abs(new_objective[i] - current_objective[i]) < 1e-6:
            continue

        if new_objective[i] < current_objective[i]:
            buffer_value -= current_objective[i] - new_objective[i]
            if buffer_value > 0:
                continue
            return False
        elif new_objective[i] > current_objective[i]:
            at_least_one_strictly_better = True

    return at_least_one_strictly_better


class _Comparison:
    def dominates(self, new_solution: Solution, current_solution: Solution) -> bool:
        raise NotImplementedError

    def dominates_buffered(
        self, new_solution: Solution, current_solution: Solution, buffer_value: float
    ) -> bool:
        raise NotImplementedError


class _Smaller(_Comparison):
    def dominates(self, new_solution: Solution, current_solution: Solution) -> bool:
        return dominates_minimize(
            new_solution.objectives, current_solution.objectives, 0.0
        )

    def dominates_buffered(
        self, new_solution: Solution, current_solution: Solution, buffer_value: float
    ) -> bool:
        return dominates_minimize(
            new_solution.objectives, current_solution.objectives, buffer_value
        )


class _Bigger(_Comparison):
    def dominates(self, new_solution: Solution, current_solution: Solution) -> bool:
        return dominates_maximize(
            new_solution.objectives, current_solution.objectives, 0.0
        )

    def dominates_buffered(
        self, new_solution: Solution, current_solution: Solution, buffer_value: float
    ) -> bool:
        return dominates_maximize(
            new_solution.objectives, current_solution.objectives, buffer_value
        )


class _AcceptBeam(_Comparison):
    def __init__(self):
        super().__init__()

        self.front: list[Solution] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def accept(self, candidate: Solution) -> bool:
        """
        Decides whether to accept candidate and update the archive.
        Updates the non-dominated archive based on Pareto dominance.
        """

        new_front = []

        for solution in self.front:
            if self.dominates(solution, candidate) or solution == candidate:
                return False

            if not self.dominates(candidate, solution):
                new_front.append(solution)

        new_front.append(candidate)
        self.front = new_front

        return True

    def get_all_solutions(self) -> list[Solution]:
        return self.front

    def get_one_current_solution(self) -> Solution:
        return random.choice(self.front)

    def clear(self):
        self.front = []


class _AcceptBeamSkewed(_AcceptBeam):
    def __init__(
        self,
        alpha: float,
        distance_metric: Callable[[Solution, Solution], float],
        buffer_size: int,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.alpha = alpha
        self.distance_metric = distance_metric

        self.buffer: deque[Solution] = deque(maxlen=buffer_size)

    def accept(self, candidate: Solution) -> bool:
        if super().accept(candidate):
            return True

        if all(
            solution is not candidate
            and self.dominates_buffered(
                candidate,
                solution,
                self.alpha * self.distance_metric(candidate, solution),
            )
            for solution in self.front
        ):
            self.buffer.append(candidate)

        return False

    def get_one_current_solution(self) -> Solution:
        size = len(self.front) + len(self.buffer)
        if size == 0:
            raise ValueError("No solutions")

        index = random.randint(0, size - 1)
        if index < len(self.front):
            return self.front[index]

        index -= len(self.front)
        return self.buffer[index]


class _AcceptBatch(_Comparison):
    def __init__(self):
        super().__init__()

        self.selection_counter = 0
        self.front: list[Solution] = []
        self.is_dominated: list[bool] = []

        self.upcoming_front: list[Solution] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def accept(self, candidate: Solution) -> bool:
        """
        Decides whether to accept candidate and update the archive.
        Updates the non-dominated archive based on Pareto dominance.
        """

        for i, solution in enumerate(self.front):
            if self.dominates(solution, candidate) or solution == candidate:
                return False

            if not self.is_dominated[i] and self.dominates(candidate, solution):
                self.is_dominated[i] = True

        if any(solution == candidate for solution in self.upcoming_front):
            return False

        self.upcoming_front.append(candidate)
        return True

    def get_all_solutions(self) -> list[Solution]:
        return self.front

    def get_one_current_solution(self) -> Solution:
        if self.selection_counter >= len(self.front):
            self._swap_fronts()
            self.selection_counter = 0

        current = self.front[self.selection_counter]
        self.selection_counter += 1

        return current

    def clear(self):
        self.front = []
        self.new_front = []

    def _swap_fronts(self):
        for solution, dominated in zip(self.front, self.is_dominated):
            if dominated:
                continue

            self.upcoming_front.append(solution)

        self.front = self.upcoming_front
        self.is_dominated = [False] * len(self.upcoming_front)
        self.upcoming_front = []


class _AcceptBatchSkewed(_AcceptBatch):
    def __init__(
        self,
        alpha: float,
        distance_metric: Callable[[Solution, Solution], float],
    ):
        super().__init__()

        self.alpha = alpha
        self.distance_metric = distance_metric

        self.skewed_front: list[Solution] = []
        self.upcoming_skewed_front: list[Solution] = []

        self.logger = logging.getLogger(self.__class__.__name__)

    def accept(self, candidate: Solution) -> bool:
        if super().accept(candidate):
            return True

        if all(
            solution is not candidate
            and self.dominates_buffered(
                candidate,
                solution,
                self.alpha * self.distance_metric(candidate, solution),
            )
            for solution in self.front
        ):
            self.upcoming_skewed_front.append(candidate)

        return False

    def get_one_current_solution(self) -> Solution:
        total_solutions = len(self.front) + len(self.skewed_front)

        if self.selection_counter >= total_solutions:
            self._swap_fronts()
            self.selection_counter = 0

        if self.selection_counter < len(self.front):
            current = self.front[self.selection_counter]
        elif self.selection_counter - len(self.front) < len(self.skewed_front):
            current = self.skewed_front[self.selection_counter - len(self.front)]
        else:
            raise RuntimeError("Unreachable")

        self.selection_counter += 1
        return current

    def _swap_fronts(self):
        super()._swap_fronts()

        self.skewed_front = self.upcoming_skewed_front
        self.upcoming_skewed_front = []


class AcceptBatchSmaller(_Smaller, _AcceptBatch, AcceptanceCriterion):
    """
    Minimization acceptance criterion.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Front acts as a single entity requiring a iteration before moving to the next front.
    - Each time it is required to take the next solution to be processed, it take the next not processed solution from pareto front.
    - In case it accepts a solution, it updates the upcoming front.
    - In case it rejects a solution, solution is discarded.
    - After iteration through the current front is done, it switches to the upcoming front being a union of accepted solutions and non-dominated solutions from the previous front.
    """


class AcceptBatchBigger(_Bigger, _AcceptBatch, AcceptanceCriterion):
    """
    Maximization acceptance criterion.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Front acts as a single entity requiring a iteration before moving to the next front.
    - Each time it is required to take the next solution to be processed, it take the next not processed solution from pareto front.
    - In case it accepts a solution, it updates the upcoming front.
    - In case it rejects a solution, solution is discarded.
    - After iteration through the current front is done, it switches to the upcoming front being a union of accepted solutions and non-dominated solutions from the previous front.
    """


class AcceptBatchSkewedSmaller(_Smaller, _AcceptBatchSkewed, AcceptanceCriterion):
    """
    Skewed Acceptance Criterion for SVNS (minimization).
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Front acts as a single entity requiring a iteration before moving to the next front.
    - Each time it is required to take the next solution to be processed, it take the next not processed solution from pareto front and if exhausted take a skewed accepted solution.
    - In case it accepts a solution, it updates the upcoming front.
    - In case it rejects a solution, but it is quite different from solutions in the front, solution is added to the "skewed front", these solutions are not present in the front, but will be selected during iterations.
    - In case it completely rejects a solution, solution is discarded.
    - After iteration through the current front and skewed front is done, it switches to the upcoming front being a union of accepted solutions and non-dominated solutions from the previous front.
    """


class AcceptBatchSkewedBigger(_Bigger, _AcceptBatchSkewed, AcceptanceCriterion):
    """
    Skewed Acceptance Criterion for SVNS (maximization).
    Maximization acceptance criterion with beam search like behavior implementing Skewed acceptance.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Front acts as a single entity requiring a iteration before moving to the next front.
    - Each time it is required to take the next solution to be processed, it take the next not processed solution from pareto front and if exhausted take a skewed accepted solution.
    - In case it accepts a solution, it updates the upcoming front.
    - In case it rejects a solution, but it is quite different from solutions in the front, solution is added to the "skewed front", these solutions are not present in the front, but will be selected during iterations.
    - In case it completely rejects a solution, solution is discarded.
    - After iteration through the current front and skewed front is done, it switches to the upcoming front being a union of accepted solutions and non-dominated solutions from the previous front.
    """


class AcceptBeamSmaller(_Smaller, _AcceptBeam, AcceptanceCriterion):
    """
    Minimization acceptance criterion with beam search like behavior.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, solution is discarded.
    """


class AcceptBeamBigger(_Bigger, _AcceptBeam, AcceptanceCriterion):
    """
    Maximization acceptance criterion with beam search like behavior.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, solution is discarded.
    """


class AcceptBeamSkewedSmaller(_Smaller, _AcceptBeamSkewed, AcceptanceCriterion):
    """
    Skewed Acceptance Criterion for SVNS (minimization).
    Minimization acceptance criterion with beam search like behavior implementing Skewed acceptance.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front or takes out a skewed-accepted solution from the buffer.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, but it is quite different from solutions in the front, solution is added to the "buffer", these solutions are not present in the front, but will be selected during iterations.
    - In case it completely rejects a solution, solution is discarded.
    """


class AcceptBeamSkewedBigger(_Bigger, _AcceptBeamSkewed, AcceptanceCriterion):
    """
    Skewed Acceptance Criterion for SVNS (maximization).
    Maximization acceptance criterion with beam search like behavior implementing Skewed acceptance.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front or takes out a skewed-accepted solution from the buffer.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, but it is quite different from solutions in the front, solution is added to the "buffer", these solutions are not present in the front, but will be selected during iterations.
    - In case it completely rejects a solution, solution is discarded.
    """
