import random
from collections import deque
from enum import Enum
from typing import Callable

from src.vns.abstract import AcceptanceCriterion, Solution


class ComparisonResult(Enum):
    """
    Represents the result of a comparison between two solutions.
    """

    STRICTLY_BETTER = 1
    NON_DOMINATED = 2
    WORSE = 3


def is_dominating_min(
    new_objective: tuple[float, ...],
    current_objective: tuple[float, ...],
) -> ComparisonResult:
    """
    Checks if objective vector new_objective dominates objective vector current_objective.
    Assumes minimization for all objectives.
    """
    if len(new_objective) != len(current_objective):
        raise ValueError("Objective vectors must have the same number of objectives.")

    has_better = False
    has_worse = False

    for new_val, current_val in zip(new_objective, current_objective):
        if abs(new_val - current_val) < 1e-6:
            continue

        if new_val < current_val:
            has_better = True

        if new_val > current_val:
            has_worse = True

    if has_better and not has_worse:
        return ComparisonResult.STRICTLY_BETTER

    if not has_better and has_worse:
        return ComparisonResult.WORSE

    return ComparisonResult.NON_DOMINATED


class AcceptBeam(AcceptanceCriterion):
    """
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, solution is discarded.
    """

    def __init__(self):
        super().__init__()

        self.front: list[Solution] = []

    def accept(self, candidate: Solution) -> bool:
        """
        Decides whether to accept candidate and update the archive.
        Updates the non-dominated archive based on Pareto dominance.
        """

        candidate_is_dominating = False

        for solution in self.front:
            if candidate == solution:
                return False

            result = is_dominating_min(candidate.objectives, solution.objectives)

            if result == ComparisonResult.WORSE:
                return False

            if result == ComparisonResult.STRICTLY_BETTER:
                candidate_is_dominating = True

        # Prune dominated solutions in-place using two pointers.
        if candidate_is_dominating:
            i = 0
            for j in range(len(self.front)):
                solution = self.front[j]

                if (
                    is_dominating_min(candidate.objectives, solution.objectives)
                    != ComparisonResult.STRICTLY_BETTER
                ):
                    self.front[i] = solution
                    i += 1

            del self.front[i:]

        self.front.append(candidate)
        return True

    def get_all_solutions(self) -> list[Solution]:
        return self.front

    def get_one_current_solution(self) -> Solution:
        if not self.front:
            raise ValueError("Front is empty. Cannot select a solution.")
        return random.choice(self.front)

    def clear(self):
        self.front.clear()


class AcceptBeamSkewed(AcceptBeam):
    """
    Skewed Acceptance Criterion for SVNS.
    Minimization acceptance criterion with beam search like behavior implementing Skewed acceptance.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front or takes out a skewed-accepted solution from the buffer.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, but it is quite different from solutions in the front, solution is added to the "buffer", these solutions are not present in the front, but will be selected during iterations.
    - In case it completely rejects a solution, solution is discarded.
    """

    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
        max_skewed_solutions=100,
        accept_skewed_non_dominated=False,
    ):
        """Init.

        Args:
            alpha (list[float]): List of alpha weights per objective.
            distance_metric ((Solution, Solution) -> float): Gives the difference distance between two solutions.
            buffer_size (int | None): Limit the number of saved skewed accepted solutions to track.
        """
        super().__init__()
        self.alpha = alpha
        self.distance_metric = distance_metric

        self.skewed_buffer: deque[Solution] = deque(maxlen=max_skewed_solutions)
        self.accept_skewed_non_dominated = accept_skewed_non_dominated

    def accept(self, candidate: Solution) -> bool:
        if len(candidate.objectives) != len(self.alpha):
            raise ValueError(
                f"Expected to have the same number of alpha weights ({len(self.alpha)}) as the number of objectives {len(candidate.objectives)}."
            )

        if super().accept(candidate):
            return True

        for solution in self.front:
            distance = self.distance_metric(candidate, solution)

            skewed_candidate_objectives = tuple(
                obj_i - self.alpha[i] * distance
                for i, obj_i in enumerate(candidate.objectives)
            )

            result = is_dominating_min(skewed_candidate_objectives, solution.objectives)
            if (
                result == ComparisonResult.STRICTLY_BETTER
                or self.accept_skewed_non_dominated
                and result == ComparisonResult.NON_DOMINATED
            ):
                self.skewed_buffer.append(candidate)
                return True

        return False

    def get_one_current_solution(self) -> Solution:
        front_size = len(self.front)
        buffer_size = len(self.skewed_buffer)
        total_size = front_size + buffer_size

        if total_size == 0:
            raise ValueError("No solutions available in the front or buffer.")

        if buffer_size == 0 or random.random() < front_size / total_size:
            return random.choice(self.front)

        return self.skewed_buffer.popleft()

    def clear(self):
        super().clear()
        self.skewed_buffer.clear()


class AcceptBatch(AcceptanceCriterion):
    """
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Front acts as a single entity requiring a iteration before moving to the next front.
    - Each time it is required to take the next solution to be processed, it take the next not processed solution from pareto front.
    - In case it accepts a solution, it updates the upcoming front.
    - In case it rejects a solution, solution is discarded.
    - After iteration through the current front is done, it switches to the upcoming front being a union of accepted solutions and non-dominated solutions from the previous front.
    """

    def __init__(self):
        super().__init__()
        # Holds the live, updated non-dominated solutions
        self.archive: AcceptBeam = AcceptBeam()
        # Holds the solutions to be iterated over in the current batch
        self.front_snapshot: list[Solution] = []

    def accept(self, candidate: Solution) -> bool:
        accepted = self.archive.accept(candidate)
        if not self.front_snapshot:
            self._take_snapshot()

        return accepted

    def get_all_solutions(self) -> list[Solution]:
        return self.archive.get_all_solutions()

    def get_one_current_solution(self) -> Solution:
        if not self.front_snapshot:
            self._take_snapshot()

        if self.front_snapshot:
            return self.front_snapshot.pop()

        raise ValueError("Archive is empty and all solutions have been processed.")

    def _take_snapshot(self):
        self.front_snapshot = self.archive.front.copy()

    def clear(self):
        self.front_snapshot.clear()
        self.archive.clear()


class AcceptBatchSkewed(AcceptanceCriterion):
    """
    Skewed Acceptance Criterion for SVNS.
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Front acts as a single entity requiring a iteration before moving to the next front.
    - Each time it is required to take the next solution to be processed, it take the next not processed solution from pareto front and if exhausted take a skewed accepted solution.
    - In case it accepts a solution, it updates the upcoming front.
    - In case it rejects a solution, but it is quite different from solutions in the front, solution is added to the "skewed front", these solutions are not present in the front, but will be selected during iterations.
    - In case it completely rejects a solution, solution is discarded.
    - After iteration through the current front and skewed front is done, it switches to the upcoming front being a union of accepted solutions and non-dominated solutions from the previous front.
    """

    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
        max_skewed_solutions=100,
        accept_skewed_non_dominated=False,
    ):
        super().__init__()
        self.archive: AcceptBeamSkewed = AcceptBeamSkewed(
            alpha, distance_metric, max_skewed_solutions, accept_skewed_non_dominated
        )
        self.front_snapshot: list[Solution] = []
        self.skewed_buffer_snapshot: deque[Solution] = deque()

    def accept(self, candidate: Solution) -> bool:
        accepted = self.archive.accept(candidate)
        if not self.front_snapshot:
            self._take_snapshot()

        return accepted

    def get_all_solutions(self) -> list[Solution]:
        return self.archive.get_all_solutions()

    def get_one_current_solution(self) -> Solution:
        if not self.front_snapshot and not self.skewed_buffer_snapshot:
            self._take_snapshot()

        if self.front_snapshot:
            return self.front_snapshot.pop()

        if self.skewed_buffer_snapshot:
            return self.skewed_buffer_snapshot.pop()

        raise ValueError("Archive and skewed buffer are empty.")

    def _take_snapshot(self):
        self.front_snapshot = self.archive.front.copy()
        self.skewed_buffer_snapshot = self.archive.skewed_buffer.copy()

        self.archive.skewed_buffer.clear()

    def clear(self):
        self.front_snapshot.clear()
        self.skewed_buffer_snapshot.clear()
        self.archive.clear()
