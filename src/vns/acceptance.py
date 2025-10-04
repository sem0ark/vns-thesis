import random
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


def compare_solutions_better(
    new_solution: Solution,
    current_solution: Solution,
) -> ComparisonResult:
    return is_dominating_min(new_solution.objectives, current_solution.objectives)


class ParetoFront(AcceptanceCriterion):
    """
    Maintains an archive of non-dominated solutions (Pareto front).

    This acceptance criterion proposes iteration through the current front to be:
    - Instead of treating front as a single entity requiring a single iteration, it treats it as a buffer of solutions.
    - Each time it is required to take the next solution to be processed, it takes a random solution from pareto front.
    - In case it accepts a solution, it updates the current front.
    - In case it rejects a solution, solution is discarded.
    """

    def __init__(self, comparison_function=compare_solutions_better):
        super().__init__()

        self.front: list[Solution] = []
        self.comparison_cache: dict[int, ComparisonResult] = {}
        self.compare_solutions = comparison_function

    def accept(self, candidate: Solution) -> bool:
        """
        Decides whether to accept candidate and update the archive.
        Updates the non-dominated archive based on Pareto dominance.
        """

        candidate_is_dominating = False

        for i, solution in enumerate(self.front):
            if candidate == solution:
                return False

            result = self.compare_solutions(candidate, solution)
            self.comparison_cache[i] = result

            if result == ComparisonResult.WORSE:
                return False

            if result == ComparisonResult.STRICTLY_BETTER:
                candidate_is_dominating = True

        # Prune dominated solutions in-place using two pointers.
        if candidate_is_dominating:
            i = 0
            for j in range(len(self.front)):
                solution = self.front[j]

                if self.comparison_cache[j] != ComparisonResult.STRICTLY_BETTER:
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
        self.comparison_cache.clear()


class AcceptBeamSkewed(AcceptanceCriterion):
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
    ):
        """Init.

        Args:
            alpha (list[float]): List of alpha weights per objective.
            distance_metric ((Solution, Solution) -> float): Gives the difference distance between two solutions.
            buffer_size (int | None): Limit the number of saved skewed accepted solutions to track.
        """
        super().__init__()
        self.alpha = alpha

        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = distance_metric(new_solution, current_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(current_solution.objectives)
            )
            return is_dominating_min(skewed_objectives, current_solution.objectives)

        self.true_front = ParetoFront()
        self.skewed_front = ParetoFront(
            comparison_function=compare_solutions_better_skewed
        )

    def accept(self, candidate: Solution) -> bool:
        if len(candidate.objectives) != len(self.alpha):
            raise ValueError(
                f"Expected to have the same number of alpha weights ({len(self.alpha)}) as the number of objectives {len(candidate.objectives)}."
            )

        self.true_front.accept(candidate)
        return self.skewed_front.accept(candidate)

    def get_one_current_solution(self) -> Solution:
        return self.skewed_front.get_one_current_solution()

    def get_all_solutions(self) -> list[Solution]:
        return self.true_front.get_all_solutions()

    def clear(self):
        self.true_front.clear()
        self.skewed_front.clear()


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
        self.true_front: ParetoFront = ParetoFront()
        # Holds the solutions to be iterated over in the current batch
        self.front_snapshot: list[Solution] = []

    def accept(self, candidate: Solution) -> bool:
        accepted = self.true_front.accept(candidate)
        if not self.front_snapshot:
            self._take_snapshot()

        return accepted

    def get_all_solutions(self) -> list[Solution]:
        return self.true_front.get_all_solutions()

    def get_one_current_solution(self) -> Solution:
        if not self.front_snapshot:
            self._take_snapshot()

        if self.front_snapshot:
            return self.front_snapshot.pop()

        raise ValueError("Archive is empty and all solutions have been processed.")

    def _take_snapshot(self):
        current_solutions = self.true_front.get_all_solutions()
        for i, solution in enumerate(current_solutions):
            if i == len(self.front_snapshot):
                self.front_snapshot.append(solution)
            else:
                self.front_snapshot[i] = solution

            del self.front_snapshot[len(current_solutions) :]

    def clear(self):
        self.front_snapshot.clear()
        self.true_front.clear()


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
    ):
        super().__init__()
        self.alpha = alpha

        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = distance_metric(new_solution, current_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(current_solution.objectives)
            )
            # return is_dominating_min(skewed_objectives, current_solution.objectives)
            if (
                is_dominating_min(skewed_objectives, current_solution.objectives)
                != ComparisonResult.WORSE
            ):
                return ComparisonResult.NON_DOMINATED
            return is_dominating_min(
                new_solution.objectives, current_solution.objectives
            )

        # Holds the live, updated non-dominated solutions
        self.true_front = ParetoFront()
        self.skewed_front = ParetoFront(
            comparison_function=compare_solutions_better_skewed
        )
        # Holds the solutions to be iterated over in the current batch
        self.front_snapshot: list[Solution] = []

    def accept(self, candidate: Solution) -> bool:
        self.true_front.accept(candidate)
        accepted = self.skewed_front.accept(candidate)

        if not self.front_snapshot:
            self._take_snapshot()

        return accepted

    def get_all_solutions(self) -> list[Solution]:
        return self.true_front.get_all_solutions()

    def get_one_current_solution(self) -> Solution:
        if not self.front_snapshot:
            self._take_snapshot()

        if self.front_snapshot:
            return self.front_snapshot.pop()

        raise ValueError("Archive and skewed buffer are empty.")

    def _take_snapshot(self):
        current_solutions = self.skewed_front.get_all_solutions()
        for i, solution in enumerate(current_solutions):
            if i == len(self.front_snapshot):
                self.front_snapshot.append(solution)
            else:
                self.front_snapshot[i] = solution

            del self.front_snapshot[len(current_solutions) :]

    def clear(self):
        self.true_front.clear()
        self.skewed_front.clear()
        self.front_snapshot.clear()
