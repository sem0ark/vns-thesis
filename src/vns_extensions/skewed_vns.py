from functools import lru_cache
from typing import Callable

from src.core.abstract import Solution
from src.vns.acceptance import (
    AcceptBatch,
    AcceptBatchWrapped,
    ComparisonResult,
    is_dominating_min,
)


class AcceptBatchSkewedV1(AcceptBatchWrapped):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = distance_metric(new_solution, current_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            return is_dominating_min(skewed_objectives, current_solution.objectives)

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV2(AcceptBatchWrapped):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            standard_result = is_dominating_min(
                new_solution.objectives, current_solution.objectives
            )
            if standard_result != ComparisonResult.WORSE:
                return standard_result

            distance = distance_metric(new_solution, current_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            skewed_result = is_dominating_min(
                skewed_objectives, current_solution.objectives
            )
            if skewed_result != ComparisonResult.WORSE:
                return ComparisonResult.NON_DOMINATED

            return ComparisonResult.WORSE

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV3(AcceptBatchWrapped):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        @lru_cache(maxsize=256)
        def calculate_average_distance(new_solution: Solution):
            all_solutions = self.get_all_solutions()
            if not all_solutions:
                return 1.0

            return sum(
                [distance_metric(new_solution, sol) for sol in all_solutions]
            ) / len(all_solutions)

        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = calculate_average_distance(new_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            return is_dominating_min(skewed_objectives, current_solution.objectives)

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV4(AcceptBatchWrapped):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        @lru_cache(maxsize=256)
        def calculate_average_distance(new_solution: Solution):
            all_solutions = self.get_all_solutions()
            if not all_solutions:
                return 1.0

            return min([distance_metric(new_solution, sol) for sol in all_solutions])

        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = calculate_average_distance(new_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            return is_dominating_min(skewed_objectives, current_solution.objectives)

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV5(AcceptBatch):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        @lru_cache(maxsize=256)
        def calculate_average_distance(new_solution: Solution):
            all_solutions = self.get_all_solutions()
            if not all_solutions:
                return 1.0

            return min([distance_metric(new_solution, sol) for sol in all_solutions])

        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = calculate_average_distance(new_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            return is_dominating_min(skewed_objectives, current_solution.objectives)

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV6(AcceptBatch):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        @lru_cache(maxsize=256)
        def calculate_average_distance(new_solution: Solution):
            all_solutions = self.get_all_solutions()
            if not all_solutions:
                return 1.0

            return min([distance_metric(new_solution, sol) for sol in all_solutions])

        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            standard_result = is_dominating_min(
                new_solution.objectives, current_solution.objectives
            )
            if standard_result != ComparisonResult.WORSE:
                return standard_result

            distance = calculate_average_distance(new_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            skewed_result = is_dominating_min(
                skewed_objectives, current_solution.objectives
            )
            if skewed_result != ComparisonResult.WORSE:
                return ComparisonResult.NON_DOMINATED

            return ComparisonResult.WORSE

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV7(AcceptBatch):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            standard_result = is_dominating_min(
                new_solution.objectives, current_solution.objectives
            )
            if standard_result != ComparisonResult.WORSE:
                return standard_result

            distance = distance_metric(new_solution, current_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            skewed_result = is_dominating_min(
                skewed_objectives, current_solution.objectives
            )
            if skewed_result != ComparisonResult.WORSE:
                return ComparisonResult.NON_DOMINATED

            return ComparisonResult.WORSE

        super().__init__(compare_solutions_better_skewed)


class AcceptBatchSkewedV8(AcceptBatch):
    def __init__(
        self,
        alpha: list[float],
        distance_metric: Callable[[Solution, Solution], float],
    ):
        def compare_solutions_better_skewed(
            new_solution: Solution, current_solution: Solution
        ) -> ComparisonResult:
            distance = distance_metric(new_solution, current_solution)
            skewed_objectives = tuple(
                obj_i - alpha[i] * distance
                for i, obj_i in enumerate(new_solution.objectives)
            )
            return is_dominating_min(skewed_objectives, current_solution.objectives)

        super().__init__(compare_solutions_better_skewed)
