import json
import random
from typing import Iterable

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.core.abstract import Problem, Solution

type MOKPSolution = Solution[np.ndarray]


class _MOKPSolution(Solution[np.ndarray]):
    def get_hash(self) -> int:
        return hash(self.objectives)

    def to_json_serializable(self):
        return self.data.tolist()

    @staticmethod
    def from_json_serializable(
        problem: Problem[np.ndarray], serialized_data: list[int]
    ) -> MOKPSolution:
        if not isinstance(serialized_data, list):
            raise ValueError(
                "Expected saved_solution_data to be list of ints (0 or 1)!"
            )
        solution_data = np.array(serialized_data)
        return _MOKPSolution(
            solution_data, problem, problem.calculate_objectives(solution_data)
        )


class MOKPProblem(Problem[np.ndarray]):
    def __init__(
        self,
        weights: list[list[int]],
        profits: list[list[int]],
        capacity: int | list[int],
    ):
        super().__init__(
            num_variables=len(weights[0]),
            num_objectives=len(profits),
            num_constraints=len(weights),
            problem_name="MOKP",
        )

        self.weights = np.array(weights, dtype=int)
        self.profits = np.array(profits, dtype=int)
        self.capacity = np.array(capacity, dtype=int)

    def get_initial_solutions(self, num_solutions: int = 50) -> Iterable[MOKPSolution]:
        """
        Generates a specified number of random feasible solutions for the MOKP.
        Each solution is created by iterating through items in a random order
        and adding them to the knapsack if they do not violate the capacity constraint.
        """
        solutions = []
        for _ in range(num_solutions):
            solution_data = np.zeros(self.num_variables, dtype=int)
            items_to_add = list(range(self.num_variables))
            random.shuffle(items_to_add)

            for item_idx in items_to_add:
                solution_data[item_idx] = 1
                if not self.satisfies_constraints(solution_data):
                    solution_data[item_idx] = 0

            solutions.append(
                _MOKPSolution(
                    solution_data, self, self.calculate_objectives(solution_data)
                )
            )
        return solutions

    def satisfies_constraints(self, solution_data: np.ndarray) -> bool:
        """Checks if a solution is feasible with respect to knapsack capacity."""
        total_weight = np.sum(solution_data * self.weights, axis=1)
        return np.all(total_weight <= self.capacity, axis=0)

    def calculate_objectives(self, solution_data: np.ndarray) -> tuple[float, ...]:
        """Calculates the profit for each objective."""
        multiplier = 1 if self.satisfies_constraints(solution_data) else -1
        result = -1 * multiplier * np.sum(solution_data * self.profits, axis=1)
        # Negated for maximization
        return tuple(result.tolist())

    def load_solution(self, saved_solution_data) -> MOKPSolution:
        return _MOKPSolution.from_json_serializable(self, saved_solution_data)

    @staticmethod
    def calculate_solution_distance(sol1: MOKPSolution, sol2: MOKPSolution) -> float:
        """Calculates a distance between two MOKP solutions in [0, 1]."""
        return float(np.sum(sol1.data != sol2.data)) / sol2.data.size

    @staticmethod
    def load(filename: str) -> "MOKPProblem":
        try:
            with open(filename, "r") as f:
                configuration = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Error: File not found at {filename}")
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")

        weights = configuration["data"]["weights"]
        profits = configuration["data"]["objectives"]
        capacity = configuration["data"]["capacity"]

        problem = MOKPProblem(weights, profits, capacity)
        return problem


class MOKPPymoo(ElementwiseProblem):
    """
    Multi-Objective Knapsack Problem.

    A problem is defined by inheriting from the Problem class and
    implementing the _evaluate method.
    """

    def __init__(self, problem: MOKPProblem):
        super().__init__(
            n_var=problem.num_variables,
            n_obj=problem.num_objectives,
            n_constr=problem.num_constraints,
            xl=0.0,
            xu=1.0,
            vtype=bool,
        )
        self.problem_instance = problem

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x)  # NSGA instance is still resulting in array of floats
        total_profits = np.sum(x * self.problem_instance.profits, axis=1)

        # Objectives: We minimize the negative profits
        out["F"] = -total_profits

        # Constraints: A solution is feasible if its weight is within capacity for all limits.
        # This will be an array of values, one for each constraint.
        total_weights = np.sum(x * self.problem_instance.weights, axis=1)
        out["G"] = total_weights - self.problem_instance.capacity
