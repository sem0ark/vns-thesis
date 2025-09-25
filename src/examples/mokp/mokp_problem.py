import json
import random
from typing import Iterable

import numpy as np
from pymoo.core.problem import ElementwiseProblem
import xxhash

from src.vns.abstract import Problem, Solution

type MOKPSolution = Solution[np.ndarray]


class _MOKPSolution(Solution[np.ndarray]):
    def equals(self, other: Solution[np.ndarray]) -> bool:
        return hash(self) == hash(other)

    def get_hash(self) -> int:
        h = xxhash.xxh64()
        h.update(self.data.tobytes())
        return h.intdigest()


class MOKPProblem(Problem[np.ndarray]):
    def __init__(
        self,
        weights: list[list[int]],
        profits: list[list[int]],
        capacity: int | list[int],
    ):
        super().__init__(self.evaluate, self.generate_initial_solutions)

        self.weights = np.array(weights, dtype=int)
        self.profits = np.array(profits, dtype=int)
        self.capacity = np.array(capacity, dtype=int)

        self.num_items = self.weights.shape[1]
        self.num_objectives = self.profits.shape[0]
        self.num_limits = self.weights.shape[0]

    def generate_initial_solutions(
        self, num_solutions: int = 50
    ) -> Iterable[MOKPSolution]:
        """
        Generates a specified number of random feasible solutions for the MOKP.
        Each solution is created by iterating through items in a random order
        and adding them to the knapsack if they do not violate the capacity constraint.
        """
        solutions = []
        for _ in range(num_solutions):
            solution_data = np.zeros(self.num_items, dtype=int)
            items_to_add = list(range(self.num_items))
            random.shuffle(items_to_add)

            for item_idx in items_to_add:
                temp_data = solution_data.copy()
                temp_data[item_idx] = 1
                if self.is_feasible(temp_data):
                    solution_data = temp_data

            solutions.append(_MOKPSolution(solution_data, self))
        return solutions

    def is_feasible(self, solution_data: np.ndarray) -> bool:
        """Checks if a solution is feasible with respect to knapsack capacity."""
        total_weight = np.sum(solution_data * self.weights, axis=1)
        return np.all(total_weight <= self.capacity, axis=0)

    def evaluate(self, solution: MOKPSolution) -> tuple[float, ...]:
        """Calculates the profit for each objective."""
        solution_data = solution.data
        mult = 1 if self.is_feasible(solution.data) else 0

        # Negate for minimization
        result = tuple(
            -1 * mult * np.sum(solution_data * self.profits[i])
            for i in range(self.num_objectives)
        )
        return result

    @staticmethod
    def calculate_solution_distance(sol1: MOKPSolution, sol2: MOKPSolution) -> float:
        """Calculates a distance between two MOKP solutions in [0, 1]."""
        return float(np.sum(sol1.data != sol2.data)) / sol2.data.size

    @staticmethod
    def load(filename: str) -> "MOKPProblem":
        """
        Creates a mock MOKP problem for the example.
        In a real scenario, this would load from a file like MOCOLib instances.
        """
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

        return MOKPProblem(weights, profits, capacity)


class MOKPPymoo(ElementwiseProblem):
    """
    Multi-Objective Knapsack Problem.

    A problem is defined by inheriting from the Problem class and
    implementing the _evaluate method.
    """

    def __init__(self, problem: MOKPProblem):
        super().__init__(
            n_var=problem.num_items,
            n_obj=problem.num_objectives,
            n_constr=problem.num_limits,
            xl=0.0,
            xu=1.0,
            vtype=bool,
        )
        self.problem_instance = problem

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution `x`.
        `x` is a NumPy array representing a single solution (a vector of booleans).
        """
        x = np.round(x)  # NSGA instance is still resulting in array of floats
        total_profits = np.sum(x * self.problem_instance.profits, axis=1)

        # Objectives: We minimize the negative profits
        out["F"] = -total_profits

        # Constraints: A solution is feasible if its weight is within capacity for all limits.
        # This will be an array of values, one for each constraint.
        total_weights = np.sum(x * self.problem_instance.weights, axis=1)
        out["G"] = total_weights - self.problem_instance.capacity
