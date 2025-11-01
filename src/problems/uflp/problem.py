import json
import random
from typing import Iterable

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.core.abstract import Problem, Solution
from src.problems.default_pymoo_configurations import PymooProblemConfig

type MOUFLPSolution = Solution[np.ndarray]


class _MOUFLPSolution(Solution[np.ndarray]):
    def get_hash(self) -> int:
        return hash(self.objectives)

    def to_json_serializable(self):
        return self.data.tolist()

    @staticmethod
    def from_json_serializable(
        problem: Problem[np.ndarray], serialized_data: list[int]
    ) -> MOUFLPSolution:
        return _MOUFLPSolution(np.array(serialized_data).astype(bool), problem)


class MOUFLPProblem(Problem[np.ndarray]):
    def __init__(
        self,
        num_customers: int,
        num_facilities: int,
        fixed_costs: list[list[int]],
        customer_costs: list[list[list[int]]],
    ):
        """
        Initializes the Uncapacitated Facility Location Problem (UFLP).

        Args:
            num_customers: Number of demand points (M).
            num_facilities: Number of potential facilities (N).
            fixed_costs: [num_fixed_obj x num_facilities] fixed cost array.
            customer_costs: [num_assignment_obj x num_customers x num_facilities] assignment cost matrix.
        """
        self.num_fixed_obj = len(fixed_costs)
        self.num_assignment_obj = len(customer_costs)
        total_objectives = self.num_fixed_obj + self.num_assignment_obj

        super().__init__(
            num_variables=num_facilities,
            num_objectives=total_objectives,
            num_constraints=1,
            problem_name="UFLP",
        )

        self.num_customers = num_customers
        self.num_facilities = num_facilities

        # Fixed costs: (num_fixed_obj x num_facilities)
        self.fixed_costs = np.array(fixed_costs, dtype=float)
        # Assignment costs: (num_assignment_obj x num_customers x num_facilities)
        self.assignment_costs = np.array(customer_costs, dtype=float)

        self.max_fixed_cost = np.sum(self.fixed_costs)
        self.max_assignment_cost = np.sum(np.min(self.assignment_costs, axis=1))

    def get_initial_solutions(
        self, num_solutions: int = 50
    ) -> Iterable[MOUFLPSolution]:
        solutions = []

        for _ in range(num_solutions):
            solution_data = np.zeros(self.num_variables, dtype=np.bool_)
            num_to_open = random.randint(1, self.num_facilities)
            open_indices = np.random.choice(
                self.num_facilities, size=num_to_open, replace=False
            )
            solution_data[open_indices] = True

            solutions.append(_MOUFLPSolution(solution_data, self))

        return solutions

    def satisfies_constraints(self, solution_data: np.ndarray) -> bool:
        return bool(np.any(solution_data))

    def calculate_objectives(self, solution_data: np.ndarray) -> tuple[float, ...]:
        if not self.satisfies_constraints(solution_data):
            return tuple([100000000]) * self.num_objectives

        # Element-wise multiplication: Zeroes out costs for closed facilities.
        zeroed_fixed_costs = self.fixed_costs * solution_data
        fixed_cost_objectives = np.sum(zeroed_fixed_costs, axis=1)

        # Create a mask to assign Inf to costs of CLOSED facilities.
        # costs_mask is (1 x 1 x num_facilities), True for open facilities
        costs_mask = solution_data[np.newaxis, np.newaxis, :]

        # Penalized costs: Set cost of closed facilities to Inf
        penalized_assignment_costs = np.where(costs_mask, self.assignment_costs, np.inf)

        min_assignment_costs = np.min(penalized_assignment_costs, axis=2)
        assignment_cost_objectives = np.sum(min_assignment_costs, axis=1)
        all_objectives_flat = np.concatenate(
            [fixed_cost_objectives, assignment_cost_objectives]
        )
        return tuple(all_objectives_flat.tolist())

    def load_solution(self, saved_solution_data) -> MOUFLPSolution:
        return _MOUFLPSolution.from_json_serializable(self, saved_solution_data)

    @staticmethod
    def calculate_solution_distance(
        sol1: MOUFLPSolution, sol2: MOUFLPSolution
    ) -> float:
        return float(np.sum(sol1.data != sol2.data)) / sol1.data.size

    @staticmethod
    def load(filename: str) -> "MOUFLPProblem":
        try:
            with open(filename, "r") as f:
                configuration = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Error: File not found at {filename}")
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")

        data = configuration["data"]

        num_facilities = data["num_facilities"]
        fixed_costs = data["fixed_costs"]

        num_customers = data["num_customers"]
        customer_costs = data["customer_costs"]

        return MOUFLPProblem(num_customers, num_facilities, fixed_costs, customer_costs)

    @staticmethod
    def calculate_solution_distance_2(
        sol1: MOUFLPSolution, sol2: MOUFLPSolution
    ) -> float:
        """Calculates a distance between two MOKP solutions in [0, 1]."""
        return float(np.sum(sol1.data != sol2.data) / np.sum(sol1.data | sol2.data))


class MOUFLPProblemPymoo(ElementwiseProblem):
    def __init__(self, problem: MOUFLPProblem):
        n_var = problem.num_variables
        n_obj = problem.num_objectives

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=1,
            xl=0.0,
            xu=1.0,
            vtype=float,
        )
        self.problem_instance = problem

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        x = np.round(x).astype(bool)
        out["F"] = np.array(self.problem_instance.calculate_objectives(x), dtype=float)
        out["G"] = int(self.problem_instance.satisfies_constraints(x)) - 1

    def to_config(self):
        return PymooProblemConfig(
            problem_instance=self.problem_instance,
            serialize_output=lambda result: [
                Solution(
                    objectives=objectives.tolist(),
                    data=data.tolist(),
                    problem=self.problem_instance,
                )
                for objectives, data in zip(result.F, np.round(result.X).astype(bool))
            ],
            pymoo_problem=self,
        )
