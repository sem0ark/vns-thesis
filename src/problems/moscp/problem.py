import json
import random
from typing import Iterable

import numpy as np
import xxhash
from pymoo.core.problem import ElementwiseProblem

from src.core.abstract import Problem, Solution

type MOSCPSolution = Solution[np.ndarray]


class _MOSCPSolution(Solution[np.ndarray]):
    """
    Represents a solution for MO-SCP.
    The data attribute is a *packed* numpy.uint8 array.
    """

    def get_hash(self) -> int:
        h = xxhash.xxh64()
        h.update(self.data.tobytes())
        return h.intdigest()

    def to_json_serializable(self):
        return self.data.astype(int).tolist()

    @staticmethod
    def from_json_serializable(
        problem: Problem[np.ndarray], serialized_data: list[int]
    ) -> MOSCPSolution:
        # Load the unpacked data, then pack it for internal storage
        return _MOSCPSolution(np.array(serialized_data).astype(bool), problem)


class MOSCPProblem(Problem[np.ndarray]):
    def __init__(
        self,
        coverage_data: list[list[int]],
        costs: list[list[int]],
        num_items: int,
        num_sets: int,
    ):
        super().__init__(
            num_variables=num_sets,
            num_objectives=len(costs),
            num_constraints=1,
        )

        self.num_items = num_items
        self.num_sets = num_sets
        self.costs = np.array(costs, dtype=int)

        # Coverage matrix: self.coverage[item_index, set_index] = 1 if set_index covers item_index
        self.coverage_unpacked = np.zeros((num_items, num_sets), dtype=np.uint8)
        for item_index, covering_sets_1based in enumerate(coverage_data):
            covering_sets_0based = [s - 1 for s in covering_sets_1based]
            self.coverage_unpacked[item_index, covering_sets_0based] = 1

        # Transpose and Pack the Sets: (num_sets x packed_item_bits)
        self.set_coverage_packed = np.array(
            [np.packbits(self.coverage_unpacked[:, j]) for j in range(num_sets)]
        )

        # Create the ALL_COVERED mask (a bit vector of all 1s for the item space)
        # Useful in case we have num_items % 8 != 0
        all_ones = np.ones(num_items, dtype=np.uint8)
        self.all_covered_mask = np.packbits(all_ones)

    def get_initial_solutions(self, num_solutions: int = 50) -> Iterable[MOSCPSolution]:
        solutions = []

        for _ in range(num_solutions):
            solution_data = np.zeros(self.num_variables, dtype=np.bool_)
            uncovered_items = set(range(self.num_constraints))

            while uncovered_items:
                item_index = random.choice(list(uncovered_items))
                possible_sets_indices = np.where(self.coverage_unpacked[item_index] == 1)[
                    0
                ]

                if not possible_sets_indices.size:
                    raise RuntimeError(
                        "Infeasible instance: An item cannot be covered."
                    )

                # Only consider sets that haven't been selected yet for a random choice
                unselected_sets = [
                    s for s in possible_sets_indices if solution_data[s] == 0
                ]
                if not unselected_sets:
                    break

                set_to_add = random.choice(unselected_sets)
                solution_data[set_to_add] = 1

                # Check coverage against current selection
                current_selection_mask = solution_data == 1
                total_coverage = self.coverage_unpacked @ current_selection_mask

                # Update uncovered items
                uncovered_items = set(np.where(total_coverage < 1)[0])

            solutions.append(_MOSCPSolution(solution_data, self))

        return solutions

    def satisfies_constraints(self, solution_data: np.ndarray) -> bool:
        """
        Checks if a solution is feasible (every item is covered).
        Requires unpacking the solution data.
        """
        set_selection_vector = solution_data

        # Get the packed coverage vectors for ALL selected sets
        selected_set_coverages = self.set_coverage_packed[set_selection_vector]
        if selected_set_coverages.size == 0:
            return False

        # Perform the Bitwise OR across all selected sets
        # np.bitwise_or.reduce computes the cumulative OR along the first axis.
        # This results in a single packed array representing the total coverage.
        total_coverage_packed = np.bitwise_or.reduce(selected_set_coverages, axis=0)

        # Check if the total coverage equals the mask where all items are set to 1
        return bool(np.all(total_coverage_packed == self.all_covered_mask))

    def calculate_objectives(self, solution_data: np.ndarray) -> tuple[float, ...]:
        """Calculates the total cost for each objective."""
        # Drastically increase costs in case of infeasible solution
        # to still keep improvement gradient even for infeasible solution in the same minimization direction
        multiplier = 1 if self.satisfies_constraints(solution_data) else 100
        result = multiplier * np.sum(solution_data * self.costs, axis=1)
        return tuple(result.tolist())

    @staticmethod
    def calculate_solution_distance(sol1: MOSCPSolution, sol2: MOSCPSolution) -> float:
        """Calculates the Hamming distance (difference in selected sets) between two MO-SCP solutions."""
        return float(np.sum(sol1.data != sol2.data)) / sol1.data.size

    @staticmethod
    def load(filename: str) -> "MOSCPProblem":
        try:
            with open(filename, "r") as f:
                configuration = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Error: File not found at {filename}")
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")

        data = configuration["data"]

        num_items = data["num_items"]
        num_sets = data["num_sets"]
        coverage_data = data["sets"]
        costs = data["costs"]

        return MOSCPProblem(coverage_data, costs, num_items, num_sets)


class MOSCPProblemPymoo(ElementwiseProblem):
    def __init__(self, problem: MOSCPProblem):
        n_var = problem.num_sets
        n_obj = problem.num_objectives
        n_constr = 0

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=0.0,
            xu=1.0,
            vtype=float,
        )
        self.problem_instance = problem

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluate a single solution vector 'x' (vector of N floats).
        """
        # np.argsort returns the indices that would sort the array.
        # Basically ranking the results from 0 to N - 1
        permutation_array = np.argsort(x).astype(int)
        z1, z2 = self.problem_instance.calculate_objectives(permutation_array)
        out["F"] = np.array([z1, z2], dtype=float)
