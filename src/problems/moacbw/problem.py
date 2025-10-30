import json
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import xxhash
from pymoo.core.problem import ElementwiseProblem

from src.cli.utils import lru_cache_custom_hash_key
from src.core.abstract import Problem, Solution

type MOACBWSolution = Solution[np.ndarray]


def get_hash(*args):
    data: np.ndarray = args[-1]
    h = xxhash.xxh64()
    h.update(data.tobytes())
    return h.intdigest()


class _MOACBWSolution(Solution[np.ndarray]):
    """
    Represents a solution for the MO-ABCW problem.
    data: A 1D numpy array representing the permutation of vertex indices (0 to N-1).
    The position in the array determines the position of a given node (0 to N-1).
    """

    def get_hash(self) -> int:
        return hash(self.objectives)

    def to_json_serializable(self) -> Any:
        return self.data.tolist()

    @staticmethod
    def from_json_serializable(
        problem: Problem[np.ndarray], serialized_data: list[int]
    ) -> MOACBWSolution:
        return _MOACBWSolution(np.array(serialized_data), problem)


class MOACBWProblem(Problem[np.ndarray]):
    """
    Implementation of the Multi-objective Antibandwidth-Cutwidth (MO-ABCW) problem.

    Objectives:
    1. Antibandwidth (z1): MAXIMIZE -> Implemented as Minimize (-z1)
    2. Cutwidth (z2): MINIMIZE -> Implemented as Minimize (z2)
    """

    def __init__(
        self,
        num_nodes: int,
        # Adjacency list: [(u, [v1, v2, ...]), ...] where u and v are 0-based integers.
        graph_adj_list: list[tuple[int, list[int]]],
    ):
        super().__init__(
            num_constraints=0,
            num_objectives=2,
            num_variables=num_nodes,
            objective_names=["Antibandwidth", "Cutwidth"],
            problem_name="MOACBW",
        )

        self.num_nodes = num_nodes
        self.num_objectives = 2

        # 0-based adjacency list
        self.adj_list: Dict[int, List[int]] = {}
        for u, neighbors in graph_adj_list:
            # Filter self-loops
            self.adj_list[u] = [v for v in neighbors if v != u]

        # Ensure all nodes are present in the keys (even isolated ones)
        for i in range(self.num_nodes):
            if i not in self.adj_list:
                self.adj_list[i] = []

    def get_initial_solutions(
        self, num_solutions: int = 1000
    ) -> Iterable[MOACBWSolution]:
        """Generates a specified number of random permutations (layouts)."""
        solutions = []
        solutions.append(_MOACBWSolution(np.arange(self.num_nodes, dtype=int), self))
        for _ in range(num_solutions):
            # Solution is a permutation of vertex indices [0, 1, ..., N-1]
            solution_data = np.arange(self.num_nodes, dtype=int)
            np.random.shuffle(solution_data)
            solutions.append(_MOACBWSolution(solution_data, self))
        return solutions

    @lru_cache_custom_hash_key(maxsize=500, key_func=get_hash)
    def get_antibandwidth_values(self, solution_data: np.ndarray) -> list[int]:
        n = self.num_nodes
        positions = np.empty(n, dtype=int)
        positions[solution_data] = np.arange(n)

        antibandwidth_values = [0] * n

        for node in range(n):
            neighbors = self.adj_list[node]
            if not neighbors:
                continue

            node_pos = positions[node]
            neighbor_positions = positions[neighbors]

            # AB(pi, u) = min_{v in N(u)} { |pi(u) - pi(v)| }
            # Vectorized calculation with early termination check
            distances = np.abs(node_pos - neighbor_positions)
            min_dist = np.min(distances)

            antibandwidth_values[positions[node]] = int(min_dist)

        return antibandwidth_values

    @lru_cache_custom_hash_key(maxsize=500, key_func=get_hash)
    def get_cutwidth_values(self, solution_data: np.ndarray) -> list[int]:
        n = self.num_nodes
        positions = np.empty(n, dtype=int)
        positions[solution_data] = np.arange(n)

        diff = [0] * (n + 1)
        for u in range(n):
            u_pos = positions[u]
            for v in self.adj_list[u]:
                if u >= v:
                    continue

                v_pos = positions[v]
                left_pos = min(u_pos, v_pos)
                right_pos = max(u_pos, v_pos)

                diff[left_pos] += 1
                diff[right_pos] -= 1

        cuts = [0] * (n - 1)
        current = 0
        for i in range(n - 1):
            current += diff[i]
            cuts[i] = current

        return cuts

    def calculate_objectives(self, solution_data: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the two objective function values (z1 and z2).
        """
        if self.num_nodes == 0:
            return 0.0, 0.0

        # Maximize z1 = min_{v in V} { min_{(u,v) in E} { |pi(u) - pi(v)| } }
        z1 = -float(min(self.get_antibandwidth_values(solution_data)))

        # Minimize z2 = max_{k=1}^{n-1} C(k)
        # C(k): cut size after the vertex with position k (i.e., between k and k+1)
        z2 = float(max(self.get_cutwidth_values(solution_data)))

        return z1, z2

    def calculate_objectives_max(
        self, solution_data: np.ndarray
    ) -> Tuple[float, float]:
        if self.num_nodes == 0:
            return 0.0, 0.0
        z1 = -float(max(self.get_antibandwidth_values(solution_data)))
        z2 = float(min(self.get_cutwidth_values(solution_data)))
        return z1, z2

    def calculate_objectives_sum(
        self, solution_data: np.ndarray
    ) -> Tuple[float, float]:
        if self.num_nodes == 0:
            return 0.0, 0.0
        z1 = -float(sum(self.get_antibandwidth_values(solution_data)))
        z2 = float(sum(self.get_cutwidth_values(solution_data)))
        return z1, z2

    def satisfies_constraints(self, solution_data: np.ndarray) -> bool:
        return True

    def load_solution(self, saved_solution_data) -> MOACBWSolution:
        return _MOACBWSolution.from_json_serializable(self, saved_solution_data)

    @staticmethod
    def calculate_solution_distance(
        sol1: MOACBWSolution, sol2: MOACBWSolution
    ) -> float:
        """
        Calculates the distance between two permutations using normalized Hamming distance.
        """
        if sol1.data.size == 0:
            return 0.0

        # Number of positions where the two permutations differ
        disagreements = np.sum(sol1.data != sol2.data)

        # Normalized distance: [0, 1]
        return float(disagreements) / sol1.data.size

    @staticmethod
    def load(filename: str) -> "MOACBWProblem":
        """
        Load JSON-formatted MO-ACBW instance.
        Expects to have two main elements defined under "data":
        - nodes -> total number of nodes in the graph
        - graph -> adjacency list like structure of form list[tuple[node_number: int, neighbor_numbers: list[int]]]
        """
        try:
            with open(filename, "r") as f:
                configuration = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Error: File not found at {filename}")
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")

        nodes = configuration["data"]["nodes"]
        graph = configuration["data"]["graph"]

        return MOACBWProblem(nodes, graph)


class MOACBWProblemPymoo(ElementwiseProblem):
    def __init__(self, problem: MOACBWProblem):
        n_var = problem.num_nodes
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

    def _evaluate(self, x: np.ndarray, out: dict, *args: Any, **kwargs: Any):
        """
        Evaluate a single solution vector 'x' (vector of N floats).
        """
        # np.argsort returns the indices that would sort the array.
        # Basically ranking the results from 0 to N - 1
        permutation_array = np.argsort(x).astype(int)
        z1, z2 = _MOACBWSolution(permutation_array, self.problem_instance).objectives
        out["F"] = np.array([z1, z2], dtype=float)
