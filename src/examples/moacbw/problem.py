import json
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np
import xxhash
from pymoo.core.problem import ElementwiseProblem

from src.vns.abstract import Problem, Solution

type MOACBWSolution = Solution[np.ndarray]


class _MOACBWSolution(Solution[np.ndarray]):
    """
    Represents a solution for the MO-ABCW problem.
    data: A 1D numpy array representing the permutation of vertex indices (0 to N-1).
    The position in the array determines the position of a given node (0 to N-1).
    """

    def equals(self, other: Solution[np.ndarray]) -> bool:
        return hash(self) == hash(other)

    def get_hash(self) -> int:
        h = xxhash.xxh64()
        h.update(self.data.tobytes())
        return h.intdigest()

    @property
    def node_positions(self):
        positions = np.zeros(self.data.size, dtype=int)
        for pos, v in enumerate(self.data):
            positions[v] = pos
        return positions


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
        super().__init__(self.evaluate, self.generate_initial_solutions)

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

    def generate_initial_solutions(
        self, num_solutions: int = 50
    ) -> Iterable[MOACBWSolution]:
        """Generates a specified number of random permutations (layouts)."""
        solutions = []
        for _ in range(num_solutions):
            # Solution is a permutation of vertex indices [0, 1, ..., N-1]
            solution_data = np.arange(self.num_nodes, dtype=int)
            np.random.shuffle(solution_data)
            solutions.append(_MOACBWSolution(solution_data, self))
        return solutions

    def get_antibandwidth(self, solution: MOACBWSolution) -> int:
        """
        Antibandwidth = min_{v in V} { min_{(u,v) in E} { |pi(u) - pi(v)| } }
        """
        sol: _MOACBWSolution = cast(Any, solution)
        positions = sol.node_positions

        antibandwidth_values = []

        for current in range(self.num_nodes):
            neighbors = self.adj_list.get(current, [])

            if not neighbors:
                continue

            current_position = positions[current]

            # AB(pi, u) = min_{v in N(u)} { |pi(u) - pi(v)| }
            neighbor_positions = positions[neighbors]
            antibandwidth_values.append(
                np.min(np.abs(current_position - neighbor_positions))
            )

        return min(antibandwidth_values) + sum(antibandwidth_values) / self.num_nodes

    def get_cutwidth(self, solution: MOACBWSolution) -> int:
        # We can calculate it using a more optimized version of:
        # for i in [0, N-1] Compute max of:
        #    cut_edges = 0
        #    set_left = set(ordering[:i+1])
        #    set_right = set(ordering[i+1:])
        #    for u in set_left:
        #        for neighbor in graph[u]: # Assuming adjacency list
        #            if neighbor in set_right:
        #                cut_edges += 1

        sol: _MOACBWSolution = cast(Any, solution)
        permutation = sol.data
        positions = sol.node_positions

        current_cut = 0  # Represents the cut size C(k)
        cuts = []

        # We iterate over the N-1 cuts. k is the position index (0 to N-2).
        # The vertex v = permutation[k] moves from the right set (R) to the left set (L),
        # defining the cut C(k+1).
        for cut_position in range(self.num_nodes - 1):
            current = permutation[cut_position]  # Vertex with position k+1 is moving.

            # Change in "cut size" when V moves from R to L
            delta_cut = 0

            for w in self.adj_list.get(current, []):
                position_w = positions[w]

                # Check neighbor's position relative to the old left set (position <= k)
                if position_w <= (cut_position + 1):
                    # w is in the new L (position 1 to k+1).
                    # If position_w <= k: Edge (v, w) stops crossing (R->L to L->L). Delta -1.
                    if position_w <= cut_position:
                        delta_cut -= 1
                    else:
                        pass

                # Check neighbor's position relative to the new right set (position > k+1)
                elif position_w > (cut_position + 1):
                    # Case 2: w is still in the new R (position k+2 to N).
                    # Edge (v, w) starts crossing (R->R to L->R). Delta +1.
                    delta_cut += 1

            # Update the cut size for C(k+1)
            current_cut += delta_cut

            # C(1) to C(N-1) are the relevant cuts
            cuts.append(current_cut)

        return max(cuts) + sum(cuts) / self.num_nodes

    def evaluate(self, solution: MOACBWSolution) -> Tuple[float, float]:
        """
        Calculates the two objective function values (z1 and z2).
        """
        if self.num_nodes == 0:
            return 0.0, 0.0

        # Maximize z1 = min_{v in V} { min_{(u,v) in E} { |pi(u) - pi(v)| } }
        z1 = -float(self.get_antibandwidth(solution))

        # Minimize z2 = max_{k=1}^{n-1} C(k)
        # C(k): cut size after the vertex with position k (i.e., between k and k+1)
        z2 = float(self.get_cutwidth(solution))

        return z1, z2

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
