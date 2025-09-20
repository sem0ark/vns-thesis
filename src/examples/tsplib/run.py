import argparse
import logging
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from distance_functions import get_distance_function
from parse_tsplib_problem import TSPLibParser
from utils import parse_time_string, setup_logging

from src.vns.abstract import Problem, Solution, VNSConfig
from src.vns.acceptance import AcceptBeamSkewedSmaller, AcceptBeamSmaller
from src.vns.local_search import best_improvement, first_improvement, noop
from src.vns.vns_base import VNSOptimizer

BASE = Path(__file__).parent.parent.parent / "data" / "tsplib"

logger = logging.getLogger("tsp-solver")

type TSPSolution = Solution[np.ndarray]


class TSPSol(Solution[np.ndarray]):
    def __init__(self, tour: np.ndarray, problem: "Problem"):
        super().__init__(tour, problem)

    def __repr__(self):
        return f"TSPSolution(Tour={self.data}, Objectives={self.objectives})"  # Convert back to list for representation


class TSPProblem:
    def __init__(self, cities: dict, distance_function: Callable[[Any, Any], float]):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_function = distance_function

        self.distance_matrix = self._precompute_distance_matrix()

    def _precompute_distance_matrix(self) -> np.ndarray:
        """Precomputes the distance matrix from the cities and distance function."""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = self.distance_function(
                        self.cities[i], self.cities[j]
                    )
        return matrix

    def get_distance(self, city1_idx: int, city2_idx: int) -> float:
        return self.distance_matrix[city1_idx, city2_idx]

    def generate_initial_solutions(self) -> Iterable[TSPSolution]:
        initial_tour = np.arange(self.num_cities)
        np.random.shuffle(initial_tour)
        return [TSPSol(initial_tour, self.to_problem())]

    def calculate_tour_difference_distance(
        self, sol1: Solution, sol2: Solution
    ) -> float:
        """
        Calculates a simple distance between two TSP tours (solutions).
        Counts the number of city positions that differ using NumPy.
        """
        tour1 = sol1.data
        tour2 = sol2.data
        if len(tour1) != len(tour2):
            raise ValueError(
                "Tours must have the same length for distance calculation."
            )

        diff_count = np.sum(tour1 != tour2)
        return float(diff_count)

    def evaluate(self, solution: Solution) -> tuple[float, ...]:
        tour = np.array(solution.data)  # Ensure tour is a NumPy array

        # Vectorized calculation of total tour length using NumPy
        # This leverages the precomputed distance matrix
        shifted_tour = np.roll(tour, -1)
        total_length = np.sum(self.distance_matrix[tour, shifted_tour])
        return (total_length,)

    def to_problem(self) -> Problem:
        return Problem(self.evaluate, self.generate_initial_solutions)


def shake_flip_tour_region(solution: Solution, k: int, _config: VNSConfig) -> Solution:
    tour = np.array(solution.data)
    n = len(tour)

    for _ in range(k):
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i

        tour = np.concatenate((tour[:i], tour[i : j + 1][::-1], tour[j + 1 :]))

    return solution.new(tour)  # Convert back to list for Solution


def shake_swap_tour_cities(solution: Solution, k: int, _config: VNSConfig) -> Solution:
    tour = np.array(solution.data)
    n = len(tour)

    for _ in range(k):
        idx1, idx2 = random.sample(range(n), 2)
        tour[[idx1, idx2]] = tour[[idx2, idx1]]

    return solution.new(tour)


def shake_shuffle_tour_region(
    solution: Solution, k: int, _config: VNSConfig
) -> Solution:
    tour = np.array(solution.data)
    n = len(tour)
    k = max(1, k)

    start_index = np.random.randint(n)

    region_indices = [(start_index + i) % n for i in range(k)]
    region_elements = tour[region_indices].copy()
    np.random.shuffle(region_elements)
    new_tour = tour.copy()

    for i in range(k):
        new_tour[region_indices[i]] = region_elements[i]

    return solution.new(new_tour)


def flip_op(solution: Solution, _config: VNSConfig) -> Iterable[Solution]:
    """Generates all possible Region flip neighbors for the given solution."""
    tour = np.array(solution.data)
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 1, n):
            new_tour = np.concatenate((tour[:i], tour[i : j + 1][::-1], tour[j + 1 :]))
            yield solution.new(new_tour)


def swap_op(solution: Solution, _config: VNSConfig) -> Iterable[Solution]:
    """Generates all possible Region flip neighbors for the given solution."""
    tour = np.array(solution.data)
    n = len(tour)
    for i in range(n):
        for j in range(i + 1, n):
            new_tour = tour.copy()
            new_tour[[i, j]] = new_tour[[j, i]]
            yield solution.new(new_tour)


def load_tsp_problem(filename: str) -> TSPProblem:
    """
    Loads a TSP problem from a TSPLIB file, handling both coordinate-based
    and explicit distance matrix formats.
    """
    try:
        with open(BASE / filename, "r") as f:
            file_content = f.read()
    except FileNotFoundError:
        raise ValueError(f"Error: File not found at {BASE / filename}")
    except Exception as e:
        raise ValueError(f"Error reading file {filename}: {e}")

    parser = TSPLibParser()
    try:
        parsed_data = parser.parse_string(file_content)
    except Exception as e:
        raise ValueError(f"Error parsing TSPLIB file {filename}: {e}")

    # Determine problem dimension
    expected_num_cities = parsed_data["specification"].get("DIMENSION")
    if expected_num_cities is None:
        logger.warning(
            f"DIMENSION not specified in TSPLIB file {filename}. Attempting to infer."
        )

        # Try to infer from NODE_COORDS or EDGE_WEIGHTS if DIMENSION is missing
        if parsed_data["data"].get("NODE_COORDS"):
            expected_num_cities = len(parsed_data["data"]["NODE_COORDS"])
        elif parsed_data["data"].get("EDGE_WEIGHTS") and isinstance(
            parsed_data["data"]["EDGE_WEIGHTS"], list
        ):
            # For FULL_MATRIX, dimension is len(matrix)
            # For other explicit formats, it's more complex, relying on parse_edge_weight_section to have created a square matrix
            if parsed_data["data"]["EDGE_WEIGHTS"] and isinstance(
                parsed_data["data"]["EDGE_WEIGHTS"][0], list
            ):
                expected_num_cities = len(parsed_data["data"]["EDGE_WEIGHTS"])
            else:
                raise ValueError(
                    "Could not infer DIMENSION from EDGE_WEIGHTS. Please specify DIMENSION in TSPLIB file."
                )
        else:
            raise ValueError(
                f"DIMENSION not specified and cannot be inferred from file {filename}."
            )

    if (
        expected_num_cities is None
    ):  # Should not happen if above logic is sound, but for safety
        raise ValueError(f"DIMENSION could not be determined for {filename}.")

    edge_weight_type = parsed_data["specification"].get("EDGE_WEIGHT_TYPE")

    if edge_weight_type == "EXPLICIT":
        distance_matrix = parsed_data["data"].get("EDGE_WEIGHTS")
        if not distance_matrix:
            raise ValueError(
                f"Error: EDGE_WEIGHTS section not found in {filename} while EDGE_WEIGHT_TYPE is EXPLICIT. Cannot create TSP problem."
            )

        # Ensure the matrix is 0-indexed and correct size
        if not isinstance(distance_matrix, list) or not all(
            isinstance(row, list) for row in distance_matrix
        ):
            raise ValueError(
                f"Explicit distance matrix in {filename} is not in expected list of lists format."
            )

        if len(distance_matrix) != expected_num_cities or not all(
            len(row) == expected_num_cities for row in distance_matrix
        ):
            raise ValueError(
                f"Explicit distance matrix in {filename} has incorrect dimensions. Expected {expected_num_cities}x{expected_num_cities}."
            )

        def distance_function_matrix(a, b):
            return distance_matrix[a][b]

        total_cities = len(distance_matrix)
        city_data = {i: i for i in range(total_cities)}

        return TSPProblem(city_data, distance_function_matrix)
    else:
        raw_cities_data = parsed_data["data"].get("NODE_COORDS")
        if not raw_cities_data:
            raise ValueError(
                f"Error: NODE_COORDS section not found in {filename}. Cannot create coordinate-based TSP problem."
            )

        min_node_id = min(raw_cities_data.keys())
        cities_data_0_indexed = {}
        for node_id, coords in raw_cities_data.items():
            new_index = node_id - min_node_id
            if new_index < 0:
                raise ValueError(
                    f"Error: Invalid node ID {node_id} results in negative 0-indexed key for file {filename}."
                )
            cities_data_0_indexed[new_index] = coords

        if len(cities_data_0_indexed) != expected_num_cities:
            raise ValueError(
                f"Mismatch: Parsed {len(cities_data_0_indexed)} cities from NODE_COORDS, but DIMENSION specified {expected_num_cities} for file {filename}."
            )

        if not all(i in cities_data_0_indexed for i in range(expected_num_cities)):
            raise ValueError(
                f"Error: 0-indexed city IDs are not contiguous from 0 to DIMENSION-1 for file {filename}."
            )

        distance_function = get_distance_function(edge_weight_type)

        return TSPProblem(cities_data_0_indexed, distance_function)


def prepare_optimizers(tsp_problem: TSPProblem) -> dict[str, VNSOptimizer]:
    bvns = VNSConfig(
        problem=tsp_problem.to_problem(),
        search_functions=[best_improvement(flip_op)],
        acceptance_criterion=AcceptBeamSmaller(),
        shake_function=shake_flip_tour_region,
    )
    rvns = VNSConfig(
        problem=tsp_problem.to_problem(),
        search_functions=[noop()],
        acceptance_criterion=AcceptBeamSmaller(),
        shake_function=shake_flip_tour_region,
    )
    gvns = VNSConfig(
        problem=tsp_problem.to_problem(),
        search_functions=[
            best_improvement(flip_op),
            first_improvement(swap_op),
        ],
        acceptance_criterion=AcceptBeamSmaller(),
        shake_function=shake_flip_tour_region,
    )
    svns = VNSConfig(
        problem=tsp_problem.to_problem(),
        search_functions=[best_improvement(flip_op)],
        acceptance_criterion=AcceptBeamSkewedSmaller(
            0.1, tsp_problem.calculate_tour_difference_distance, 1
        ),
        shake_function=shake_flip_tour_region,
    )

    return {
        "BVNS_BI": VNSOptimizer(
            replace(bvns, acceptance_criterion=AcceptBeamSmaller())
        ),
        "BVNS_FI": VNSOptimizer(
            replace(
                bvns,
                search_functions=[first_improvement(flip_op)],
                acceptance_criterion=AcceptBeamSmaller(),
            )
        ),
        "RVNS": VNSOptimizer(replace(rvns, acceptance_criterion=AcceptBeamSmaller())),
        "GVNS": VNSOptimizer(replace(gvns, acceptance_criterion=AcceptBeamSmaller())),
        "SVNS_BI": VNSOptimizer(
            replace(
                svns,
                acceptance_criterion=AcceptBeamSkewedSmaller(
                    0.1, tsp_problem.calculate_tour_difference_distance, 1
                ),
            )
        ),
    }


def run_vns_example_from_tsplib(
    filename: str,
    optimizer_type: str,
    run_time: str,
    max_iterations_no_improvement: int,
    optimal_value: Optional[float] = None,
):
    setup_logging(level=logging.INFO)
    max_run_time_seconds = parse_time_string(run_time)

    logger.info(
        f"--- Running {optimizer_type} Example with TSPLIB file: {filename} ---"
    )
    logger.info(f"Max run time: {run_time} ({max_run_time_seconds:.2f} seconds)")
    logger.info(f"Max iterations without improvement: {max_iterations_no_improvement}")
    if optimal_value is not None:
        logger.info(f"Optimal solution value provided: {optimal_value:.2f}")

    tsp_problem = load_tsp_problem(filename)
    optimizers = prepare_optimizers(tsp_problem)

    vns_optimizer = None
    if optimizer_type in optimizers:
        vns_optimizer = optimizers[optimizer_type]
    else:
        logger.error(f"Unknown optimizer type: {optimizer_type}")
        return

    best_objectives_data = []
    elapsed_times_data = []

    start_time = time.time()
    last_improved = 1

    for iteration, improved in enumerate(vns_optimizer.optimize(), 1):
        obj_value = vns_optimizer.config.acceptance_criterion.get_all_solutions()[
            0
        ].objectives[0]
        elapsed_time = time.time() - start_time

        if (
            len(best_objectives_data) > 2
            and best_objectives_data[-2] == best_objectives_data[-1] == obj_value
        ):
            best_objectives_data.pop()
            elapsed_times_data.pop()

        best_objectives_data.append(obj_value)
        elapsed_times_data.append(elapsed_time)

        if improved:
            logger.info("Iteration %d: Best Objective = %.2f", iteration, obj_value)

        if elapsed_time > max_run_time_seconds:
            logger.info(f"Timeout. Best Objective = {obj_value}")
            break

        if improved:
            last_improved = iteration
        elif iteration - last_improved > max_iterations_no_improvement:
            logger.info(
                f"No improvements for {max_iterations_no_improvement} iterations. Best Objective = {obj_value}"
            )
            break

    logger.info(f"--- {optimizer_type} Optimization Complete ---")
    if best_objectives_data:
        final_best_objective = min(
            best_objectives_data
        )  # Find the true best among all iterations
        logger.info(f"Overall Best Tour Length found: {final_best_objective:.2f}")
    else:
        logger.info("No solution data collected.")

    plt.figure(figsize=(10, 6))
    plt.plot(
        elapsed_times_data,
        best_objectives_data,
        marker="o",
        linestyle="-",
        markersize=4,
        label="Best Objective Found",
    )  # Plot against time

    if optimal_value is not None:
        plt.axhline(
            y=optimal_value,
            color="r",
            linestyle="--",
            label=f"Optimal Value ({optimal_value:.2f})",
        )
        plt.title(
            f"{optimizer_type} Optimization Progress for {Path(filename).name} (Optimal: {optimal_value:.2f})"
        )
    else:
        plt.title(f"{optimizer_type} Optimization Progress for {Path(filename).name}")

    plt.xlabel("Time Elapsed (seconds)")
    plt.ylabel("Best Objective Value (Tour Length)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_vns_optimizers(
    filename: str,
    run_time: str,
    max_iterations_no_improvement: int,
    optimal_value: Optional[float] = None,
):
    setup_logging(level=logging.INFO)
    max_run_time_seconds = parse_time_string(run_time)

    logger.info(f"--- Running optimizers with TSPLIB file: {filename} ---")
    logger.info(f"Max run time: {run_time} ({max_run_time_seconds:.2f} seconds)")
    logger.info(f"Max iterations without improvement: {max_iterations_no_improvement}")
    if optimal_value is not None:
        logger.info(f"Optimal solution value provided: {optimal_value:.2f}")

    tsp_problem = load_tsp_problem(filename)
    optimizers = prepare_optimizers(tsp_problem)

    for optimizer_type, vns_optimizer in optimizers.items():
        logger.info(
            f"Starting {optimizer_type} for TSP with {tsp_problem.num_cities} cities."
        )
        best_objectives_data = []
        elapsed_times_data = []

        start_time = time.time()
        last_improved = 1

        for iteration, improved in enumerate(vns_optimizer.optimize(), start=1):
            obj_value = vns_optimizer.config.acceptance_criterion.get_all_solutions()[
                0
            ].objectives[0]
            elapsed_time = time.time() - start_time

            if (
                len(best_objectives_data) > 2
                and best_objectives_data[-2] == best_objectives_data[-1] == obj_value
            ):
                best_objectives_data.pop()
                elapsed_times_data.pop()

            best_objectives_data.append(obj_value)
            elapsed_times_data.append(elapsed_time)

            if elapsed_time > max_run_time_seconds:
                logger.info(f"Timeout. Best Objective = {obj_value}")
                break

            if improved:
                logger.info("Iteration %d: Best Objective = %.2f", iteration, obj_value)
                last_improved = iteration
            elif iteration - last_improved > max_iterations_no_improvement:
                logger.info(
                    f"No improvements for {max_iterations_no_improvement} iterations. Best Objective = {obj_value}"
                )
                break

        logger.info(f"--- {optimizer_type} Optimization Complete ---")
        if best_objectives_data:
            final_best_objective = min(best_objectives_data)
            logger.info(f"Overall Best Tour Length found: {final_best_objective:.2f}")
        else:
            logger.info("No solution data collected.")

        plt.plot(
            elapsed_times_data,
            best_objectives_data,
            marker="o",
            linestyle="-",
            markersize=4,
            label=optimizer_type,
        )

    if optimal_value is not None:
        plt.axhline(
            y=optimal_value,
            color="r",
            linestyle="--",
            label=f"Optimal Value ({optimal_value:.2f})",
        )
        plt.title(
            f"Optimization Progress for {Path(filename).name} (Optimal: {optimal_value:.2f})"
        )
    else:
        plt.title(f"Optimization Progress for {Path(filename).name}")

    plt.xlabel("Time Elapsed (seconds)")
    plt.ylabel("Best Objective Value (Tour Length)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VNS optimization for TSP using a TSPLIB file."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Path to the TSPLIB .tsp file (e.g., data/linho_100_tsp.tsp)",
    )
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="SkewedVNS",
        help="Type of VNS optimizer to run.",
    )
    parser.add_argument(
        "--run-time",
        type=str,
        default="10s",
        help="Maximum duration for the script to run (e.g., '30s', '5m', '1h').",
    )
    parser.add_argument(
        "--max-no-improvements",
        type=int,
        default=sys.maxsize,
        help="Secondary limit: Maximum number of VNS iterations without any improvements.",
    )
    parser.add_argument(
        "--optimal-value",
        type=float,
        help="Optional: Known optimal solution value for comparison in plot.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
    )

    args = parser.parse_args()

    if args.compare:
        compare_vns_optimizers(
            args.filename,
            args.run_time,
            args.max_no_improvements,
            args.optimal_value,
        )

    else:
        run_vns_example_from_tsplib(
            args.filename,
            args.optimizer_type,
            args.run_time,
            args.max_no_improvements,
            args.optimal_value,
        )
