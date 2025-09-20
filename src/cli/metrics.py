from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import tabulate
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.cli.shared import SavedRun


@dataclass
class Metrics:
    epsilon: float
    hypervolume: float
    r_metric: float
    inverted_generational_distance: float


def load_instance_data_json(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}

    with open(file_path, "r") as f:
        return json.load(f)


def calculate_reference_front(
    runs: list[SavedRun], predefined_front: list | None = None
) -> np.ndarray:
    """
    Calculates the reference front by combining and sorting all non-dominated
    solutions from all runs. It can be overridden by a predefined front.

    Args:
        runs: A list of SavedRun objects.
        predefined_front: A list of objective vectors to use as the reference front.

    Returns:
        A NumPy array representing the non-dominated reference front.
    """
    if predefined_front:
        print("Using predefined reference front!")
        return np.array(predefined_front)

    print("Making reference front from merging all available runs.")
    all_objectives = []

    for run in runs:
        for sol in run.solutions:
            all_objectives.append(sol.objectives)

    if not all_objectives:
        logging.warning("No solutions found in any run to create a reference front.")
        return np.array([])

    combined_objectives = np.array(all_objectives)

    # Negate objectives to convert maximization to minimization
    negated_objectives = -combined_objectives

    nd_sorting = NonDominatedSorting()
    non_dominated_indices = nd_sorting.do(
        negated_objectives, only_non_dominated_front=True
    )

    # Return the non-dominated front with original (non-negated) values
    reference_front = combined_objectives[non_dominated_indices]

    # Remove duplicates from the reference front
    reference_front = np.unique(reference_front, axis=0)

    return reference_front


def calculate_multiplicative_epsilon(A: np.ndarray, R: np.ndarray) -> float:
    """
    Calculates the multiplicative epsilon indicator without using pymoo.

    Args:
        A: The approximation front (a NumPy array).
        R: The reference front (a NumPy array).

    Returns:
        The multiplicative epsilon indicator value. For minimization problems,
        a value closer to 1 is better.
    """
    if A.size == 0 or R.size == 0:
        return np.nan

    # Pymoo's Epsilon indicator is for minimization. The formula for the multiplicative
    # epsilon for minimization is: max_{a in A} min_{r in R} max_i(a_i / r_i).
    # Since the user's problem is maximization and objectives are positive, we apply this
    # to the negated fronts.
    negated_A = -A
    negated_R = -R

    ratios = np.zeros((negated_A.shape[0], negated_R.shape[0]))

    for i in range(negated_A.shape[0]):
        for j in range(negated_R.shape[0]):
            ratios[i, j] = np.max(negated_A[i, :] / negated_R[j, :])

    return np.max(np.min(ratios, axis=1))


def calculate_r2_metric(
    front: np.ndarray, ideal_point: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculates the R2 unary indicator based on the weighted Tchebycheff utility function.

    Args:
        front: A NumPy array of objective vectors (solutions).
        ideal_point: The ideal point as a NumPy array.
        weights: A NumPy array of weight vectors.

    Returns:
        The R2 indicator value.
    """
    if front.size == 0 or weights.size == 0:
        return np.nan

    # The formula for R2 is for minimization, so we use negated objectives
    negated_front = -front
    negated_ideal_point = -ideal_point

    tchebycheff_values = np.max(
        weights * np.abs(negated_ideal_point - negated_front[:, None, :]), axis=2
    )
    min_utilities = np.min(tchebycheff_values, axis=1)

    return np.mean(min_utilities)


def _get_hypervolume_ref_point(fronts: list[np.ndarray]) -> np.ndarray:
    """
    Determines a suitable reference point for Hypervolume calculation.

    For maximization, the reference point must be worse than all points in the combined fronts.
    We find the minimum value for each objective across all fronts and set the reference
    point slightly below that.
    """
    if not fronts:
        return np.array([])

    all_points = np.concatenate(fronts, axis=0)
    min_objectives = np.min(all_points, axis=0)
    return min_objectives - 1e-6  # A small epsilon to ensure all points are dominated


def _generate_uniform_weights(num_objectives: int, num_weights: int) -> np.ndarray:
    """
    Generates uniformly distributed weight vectors for R2 metric.
    """
    if num_objectives == 2:
        weights = np.zeros((num_weights, 2))
        for i in range(num_weights):
            weights[i, 0] = i / (num_weights - 1)
            weights[i, 1] = 1.0 - weights[i, 0]
        return weights
    else:
        # A simple approximation for more than 2 objectives
        weights = np.random.rand(num_weights, num_objectives)
        return weights / np.sum(weights, axis=1, keepdims=True)


def calculate_metrics(
    instance_path: Path, runs_grouped: dict[str, list[SavedRun]]
) -> dict[str, Metrics]:
    """
    Calculates performance metrics for each run, and then averages them across multiple runs
    for the same configuration.

    Args:
        instance_path: The path to the problem instance file.
        runs_grouped: A dictionary where keys are run names and values are lists of SavedRun objects.

    Returns:
        A dictionary where keys are run names and values are a single Metrics object
        with the averaged metric values.
    """
    problem_data = load_instance_data_json(instance_path)

    # Combine all solutions to create a single, shared reference front
    all_runs = [run for runs in runs_grouped.values() for run in runs]
    reference_front = calculate_reference_front(
        all_runs, problem_data.get("reference_front")
    )

    if reference_front.size == 0:
        logging.error(
            "Failed to create a reference front from the provided runs or file."
        )
        return {}

    num_objectives = reference_front.shape[1]

    # Calculate a reference point for Hypervolume (worst point)
    hv_ref_point = _get_hypervolume_ref_point([reference_front])

    # Calculate an ideal point for R2 (best point)
    # The ideal point for a maximization problem is the maximum value for each objective.
    r2_ideal_point = np.max(reference_front, axis=0)
    r2_weights = _generate_uniform_weights(num_objectives, num_weights=100)

    metrics_results = {}

    for run_name, runs in runs_grouped.items():
        run_metrics_list = []
        for run in runs:
            front = np.array([sol.objectives for sol in run.solutions])

            if front.size == 0:
                logging.warning(
                    f"Run '{run_name}' has an empty front. Skipping metrics calculation."
                )
                run_metrics_list.append(
                    Metrics(
                        epsilon=np.nan,
                        hypervolume=np.nan,
                        r_metric=np.nan,
                        inverted_generational_distance=np.nan,
                    )
                )
                continue

            # For all pymoo metrics, we negate the fronts to convert to a minimization problem
            negated_front = -front
            negated_reference_front = -reference_front
            negated_hv_ref_point = -hv_ref_point

            # Hypervolume: The original hypervolume is maximized, but this implementation
            # calculates the hypervolume of the dominated space. For minimization, a larger
            # dominated space is better, so the result is better for better fronts.
            hv_indicator = HV(ref_point=negated_hv_ref_point)
            hypervolume = hv_indicator.do(negated_front)

            # Epsilon: The user requested a custom implementation.
            epsilon = calculate_multiplicative_epsilon(front, reference_front)

            # R2 Unary Indicator: For minimization, smaller is better.
            r_metric = calculate_r2_metric(front, r2_ideal_point, r2_weights)

            # Inverted Generational Distance (IGD): For minimization, smaller is better.
            igd_indicator = IGD(negated_reference_front)
            inverted_generational_distance = igd_indicator.do(negated_front)

            run_metrics_list.append(
                Metrics(
                    epsilon=epsilon,
                    hypervolume=hypervolume or float("inf"),
                    r_metric=r_metric,
                    inverted_generational_distance=inverted_generational_distance
                    or float("inf"),
                )
            )

        # Average the metrics for each run group
        if run_metrics_list:
            avg_epsilon = np.nanmean([m.epsilon for m in run_metrics_list])
            avg_hypervolume = np.nanmean([m.hypervolume for m in run_metrics_list])
            avg_r_metric = np.nanmean([m.r_metric for m in run_metrics_list])
            avg_igd = np.nanmean(
                [m.inverted_generational_distance for m in run_metrics_list]
            )

            metrics_results[run_name] = Metrics(
                epsilon=float(avg_epsilon),
                hypervolume=float(avg_hypervolume),
                r_metric=float(avg_r_metric),
                inverted_generational_distance=float(avg_igd),
            )

    return metrics_results


def display_metrics(metrics: dict[str, Metrics]) -> None:
    """
    Displays the calculated metrics in a sorted console table.
    """
    # Sort the metrics. Lower values are better for all of them.
    sorted_metrics = sorted(
        metrics.items(),
        key=lambda item: (
            item[1].inverted_generational_distance,
            item[1].epsilon,
            item[1].hypervolume,
            item[1].r_metric,
        ),
    )

    headers = [
        "Config",
        "Epsilon",
        "Hypervolume",
        "R-Metric",
        "IGD",
    ]

    table_data = []
    for run_name, metric_values in sorted_metrics:
        row = [
            run_name,
            f"{metric_values.epsilon:.4f}"
            if not np.isnan(metric_values.epsilon)
            else "N/A",
            f"{metric_values.hypervolume:.4f}"
            if not np.isnan(metric_values.hypervolume)
            else "N/A",
            f"{metric_values.r_metric:.4f}"
            if not np.isnan(metric_values.r_metric)
            else "N/A",
            f"{metric_values.inverted_generational_distance:.4f}"
            if not np.isnan(metric_values.inverted_generational_distance)
            else "N/A",
        ]
        table_data.append(row)

    print(tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
