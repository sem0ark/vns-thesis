import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import tabulate

from dataclasses import dataclass
from matplotlib.transforms import Bbox
from pathlib import Path
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from src.cli.shared import SavedRun
from typing import Any, Dict, List, Tuple


def load_instance_data_json(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}

    with open(file_path, "r") as f:
        return json.load(f)

@dataclass
class Metrics:
    epsilon: float
    hypervolume: float
    r_metric: float
    inverted_generational_distance: float


def flip_objectives_to_positive(front: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Flips the sign of any column in a NumPy array if all values in that column are negative.
    This standardization step is often required for metrics like Hypervolume and Epsilon
    which require positive objective values.

    Returns the modified array and a list of indices of the flipped columns.
    """
    flipped_indices = []
    # Make a copy to avoid modifying the original array
    front_copy = front.copy()
    for i in range(front_copy.shape[1]):
        if np.all(front_copy[:, i] < 0):
            front_copy[:, i] *= -1
            flipped_indices.append(i)
    return front_copy, flipped_indices


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
        print("Got a predefined reference front!")
        predefined_front_np, _ = flip_objectives_to_positive(np.array(predefined_front))
        return -predefined_front_np

    print("Making reference front from merging all available runs.")
    all_objectives = []

    for run in runs:
        for sol in run.solutions:
            all_objectives.append(sol.objectives)

    if not all_objectives:
        logging.warning("No solutions found in any run to create a reference front.")
        return np.array([])

    combined_objectives = np.array(all_objectives)
    nd_sorting = NonDominatedSorting()
    # NonDominatedSorting assumes minimization, which is what we want.
    non_dominated_indices = nd_sorting.do(
        combined_objectives, only_non_dominated_front=True
    )

    reference_front = combined_objectives[non_dominated_indices]
    reference_front = np.unique(reference_front, axis=0)

    return reference_front


def calculate_multiplicative_epsilon(A: np.ndarray, R: np.ndarray) -> float:
    """
    Calculates the multiplicative epsilon indicator.

    This indicator is for minimization, where a value closer to 1 is better,
    and a value of 1 means A is covered by R.
    The formula is: max_{r in R} min_{a in A} max_i(a_i / r_i).

    Args:
        A: The approximation front (a NumPy array).
        R: The reference front (a NumPy array).

    Returns:
        The multiplicative epsilon indicator value.
    """
    if A.size == 0 or R.size == 0:
        return np.nan

    # A (approximation) dominates R (reference) if the ratio is close to 1
    # The calculation is for E(A, R), which is the minimum factor to multiply R by
    # to cover A. This is the definition for minimization when objectives are positive.
    ratios = np.zeros((R.shape[0], A.shape[0]))

    for j in range(R.shape[0]):
        for i in range(A.shape[0]):
            ratios[j, i] = np.max(R[j, :] / A[i, :])

    return np.max(np.min(ratios, axis=1))


def calculate_r2_metric(
    front: np.ndarray, ideal_point: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculates the R2 unary indicator based on the weighted Tchebycheff utility function.
    This function is now explicitly implemented for a minimization problem.

    Args:
        front: A NumPy array of objective vectors (solutions).
        ideal_point: The ideal point (Z_ID) as a NumPy array.
        weights: A NumPy array of weight vectors (reference vectors).

    Returns:
        The R2 indicator value.
    """
    if front.size == 0 or weights.size == 0:
        return np.nan

    # R2 is defined based on the Tchebycheff utility function, U(z, lambda)
    # The term to minimize is U(z, lambda) = max_i(lambda_i * |z_i - z_ID_i|)

    # The ideal point (Z_ID) is broadcast against the front (z) and weights (lambda)
    # front[:, None, :] -> (num_solutions, 1, num_objectives)
    # ideal_point[None, None, :] -> (1, 1, num_objectives)
    # weights[None, :, :] -> (1, num_weights, num_objectives)

    # Calculate |z_i - z_ID_i|
    difference_abs = np.abs(front[:, None, :] - ideal_point[None, None, :])

    # Calculate lambda_i * |z_i - z_ID_i|
    weighted_difference = weights[None, :, :] * difference_abs

    # Calculate U(z, lambda) = max_i(...) (Tchebycheff utility)
    tchebycheff_values = np.max(weighted_difference, axis=2)
    # tchebycheff_values shape: (num_solutions, num_weights)

    # For each weight vector, find the minimum utility across all solutions in the front:
    # min_{z in front} U(z, lambda)
    min_utilities_per_weight = np.min(tchebycheff_values, axis=0)
    # min_utilities_per_weight shape: (num_weights,)

    # The R2 metric is the mean of these minimum utilities:
    # R2 = (1 / num_weights) * sum_lambda ( min_{z in front} U(z, lambda) )
    return np.mean(min_utilities_per_weight)


def _get_hypervolume_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    """
    Determines a suitable reference point for Hypervolume calculation.
    """
    if not fronts:
        return np.array([])

    all_points = np.concatenate(fronts, axis=0)
    # Get the maximum value for each objective across all fronts
    max_objectives = np.max(all_points, axis=0)
    
    # The reference point is set slightly worse (larger) than the maximum observed values.
    # This is correct for minimization.
    return max_objectives + 1e-6


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
    instance_path: Path,
    runs_grouped: dict[str, list[SavedRun]],
    max_time_seconds: float,
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

    all_runs = [run for runs in runs_grouped.values() for run in runs]
    reference_front = calculate_reference_front(
        all_runs, problem_data.get("reference_front")
    )

    if reference_front.size == 0:
        return {}

    if reference_front.size == 0:
        logging.error(
            "Failed to create a reference front from the provided runs or file."
        )
        return {}

    num_objectives = reference_front.shape[1]
    hypervolume_reference_point = _get_hypervolume_reference_point([reference_front])
    reference_front_hypervolume = HV(ref_point=hypervolume_reference_point).do(reference_front) or np.nan

    # Calculate an ideal point for R2 (best point)
    r2_ideal_point = np.min(reference_front, axis=0)
    r2_weights = _generate_uniform_weights(num_objectives, num_weights=100)

    metrics_results = {}

    for run_name, runs in runs_grouped.items():
        run_metrics_list = []
        for run in runs:
            if abs(run.metadata.run_time_seconds - max_time_seconds) > 1e-3:
                continue

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

            hypervolume_indicator = HV(ref_point=hypervolume_reference_point)
            relative_hypervolume = (reference_front_hypervolume - (hypervolume_indicator.do(front) or np.nan)) / reference_front_hypervolume

            # Epsilon: The user requested a custom implementation.
            epsilon = calculate_multiplicative_epsilon(front, reference_front)

            # R2 Unary Indicator: For minimization, smaller is better.
            r_metric = calculate_r2_metric(front, r2_ideal_point, r2_weights)

            # Inverted Generational Distance (IGD): For minimization, smaller is better.
            igd_indicator = IGD(reference_front)
            inverted_generational_distance = igd_indicator.do(front)

            run_metrics_list.append(
                Metrics(
                    epsilon=epsilon,
                    hypervolume=relative_hypervolume,
                    r_metric=r_metric,
                    inverted_generational_distance=0 if epsilon == 1 else (inverted_generational_distance or np.nan),
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
        "Epsilon (multiplicative)",
        "Hypervolume (Relative loss)",
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


def plot_runs(
    instance_path: Path,
    runs_grouped: Dict[str, List[SavedRun]],
    max_time_seconds: float,
) -> None:
    print("""Plotting the results:
Controls:
Press 'h' to hide all graphs except reference front.
Use arrow keys to move legend around.
You can also click on graphs in legend to show/hide any specific one.
""")

    problem_data = load_instance_data_json(instance_path)

    all_runs = [run for runs in runs_grouped.values() for run in runs]
    reference_front = calculate_reference_front(
        all_runs, problem_data.get("reference_front")
    )

    runs_grouped = {
        # take latest run with correct max run time.
        name: [sorted([run for run in runs if abs(run.metadata.run_time_seconds - max_time_seconds) < 1e-3], key=lambda run: run.metadata.date)[-1]]
        for name, runs in runs_grouped.items()
    }

    if reference_front.size == 0:
        return

    all_flipped_indices = set()
    reference_front, flipped_indices = flip_objectives_to_positive(reference_front)
    all_flipped_indices.update(flipped_indices)

    if reference_front.size == 0:
        logging.error(
            "Failed to create a reference front from the provided runs or file."
        )
        return

    if reference_front.shape[1] != 2:
        logging.error(
            "Plotting is only supported for problems with exactly 2 objectives."
        )
        return

    # Calculate hypervolume for each run and prepare data for plotting
    plot_data = []

    hypervolume_indicator = HV(ref_point=(0, 0))

    for run_name, runs in runs_grouped.items():
        merged_front_objectives = sorted(
            [tuple(sol.objectives) for sol in runs[0].solutions]
        )

        if not merged_front_objectives:
            continue

        combined_objectives = np.array(merged_front_objectives)

        nd_sorting = NonDominatedSorting()
        non_dominated_indices = nd_sorting.do(
            combined_objectives, only_non_dominated_front=True
        )
        merged_front = combined_objectives[non_dominated_indices]

        if merged_front.size > 0:
            hypervolume = hypervolume_indicator.do(merged_front)
        else:
            hypervolume = -np.inf

        merged_front, flipped_indices = flip_objectives_to_positive(merged_front)
        all_flipped_indices.update(flipped_indices)

        plot_data.append(
            {"name": run_name, "front": merged_front, "hypervolume": hypervolume}
        )

    sorted_plot_data = sorted(plot_data, key=lambda d: d["hypervolume"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(right=0.7)

    lines_dict = {}

    for data in sorted_plot_data:
        run_name = data["name"]
        merged_front = data["front"]

        (line,) = ax.plot(
            merged_front[:, 0],
            merged_front[:, 1],
            marker="o",
            linestyle="-",
            label=run_name,
            alpha=0.6,
        )
        lines_dict[run_name] = line

    # Plot the reference front
    (ref_line,) = ax.plot(
        reference_front[:, 0],
        reference_front[:, 1],
        linestyle="-",
        label="Reference Front",
        color="red",
        markersize=8,
        fillstyle="none",
        linewidth=2,
    )
    lines_dict["Reference Front"] = ref_line

    if all_runs:
        metadata = all_runs[0].metadata
        title_str = f"{metadata.problem_name.upper()}, {metadata.instance_name}, {metadata.run_time_seconds}s"

        if all_flipped_indices:
            flipped_obj_labels = sorted([f"Z{i + 1}" for i in all_flipped_indices])
            note = f" (Note: {', '.join(flipped_obj_labels)} were negated)"
            title_str += note

        ax.set_title(title_str)
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")

    ax.grid(True)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0, 0.07, 1))
    legend_lines = {}
    for line, text in zip(legend.get_lines(), legend.get_texts()):
        line.set_picker(True)
        line.set_pickradius(10)
        run_name = text.get_text()
        legend_lines[line] = lines_dict[run_name]

    # Define the scroll function
    def on_scroll(event):
        if legend.contains(event):
            bbox = legend.get_bbox_to_anchor()
            d = {"down": 30, "up": -30}
            bbox = Bbox.from_bounds(
                bbox.x0, bbox.y0 + d.get(event.button, 0), bbox.width, bbox.height
            )
            tr = legend.axes.transAxes.inverted()
            legend.set_bbox_to_anchor(bbox.transformed(tr))
            fig.canvas.draw_idle()

    # Define the pick function for toggling visibility
    def on_pick(event):
        legend_line = event.artist
        plotted_line = legend_lines[legend_line]
        visible = not plotted_line.get_visible()
        plotted_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.1)
        fig.canvas.draw_idle()

    def on_key_press(event):
        directions = {
            "up": [0, -50],
            "down": [0, 50],
            "left": [-50, 0],
            "right": [50, 0],
        }

        if event.key == "h":
            for line_name, line in lines_dict.items():
                if line_name != "Reference Front":
                    line.set_visible(False)

            for leg_line, plotted_line in legend_lines.items():
                if not plotted_line.get_visible():
                    leg_line.set_alpha(0.1)

            fig.canvas.draw_idle()
        elif event.key in directions:
            x, y = directions[event.key]
            bbox = legend.get_bbox_to_anchor()
            bbox = Bbox.from_bounds(bbox.x0 + x, bbox.y0 + y, bbox.width, bbox.height)
            tr = legend.axes.transAxes.inverted()
            legend.set_bbox_to_anchor(bbox.transformed(tr))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.show()
