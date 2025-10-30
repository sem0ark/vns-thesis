import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
from matplotlib.transforms import Bbox
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from src.cli.shared import SavedRun
from src.vns.acceptance import ParetoFront

COMMON_METRICS_FOLDER = Path(__file__).parent.parent.parent / "metrics"


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
    This is useful for intuitively plotting objective results to not show negated values,
    because these objectives were to be maximized.

    Returns the modified array and a list of indices of the flipped columns.
    """
    flipped_indices = []
    # Avoid modifying the original array
    front_copy = front.copy()
    for i in range(front_copy.shape[1]):
        if np.all(front_copy[:, i] < 0):
            front_copy[:, i] *= -1
            flipped_indices.append(i)
    return front_copy, flipped_indices


def merge_runs_to_non_dominated_front(runs: list[SavedRun]) -> np.ndarray:
    """
    Calculates the reference front by combining and sorting all non-dominated
    solutions from all runs. It can be overridden by a predefined front.

    Args:
        runs: A list of SavedRun objects.
        predefined_front: A list of objective vectors to use as the reference front.

    Returns:
        A NumPy array representing the non-dominated reference front.
    """

    reference_front = ParetoFront()
    for run in runs:
        for sol in run.solutions:
            reference_front.accept(cast(Any, sol))

    all_objectives = []
    for sol in reference_front.solutions:
        all_objectives.append(sol.objectives)

    if not all_objectives:
        logging.warning("No solutions found in any run to create a reference front.")
        return np.array([])

    return np.unique(np.array(all_objectives), axis=0)


def epsilon_indicator(A, B, additive=False):
    """
    Calculates the multiplicative epsilon indicator.
    Reference: https://davidwalz.github.io/posts/2020/multiobjective-metrics/#Epsilon-indicator

    Args:
        A: The approximation front (a NumPy array).
        R: The reference front (a NumPy array).
        additive: toggle between additive and multiplicative.

    Returns:
        The epsilon indicator value.
    """
    if additive:
        r = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    else:  # multiplicative
        r = A[:, np.newaxis, :] / B[np.newaxis, :, :]
    return r.max(axis=2).min(axis=0).max()


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
        # Use approximation for more than 2 objectives
        weights = np.random.rand(num_weights, num_objectives)
        # Scale to [0, 1]
        return weights / np.sum(weights, axis=1, keepdims=True)


def calculate_r2_metric(
    front: np.ndarray, ideal_point: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculates the R2 unary indicator based on the weighted Tchebycheff utility function.
    Paper: https://hal.science/hal-01329559/

    Args:
        front: np array of objective vectors (solutions).
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


def calculate_coverage(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculates the coverage metric C(A, B), the ratio of solutions in B
    that are dominated by at least one solution in A.

    C(A, B) = |{b in B | ∃ a in A, a dominates b}| / |B|

    Args:
        A: The approximation front A (a NumPy array).
        B: The approximation front B (a NumPy array).

    Returns:
        The coverage metric C(A, B) (a value between 0 and 1).
    """
    if B.size == 0:
        return np.nan if A.size == 0 else 0.0
    if A.size == 0:
        return 0.0

    num_dominated = 0
    # A solution 'a' dominates 'b' if a is better than b in ALL objectives (for minimization).
    # Since we assume minimization: a_i <= b_i for all i, and a_j < b_j for at least one j.

    # Iterate over all solutions in B (the covered front)
    for b in B:
        for a in A:
            is_at_least_as_good = np.all(a <= b + 1e-9)
            is_strictly_better = np.any(a < b - 1e-9)

            if is_at_least_as_good and is_strictly_better:
                num_dominated += 1
                break

    return num_dominated / B.shape[0]


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


def calculate_metrics(
    instance_path: Path,
    all_runs_grouped: dict[str, list[SavedRun]],
    filtered_runs_grouped: dict[str, list[SavedRun]],
) -> tuple[dict[str, Metrics], dict[str, list[Metrics]]]:
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

    all_runs = [run for runs in all_runs_grouped.values() for run in runs]

    predefined_front = problem_data.get("reference_front")
    if predefined_front:
        print("Got a predefined reference front!")
        reference_front = np.array(predefined_front)
    else:
        print(f"Merging all runs ({len(all_runs)}) to get a reference front...")
        reference_front = merge_runs_to_non_dominated_front(all_runs)

    if reference_front.size == 0:
        return {}, {}

    if reference_front.size == 0:
        logging.error(
            "Failed to create a reference front from the provided runs or file."
        )
        return {}, {}

    num_objectives = reference_front.shape[1]
    hypervolume_reference_point = _get_hypervolume_reference_point([reference_front])
    reference_front_hypervolume = (
        HV(ref_point=hypervolume_reference_point).do(reference_front) or np.nan
    )

    # Calculate an ideal point for R2 (best point)
    r2_ideal_point = np.min(reference_front, axis=0)
    r2_weights = _generate_uniform_weights(num_objectives, num_weights=100)

    reference_r2_metric = calculate_r2_metric(
        reference_front, r2_ideal_point, r2_weights
    )

    metrics_results_aggregated = {}
    metrics_results = {}

    for run_name, runs in filtered_runs_grouped.items():
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

            # Relative missing hypervolume compared to reference front. For minimization, smaller is better.
            hypervolume_indicator = HV(ref_point=hypervolume_reference_point)
            relative_hypervolume = (
                reference_front_hypervolume - (hypervolume_indicator.do(front) or 0.0)
            ) / (reference_front_hypervolume + 1e-6)

            # Multiplicative Epsilon. For minimization, smaller is better.
            epsilon = epsilon_indicator(front, reference_front)

            # Relative missing R2 Unary Indicator compared to reference front. For minimization, smaller is better.
            relative_r_metric = (
                calculate_r2_metric(front, r2_ideal_point, r2_weights)
                - reference_r2_metric
            )

            # Inverted Generational Distance (IGD). For minimization, smaller is better.
            igd_indicator = IGD(reference_front)
            inverted_generational_distance = igd_indicator.do(front)

            run_metrics_list.append(
                Metrics(
                    epsilon=epsilon,
                    hypervolume=relative_hypervolume,
                    r_metric=relative_r_metric,
                    inverted_generational_distance=0
                    if epsilon == 1
                    else (inverted_generational_distance or np.nan),
                )
            )

        if run_metrics_list:
            avg_epsilon = np.nanmean([m.epsilon for m in run_metrics_list])
            avg_hypervolume = np.nanmean([m.hypervolume for m in run_metrics_list])
            avg_r_metric = np.nanmean([m.r_metric for m in run_metrics_list])
            avg_igd = np.nanmean(
                [m.inverted_generational_distance for m in run_metrics_list]
            )

            metrics_results_aggregated[run_name] = Metrics(
                epsilon=float(avg_epsilon),
                hypervolume=float(avg_hypervolume),
                r_metric=float(avg_r_metric),
                inverted_generational_distance=float(avg_igd),
            )

            metrics_results[run_name] = run_metrics_list

    return metrics_results_aggregated, metrics_results


def export_unary_metrics(
    instance_path: Path,
    all_runs_grouped: dict[str, list[SavedRun]],
    filtered_runs_grouped: dict[str, list[SavedRun]],
):
    _, metrics_all = calculate_metrics(
        instance_path, all_runs_grouped, filtered_runs_grouped
    )

    metrics_for_export = {}

    for run_name, metric_list in metrics_all.items():
        run_output = defaultdict(list)
        for metrics_obj in metric_list:
            for key, value in asdict(metrics_obj).items():
                run_output[key].append(float(value))

        metrics_for_export[run_name] = run_output

    COMMON_METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
    metadata = next(iter(filtered_runs_grouped.values()))[0].metadata
    problem_name = metadata.problem_name
    instance_name = metadata.instance_name
    json_output_path = (
        COMMON_METRICS_FOLDER / f"{problem_name}_{instance_name}_unary_metrics.json"
    )
    with open(json_output_path, mode="w") as f:
        json.dump(
            {
                "metadata": {
                    "instance_name": instance_name,
                    "problem_name": problem_name,
                },
                "metrics": metrics_for_export,
            },
            f,
        )

    print(f"Saved unary metrics file to {json_output_path}.")


def calculate_coverage_metrics(
    fronts: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Calculates the coverage metric C(RunA, RunB) for all pairs of runs.
    """
    run_names = list(fronts.keys())
    coverage_matrix = defaultdict(dict)

    for name_a in run_names:
        for name_b in run_names:
            coverage_matrix[name_a][name_b] = calculate_coverage(
                fronts[name_a], fronts[name_b]
            )

    return coverage_matrix


def export_table(
    table_data: List[List[Any]],
    headers: List[str],
    output_path: Path,
    sheet_name: str = "Metrics",
):
    """
    Exports a list of table data and headers to CSV or Excel based on the file extension.
    """
    df = pd.DataFrame(table_data, columns=headers)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        print(f"Metrics successfully exported to CSV: {output_path}")
    elif output_path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Metrics successfully exported to Excel: {output_path}")
        except ImportError:
            print(
                "Error: Please install 'xlsxwriter' (or 'openpyxl') for Excel export."
            )
            df.to_csv(output_path.with_suffix(".csv"), index=False)
            print(f"Falling back to CSV export: {output_path.with_suffix('.csv')}")
    else:
        raise ValueError(
            f"Unsupported export format: {output_path.suffix}. Use .csv or .xlsx."
        )


def prepare_coverage_table_data(
    coverage_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[str], List[List[Any]]]:
    """
    Prepares the coverage metric table data, excluding strictly dominated runs.
    Returns headers and table data.
    """
    run_names = list(coverage_matrix.keys())
    dominated_runs: set[str] = set()

    # Determine strictly dominated runs
    for name_b in run_names:
        for name_a in run_names:
            if name_a == name_b:
                continue

            coverage_a_b = coverage_matrix[name_a].get(name_b, 0.0)
            coverage_b_a = coverage_matrix[name_b].get(name_a, 0.0)

            # B is strictly inferior if A covers B completely AND B does not cover A completely
            if coverage_a_b >= 1.0 and coverage_b_a < 1.0:
                dominated_runs.add(name_b)
                break

    # Prepare table data, excluding dominated runs
    display_names = [name for name in run_names if name not in dominated_runs]

    headers = ["C(A, B)"] + display_names
    table_data = []

    for name_a in display_names:
        row = [name_a]
        for name_b in display_names:
            coverage = coverage_matrix[name_a][name_b]
            row.append(f"{coverage:.4f}" if not np.isnan(coverage) else "N/A")
        table_data.append(row)

    print(f"(Hiding runs completely dominated by another run: {list(dominated_runs)})")

    return headers, table_data


def prepare_unary_metrics_table_data(
    instance_path: Path,
    all_runs_grouped: dict[str, list[SavedRun]],
    filtered_runs_grouped: dict[str, list[SavedRun]],
) -> tuple[list[str], list[list[Any]]]:
    """
    Calculates metrics and prepares the unary metrics table data.
    Returns (metrics, (headers, table_data)).
    """
    metrics, _ = calculate_metrics(
        instance_path, all_runs_grouped, filtered_runs_grouped
    )
    # Sort the metrics. Lower values are better for all of them.
    sorted_metrics = sorted(
        metrics.items(),
        key=lambda item: (
            item[1].inverted_generational_distance,
            item[1].epsilon,
            item[1].hypervolume,  # Assumed to be relative loss (smaller is better)
            item[1].r_metric,
        ),
    )

    headers = [
        "Config",
        "Epsilon (multiplicative)",
        "Hypervolume (Relative loss)",
        "R-Metric (Relative loss)",
        "IGD",
    ]

    table_data = []
    for run_name, metric_values in sorted_metrics:
        row = [
            run_name,
            metric_values.epsilon if not np.isnan(metric_values.epsilon) else "N/A",
            metric_values.hypervolume
            if not np.isnan(metric_values.hypervolume)
            else "N/A",
            metric_values.r_metric if not np.isnan(metric_values.r_metric) else "N/A",
            metric_values.inverted_generational_distance
            if not np.isnan(metric_values.inverted_generational_distance)
            else "N/A",
        ]

        # Format the numbers as strings for console display only after table preparation
        row_str = [row[0]] + [
            f"{val:.4f}" if isinstance(val, float) else val for val in row[1:]
        ]
        table_data.append(row_str)

    return headers, table_data


def display_unary_metrics(
    instance_path: Path,
    all_runs_grouped: dict[str, list[SavedRun]],
    filtered_runs_grouped: dict[str, list[SavedRun]],
    output_file: Path | None = None,
):
    (unary_headers, unary_table_data) = prepare_unary_metrics_table_data(
        instance_path, all_runs_grouped, filtered_runs_grouped
    )

    print("--- Unary Metrics Table ---")
    print(
        tabulate.tabulate(
            unary_table_data, headers=unary_headers, tablefmt="fancy_grid"
        )
    )

    if output_file:
        base_name = output_file.stem
        suffix = output_file.suffix

        unary_export_path = output_file.with_name(f"{base_name}_unary{suffix}")
        export_table(
            table_data=[
                [row[0]]
                + [
                    float(val) if val != "N/A" and isinstance(val, str) else val
                    for val in row[1:]
                ]
                for row in unary_table_data  # Use raw float values for export
            ],
            headers=unary_headers,
            output_path=unary_export_path,
            sheet_name="Unary Metrics",
        )


def display_metrics(
    instance_path: Path,
    all_runs_grouped: dict[str, list[SavedRun]],
    filtered_runs_grouped: dict[str, list[SavedRun]],
    unary: bool,
    coverage: bool,
    export_to_common_json_format: bool,
    output_file: Path | None = None,
):
    """
    Calculates and displays all metrics (unary and coverage),
    with an option to export them.
    """
    if export_to_common_json_format:
        export_unary_metrics(
            instance_path,
            all_runs_grouped,
            filtered_runs_grouped,
        )

    if unary:
        display_unary_metrics(
            instance_path,
            all_runs_grouped,
            filtered_runs_grouped,
            output_file,
        )

    if coverage:
        coverage_metrics = calculate_coverage_metrics(
            {
                name: merge_runs_to_non_dominated_front(runs)
                for name, runs in filtered_runs_grouped.items()
            }
        )
        coverage_headers, coverage_table_data = prepare_coverage_table_data(
            coverage_metrics
        )

        print("--- Coverage (C(A, B)) Matrix ---")
        print(
            tabulate.tabulate(
                coverage_table_data, headers=coverage_headers, tablefmt="fancy_grid"
            )
        )

        if output_file:
            base_name = output_file.stem
            suffix = output_file.suffix
            coverage_export_path = output_file.with_name(
                f"{base_name}_coverage{suffix}"
            )
            export_table(
                table_data=[
                    [row[0]]
                    + [
                        float(val) if val != "N/A" and isinstance(val, str) else val
                        for val in row[1:]
                    ]
                    for row in coverage_table_data
                ],
                headers=coverage_headers,
                output_path=coverage_export_path,
                sheet_name="Coverage Matrix",
            )


def plot_runs(
    instance_path: Path,
    all_runs_grouped: Dict[str, List[SavedRun]],
    filtered_runs_grouped: dict[str, list[SavedRun]],
    objective_names: list[str],
    plot_lines=True,
) -> None:
    print("""Plotting the results:
Controls:
Press 'h' to hide all graphs except reference front.
Use arrow keys to move legend around.
You can also click on graphs in legend to show/hide any specific one.
""")

    problem_data = load_instance_data_json(instance_path)

    all_runs = [run for runs in all_runs_grouped.values() for run in runs]
    predefined_front = problem_data.get("reference_front")

    if predefined_front:
        print("Got a predefined reference front!")
        reference_front = np.array(predefined_front)
    else:
        print(
            f"Merging all runs ({len(all_runs)}) to get an approximate reference front..."
        )
        reference_front = merge_runs_to_non_dominated_front(all_runs)

    if reference_front.size == 0:
        print("Error: reference front is empty! Exiting...")
        return

    all_flipped_indices = set()
    reference_front_flipped, flipped_indices = flip_objectives_to_positive(
        reference_front
    )
    all_flipped_indices.update(flipped_indices)

    if reference_front_flipped.shape[1] != 2:
        logging.error(
            "Plotting is only supported for problems with exactly 2 objectives."
        )
        return

    # Calculate hypervolume for each run and prepare data for plotting
    plot_data = []
    hypervolume_reference_point = _get_hypervolume_reference_point([reference_front])
    hypervolume_indicator = HV(ref_point=hypervolume_reference_point)

    for run_name, runs in filtered_runs_grouped.items():
        if not runs:
            continue

        merged_front = merge_runs_to_non_dominated_front(runs)
        if merged_front.size > 0:
            hypervolume = hypervolume_indicator.do(merged_front)
        else:
            hypervolume = np.nan

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
            linestyle="-" if plot_lines else "",
            label=run_name,
            alpha=0.6,
        )
        lines_dict[run_name] = line

    # Plot the reference front
    (ref_line,) = ax.plot(
        reference_front_flipped[:, 0],
        reference_front_flipped[:, 1],
        linestyle="-",
        label="Reference Front",
        color="red",
        markersize=8,
        fillstyle="none",
        linewidth=2,
    )
    lines_dict["Reference Front"] = ref_line

    if filtered_runs_grouped:
        metadata = list(filtered_runs_grouped.values())[0][0].metadata
        title_str = f"{metadata.problem_name.upper()}, {metadata.instance_name}"

        if all_flipped_indices:
            flipped_obj_labels = [
                objective_names[i] for i in sorted(list(all_flipped_indices))
            ]
            note = f" (Note: {', '.join(flipped_obj_labels)} were negated)"
            title_str += note

        ax.set_title(title_str)
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])

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
