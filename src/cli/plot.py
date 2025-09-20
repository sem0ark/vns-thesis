import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.pyplot as plt

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.cli.shared import SavedRun


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
        print("Got a predefined reference front!")
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


def plot_runs(instance_path: Path, runs_grouped: Dict[str, List[SavedRun]]) -> None:
    print("""Plotting the results:
Controls:
Press 'h' to hide all graphs except reference front.
Use arrow keys to move legend around.
You can also click on graphs in legend to show/hide any specific one.
""")

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
        return

    if reference_front.shape[1] != 2:
        logging.error(
            "Plotting is only supported for problems with exactly 2 objectives."
        )
        return

    # Calculate hypervolume for each run and prepare data for plotting
    plot_data = []

    # We need a shared reference point for HV calculation
    all_merged_fronts = []
    for runs in runs_grouped.values():
        merged_front_objectives = [
            sol.objectives for run in runs for sol in run.solutions
        ]
        if merged_front_objectives:
            all_merged_fronts.append(np.array(merged_front_objectives))

    hv_indicator = HV(ref_point=(0, 0))

    for run_name, runs in runs_grouped.items():
        merged_front_objectives = sorted(
            [tuple(sol.objectives) for sol in runs[-1].solutions]
        )

        if not merged_front_objectives:
            continue

        combined_objectives = np.array(merged_front_objectives)
        negated_objectives = -combined_objectives

        nd_sorting = NonDominatedSorting()
        non_dominated_indices = nd_sorting.do(
            negated_objectives, only_non_dominated_front=True
        )
        merged_front = combined_objectives[non_dominated_indices]

        if merged_front.size > 0:
            hypervolume = hv_indicator.do(-merged_front)
        else:
            hypervolume = -np.inf

        plot_data.append(
            {"name": run_name, "front": merged_front, "hypervolume": hypervolume}
        )

    sorted_plot_data = sorted(plot_data, key=lambda d: d["hypervolume"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(right=0.7)

    lines_dict = {}

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

    if all_runs:
        metadata = all_runs[0].metadata
        title_str = f"{metadata.problem_name.upper()}, {metadata.instance_name}, {metadata.run_time_seconds}s"
        ax.set_title(title_str)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")

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
