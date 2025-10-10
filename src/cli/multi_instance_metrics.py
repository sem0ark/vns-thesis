import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs
from scipy import stats

from src.cli.filter_param import ClickFilterExpression, FilterExpression

METRIC_MAPPING = {
    "hypervolume": "relative hypervolume loss",
    "epsilon": "multiplicative epsilon",
    "inverted_generational_distance": "IGD",
    "r_metric": "R2 metric",
}
METRIC_MAPPING_SHORT = {
    "hypervolume": "HV loss",
    "epsilon": "E.mult.",
    "inverted_generational_distance": "IGD",
    "r_metric": "R2 metric",
    "relative hypervolume loss": "HV loss",
    "multiplicative epsilon": "E.mult.",
    "IGD": "IGD",
    "R2 metric": "R2 metric",
}
METRIC_KEYS = list(METRIC_MAPPING.keys())
METRIC_NAMES = list(METRIC_MAPPING.values())

# NOTE: at least for now all provided metrics are to be minimized
MINIMIZED_METRICS = METRIC_KEYS + METRIC_NAMES

AggregationMode = Literal["min", "max", "mean"]


def load_metrics_data(file_paths: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    """
    Loads data from multiple JSON files into a nested dictionary structure:
    {file_name: {config_name: {metric_name: value, ...}, ...}, ...}
    """
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for path in file_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                metrics_data = data["metrics"]

                if not isinstance(metrics_data, dict):
                    click.echo(
                        f"Warning: Skipping file {path.name}. Root is not a dictionary.",
                        err=True,
                    )
                    continue

                processed_data = {}
                for config_name, metrics in metrics_data.items():
                    if isinstance(metrics, dict):
                        processed_data[config_name] = {
                            k: float(v)
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))
                        }

                all_metrics[path.name] = processed_data

        except json.JSONDecodeError:
            click.echo(f"Warning: Skipping file {path.name}. Malformed JSON.", err=True)
        except Exception as e:
            click.echo(
                f"Warning: Skipping file {path.name}. An unexpected error occurred: {e}",
                err=True,
            )

    return all_metrics


def calculate_aggregated_metrics(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    filters_info: list[tuple[str, FilterExpression]],
    metric_keys: list[str] = METRIC_KEYS,
    aggregation_type: Literal["min", "max", "mean"] = "mean",
) -> pd.DataFrame:
    """
    Calculates the average value for all standard metrics, grouped by filter
    and instance, and sorts the results by instance name.

    The resulting DataFrame has:
    - Index: MultiIndex (Instance, Metric_Key)
    - Columns: Filter names
    - Values: Average metric value for matched configs
    """
    results: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    for file_name, instance_data in all_metrics.items():
        for filter_name, filter_exp in filters_info:
            matched_values_by_metric: dict[str, list[float]] = {
                key: [] for key in metric_keys
            }

            for config_name, metrics in instance_data.items():
                if filter_exp.is_match(config_name):
                    for key in metric_keys:
                        if key in metrics:
                            matched_values_by_metric[key].append(metrics[key])

            for metric_key in metric_keys:
                matched_values = matched_values_by_metric[metric_key]

                if matched_values:
                    if aggregation_type == "max":
                        average_value = max(matched_values)
                    elif aggregation_type == "min":
                        average_value = min(matched_values)
                    else:
                        average_value = sum(matched_values) / len(matched_values)
                else:
                    average_value = float("nan")

                results[(file_name, metric_key)][filter_name] = average_value

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Instance", "Metric_Key"])
    df = df.sort_index(level="Instance", axis=0)
    df.columns.name = "Filter"

    return df


def perform_nemenyi_test_on_metrics(
    avg_df: pd.DataFrame, metric_names: list[str], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Calculate all necessary information from based on the averaged mentrics values.

    Args:
        avg_df (pd.DataFrame): Dataframe with aggregated metrics values per instance and ocnfiguration.
        metric_names (list[str]): Metrics to be considered in the tests
        alpha (float, optional): Confidence value. Defaults to 0.05.

    Returns:
        dict[str, dict[str, Any]]: test results per metric.
    """

    results_by_metric: dict[str, dict[str, Any]] = {}

    for display_name in metric_names:
        metric_key = next(
            (k for k, v in METRIC_MAPPING.items() if v == display_name), None
        )
        if not metric_key:
            continue

        try:
            metric_data = avg_df.xs(display_name, level="Metric_Key", drop_level=True)
        except KeyError:
            print(f"Skipping {display_name}: No data found in DataFrame.")
            continue

        is_minimized_metric = display_name in MINIMIZED_METRICS
        ranks = metric_data.rank(
            axis=1, method="average", ascending=is_minimized_metric
        )

        rank_array = np.asarray(ranks.dropna())

        if rank_array.shape[0] < 2 or rank_array.shape[1] < 2:
            print(
                f"Skipping {display_name}: Insufficient data ({rank_array.shape[0]} instances, {rank_array.shape[1]} methods)."
            )
            continue

        # --- Friedman Test ---
        stat, p = stats.friedmanchisquare(*rank_array.T)
        reject_h0 = p <= alpha

        # --- Nemenyi Test ---
        nemenyi_scores = pd.DataFrame()
        if reject_h0:
            nemenyi_scores = scikit_posthocs.posthoc_nemenyi_friedman(rank_array)
            labels = ranks.columns
            nemenyi_scores = nemenyi_scores.set_axis(labels, axis="columns")
            nemenyi_scores = nemenyi_scores.set_axis(labels, axis="rows")

        # --- Calculate Average Ranks ---
        average_ranks = cast(pd.Series, ranks.mean(axis=0)).sort_values(ascending=True)

        results_by_metric[display_name] = {
            "friedman_p": p,
            "rejected": reject_h0,
            "nemenyi_scores": nemenyi_scores,
            "ranks": ranks,  # Keep for the new ranking function
            "average_ranks": average_ranks,  # New information
            "is_minimized_metric": is_minimized_metric,
        }

    return results_by_metric


def display_and_get_ranking(
    statistical_results: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """
    Generates a ranking table based on the average ranks from the Friedman test.
    """
    ranking_results = {}

    click.echo("\n--- Overall Method Ranking (Based on Average Ranks) ---")

    for metric_name, result in statistical_results.items():
        if result["ranks"].empty:
            continue

        ranks: pd.DataFrame = result["ranks"]

        # Calculate key statistics for ranking table
        avg_rank = ranks.mean().rename("Avg. Rank")
        std_rank = ranks.std().rename("Std. Rank")

        # Determine the sorting order based on the metric type
        is_minimized = result["is_minimized_metric"]
        rank_sort_ascending = True if is_minimized else False  # Rank 1 is always best

        # Combine, sort, and format
        ranking_df = pd.concat([avg_rank, std_rank], axis=1)
        ranking_df["Rank"] = ranking_df["Avg. Rank"].rank(method="min").astype(int)

        # Sort the table to show the best-performing algorithm first
        ranking_df = ranking_df.sort_values(
            by="Avg. Rank", ascending=rank_sort_ascending
        )

        # Reorder columns
        ranking_df = ranking_df[["Rank", "Avg. Rank", "Std. Rank"]]

        # Determine if rank 1 is "Best (Lower)" or "Best (Higher)"
        best_label = (
            "BEST (Lower is Better)" if is_minimized else "BEST (Higher is Better)"
        )

        click.echo(f"\nMetric: {metric_name} ({best_label})")
        print(ranking_df.to_string(float_format="%.3f"))

        ranking_results[metric_name] = ranking_df

    return ranking_results


def plot_nemenyi_scores(results: dict[str, dict[str, Any]], plot_file: Path):
    """
    Generates and saves a sign-plot of the Nemenyi post-hoc scores for each metric
    in a single-column (vertical) layout and includes a single, shared colorbar.
    """
    plottable_results = {k: v for k, v in results.items() if v.get("rejected", False)}
    num_plottable = len(plottable_results)

    if num_plottable == 0:
        click.echo("\nNo significant differences found to plot.", err=True)
        return

    # Use a dynamic grid for multiple plots in a single figure
    fig, axes = plt.subplots(
        1,
        num_plottable,
        figsize=(5 * num_plottable, 5),
        squeeze=False,
    )
    axes = axes.flatten()
    heatmap_args = {
        "linewidths": 0.25,
        "linecolor": "0.5",
        "square": True,
        "cbar": False,
    }

    for i, (metric_name, result) in enumerate(plottable_results.items()):
        ax = axes[i]
        scores: pd.DataFrame = result["nemenyi_scores"]
        scikit_posthocs.sign_plot(scores, ax=ax, **heatmap_args)
        ax.set_title(f"Nemenyi Post-Hoc Test ({metric_name})", fontsize=10)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

    try:
        plt.savefig(plot_file, bbox_inches="tight", dpi=300)
        click.echo(f"\nNemenyi sign-plots saved to: {plot_file}")
    except Exception as e:
        click.echo(f"\nError saving plot file {plot_file}: {e}", err=True)
    finally:
        plt.close(fig)


@click.command(
    help="Compare average metrics across multiple instance files using N filters."
)
@click.option(
    "-i",
    "--input-pattern",
    required=True,
    type=str,
    help="Glob pattern for metrics JSON files (e.g., 'metrics_output/*.json').",
)
@click.option(
    "-f",
    "--filter-expression",
    "filters",
    type=ClickFilterExpression(),
    required=True,
    multiple=True,
    default=[],
    help="Boolean filter expression for config names (e.g., '(vns or nsga2) and 120s').",
)
@click.option(
    "-n",
    "--filter-name",
    "filter_names",
    type=str,
    multiple=True,
    default=[],
    help="Names for each filter expression used for logging and plotting.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to save the resulting CSV or Excel file.",
)
@click.option(
    "-p",
    "--plot-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to save the Nemenyi sign-plot (e.g., 'nemenyi_plot.png').",
)
@click.option(
    "--alpha",
    type=click.FloatRange(0.01, 0.5),
    default=0.05,
    show_default=True,
    help="Significance level (alpha) for statistical tests.",
)
@click.option(
    "--agg",
    "aggregate_func",
    type=click.Choice(["min", "max", "mean"]),
    default="mean",
    show_default=True,
    help="Function by which we will group metric results per filter group.",
)
def compare_metrics(
    input_pattern: str,
    filters: list[FilterExpression],
    filter_names: list[str],
    output_file: Path | None,
    plot_file: Path | None,
    alpha: float,
    aggregate_func: AggregationMode,
):
    """
    Main function to load, filter, average, and display cross-instance metrics.
    """
    file_paths = [Path(p) for p in glob.glob(input_pattern)]

    if not file_paths:
        click.echo(
            f"Error: No metrics files found matching the pattern '{input_pattern}'. Exiting."
        )
        return

    if len(filter_names) != len(filters):
        click.echo(
            "Warning: number of filter names does not correspond to the number of filters. "
            "Padding missing names with their filter expressions.",
            err=True,
        )
        filter_names = list(filter_names)
        filters = list(filters)

        num_missing = len(filters) - len(filter_names)
        if num_missing > 0:
            start_index = len(filter_names)
            for i in range(start_index, len(filters)):
                filter_names.append(str(filters[i]))

        elif num_missing < 0:
            click.echo(
                f"Warning: Too many filter names provided ({len(filter_names)}). "
                f"Truncating names to match the number of filters ({len(filters)}).",
                err=True,
            )
            filter_names = filter_names[: len(filters)]

    filter_objects = list(zip(filter_names, filters))

    click.echo(f"Found {len(file_paths)} files.")
    for name, exp in filter_objects:
        click.echo(f"- {name}: {exp.initial_string}")

    all_metrics = load_metrics_data(file_paths)

    if not all_metrics:
        click.echo("Error: Failed to load valid metrics data from any file. Exiting.")
        return

    try:
        avg_df = calculate_aggregated_metrics(
            all_metrics, filter_objects, METRIC_KEYS, aggregate_func
        )

        new_index = avg_df.index.to_frame(index=False)
        new_index["Metric_Key"] = new_index["Metric_Key"].map(METRIC_MAPPING)
        avg_df.index = pd.MultiIndex.from_frame(new_index)

    except Exception as e:
        click.echo(f"An error occurred during metric calculation: {e}", err=True)
        return

    click.echo("\n--- Averaged Metrics Table (Instance x Metric x Filter) ---")
    print(avg_df.to_string(na_rep="N/A"))

    # 3. Perform Statistical Tests (Friedman + Nemenyi)
    statistical_results = perform_nemenyi_test_on_metrics(
        avg_df.copy(), METRIC_NAMES, alpha=alpha
    )

    # 4. Display Statistical Results
    click.echo("\n--- Nemenyi Post-Hoc Results ---")
    for metric, result in statistical_results.items():
        click.echo(f"\nMetric: {metric}")
        click.echo(f"  Friedman p-value: {result['friedman_p']:.4f}")

        if result["rejected"]:
            click.echo("  Status: SIGNIFICANT DIFFERENCE FOUND (H0 Rejected).")
            click.echo("  Nemenyi p-value matrix:")
            print(result["nemenyi_scores"].to_string())
        else:
            click.echo("  Status: NO SIGNIFICANT DIFFERENCE (H0 Not Rejected).")

    # 5. Display Method Ranking (NEW)
    ranking_data = display_and_get_ranking(statistical_results)

    # 6. Save Output
    if output_file:
        try:
            if output_file.suffix.lower() in [".xlsx", ".xls"]:
                with pd.ExcelWriter(output_file) as writer:
                    avg_df.to_excel(writer, sheet_name="Average_Metrics")

                    for metric, result in statistical_results.items():
                        metric_name = METRIC_MAPPING_SHORT[metric]
                        if result["rejected"]:
                            sheet_name = f"Nemenyi {metric_name}"
                            result["nemenyi_scores"].to_excel(
                                writer, sheet_name=sheet_name
                            )

                        # Save Ranking
                        ranking_data[metric].to_excel(
                            writer, sheet_name=f"Ranking {metric_name}"
                        )

                click.echo(
                    f"Results successfully saved to Excel (Metrics, Nemenyi, and Ranking): {output_file}"
                )
            else:
                avg_df.to_csv(output_file)
                click.echo(f"\nAverage metrics saved to: {output_file}")

        except Exception as e:
            click.echo(
                f"\nError: Could not save output file {output_file}: {e}", err=True
            )

    # 7. Plot Nemenyi Sign-Plot
    if plot_file:
        plot_nemenyi_scores(statistical_results, plot_file)


if __name__ == "__main__":
    compare_metrics()
