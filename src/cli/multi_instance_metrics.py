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
    # "epsilon": "multiplicative epsilon",
    "inverted_generational_distance": "IGD",
    "r_metric": "R2 metric",
}
METRIC_MAPPING_SHORT = {
    "hypervolume": "HV loss",
    # "epsilon": "E.mult.",
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


# dictionary mapping (Instance, Filter_Name, Metric_Key) to a list of configuration names and their aggregated values.
DetailedResults = dict[tuple[str, str, str], list[tuple[str, float]]]


def calculate_aggregated_metrics(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    filters_info: list[tuple[str, Any]], # Use Any for FilterExpression type
    metric_keys: list[str],
    aggregation_type: Literal["min", "max", "mean"] = "mean",
) -> tuple[pd.DataFrame, DetailedResults]: # <-- RETURN TYPE CHANGE
    """
    Calculates the average metric value per filter and instance.
    Also returns detailed, config-level metric results for internal ranking.
    """
    results: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    detailed_results: DetailedResults = defaultdict(list)

    for file_name, instance_data in all_metrics.items():
        for filter_name, filter_exp in filters_info:
            matched_values_by_metric: dict[str, list[float]] = {
                key: [] for key in metric_keys
            }
            # NEW: Dictionary to track configuration name and its metric value
            matched_configs_by_metric: dict[str, list[tuple[str, float]]] = {
                key: [] for key in metric_keys
            }

            for config_name, metrics in instance_data.items():
                if filter_exp.is_match(config_name):
                    for key in metric_keys:
                        if key in metrics:
                            # For the main DF: list of values for aggregation
                            matched_values_by_metric[key].append(metrics[key])

                            # For the detailed results: (config_name, value)
                            matched_configs_by_metric[key].append((config_name, metrics[key]))

            for metric_key in metric_keys:
                matched_values = matched_values_by_metric[metric_key]

                if matched_values:
                    if aggregation_type == "max":
                        agg_value = max(matched_values)
                    elif aggregation_type == "min":
                        agg_value = min(matched_values)
                    else:
                        agg_value = sum(matched_values) / len(matched_values)
                else:
                    agg_value = float("nan")

                results[(file_name, metric_key)][filter_name] = agg_value

                # NEW: Store detailed results
                if matched_configs_by_metric[metric_key]:
                    detailed_results[(file_name, filter_name, metric_key)] = matched_configs_by_metric[metric_key]


    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Instance", "Metric_Key"])
    df = df.sort_index(level="Instance", axis=0)
    df.columns.name = "Filter"

    return df, detailed_results


def get_internal_ranking(
    detailed_results: DetailedResults,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Calculates the average rank and std deviation for each CONFIGURATION NAME
    within each filter group (across all instances).

    Returns a dictionary keyed by (Filter Name, Metric Key).
    """
    internal_ranking_data = defaultdict(lambda: defaultdict(list)) # Key: (Filter, Metric), Value: {Config: [Ranks...]}

    # 1. Calculate ranks per instance
    for (file_name, filter_name, metric_key), config_data in detailed_results.items():
        is_minimized = metric_key in MINIMIZED_METRICS

        # Unpack data into a DataFrame for ranking
        df_data = pd.DataFrame(config_data, columns=["Configuration", "Metric_Value"])

        # Rank the configurations for this specific instance/filter group
        # Smaller values get smaller ranks if minimized, larger values get smaller ranks if maximized
        rank_ascending = is_minimized # True if minimizing (smaller value = better rank)

        df_data["Rank"] = df_data["Metric_Value"].rank(
            method="average", ascending=rank_ascending
        )

        # Store the rank of each configuration
        for _, row in df_data.iterrows():
            key = (filter_name, metric_key)
            config = row["Configuration"]
            rank = row["Rank"]
            internal_ranking_data[key][config].append(rank)

    # 2. Compute Average Rank and Std Dev
    final_internal_rankings = {}

    for (filter_name, metric_key), config_ranks in internal_ranking_data.items():
        ranking_list = []
        is_minimized = is_minimized = metric_key in MINIMIZED_METRICS

        for config, ranks in config_ranks.items():
            avg_rank = np.mean(ranks)
            std_rank = np.std(ranks)
            ranking_list.append(
                {"Configuration": config, "Avg. Rank": avg_rank, "Std. Rank": std_rank}
            )

        ranking_df = pd.DataFrame(ranking_list).set_index("Configuration")

        # Determine the final ranking based on average rank (always ascending)
        ranking_df["Rank"] = ranking_df["Avg. Rank"].rank(method="min").astype(int)
        ranking_df = ranking_df.sort_values(by="Avg. Rank", ascending=True)

        # Reorder columns and store
        ranking_df = ranking_df[["Rank", "Avg. Rank", "Std. Rank"]]
        final_internal_rankings[(filter_name, metric_key)] = ranking_df

    return final_internal_rankings


def perform_nemenyi_test_on_metrics(
    avg_df: pd.DataFrame, metric_names: list[str], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Calculate all necessary information from based on the averaged metrics values.

    Args:
        avg_df (pd.DataFrame): Dataframe with aggregated metrics values per instance and configuration.
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
            axis=1, method="average", ascending=is_minimized_metric # type: ignore
        )

        rank_array = np.asarray(ranks.dropna())

        if rank_array.shape[0] < 2 or rank_array.shape[1] < 2:
            print(
                f"Skipping {display_name}: Insufficient data ({rank_array.shape[0]} instances, {rank_array.shape[1]} methods)."
            )
            continue

        stat, p = stats.friedmanchisquare(*rank_array.T)
        reject_h0 = p <= alpha

        nemenyi_scores = pd.DataFrame()
        if reject_h0:
            nemenyi_scores = scikit_posthocs.posthoc_nemenyi_friedman(rank_array)
            labels = ranks.columns
            nemenyi_scores = nemenyi_scores.set_axis(labels, axis="columns")
            nemenyi_scores = nemenyi_scores.set_axis(labels, axis="rows") # type: ignore

        average_ranks = cast(pd.Series, ranks.mean(axis=0)).sort_values(ascending=True)

        results_by_metric[display_name] = {
            "friedman_p": p,
            "rejected": reject_h0,
            "nemenyi_scores": nemenyi_scores,
            "ranks": ranks,
            "average_ranks": average_ranks,
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


def plot_combined_results(
    statistical_results: dict[str, dict[str, Any]],
    detailed_results: DetailedResults,
    metric_keys: list[str],
    plot_file: Path,
    metric_mapping: dict[str, str],
):
    """
    Generates a combined figure: Nemenyi sign plots on the top row and
    Box plots on the bottom row, aligned by metric.
    """

    # 1. Prepare data and plot configuration
    all_metrics = list(metric_keys)
    num_metrics = len(all_metrics)

    # Restructure detailed_results into a DataFrame for easy grouping
    plot_data_list = []
    for (file_name, filter_name, metric_key), config_data in detailed_results.items():
        for _, metric_value in config_data:
            plot_data_list.append({
                "Filter_Group": filter_name,
                "Metric_Key": metric_key,
                "Metric_Value": metric_value,
            })

    if not plot_data_list:
        click.echo("Error: No data found to plot.", err=True)
        return

    full_df = pd.DataFrame(plot_data_list)

    # Get all unique filter groups (X-axis labels)
    filter_groups = sorted(full_df["Filter_Group"].unique().tolist())
    num_filters = len(filter_groups)

    # Use a 2-row, N-column grid (N = number of metrics)
    fig, axes = plt.subplots(
        2,
        num_metrics,
        figsize=(4 * num_metrics, 8), # Adjust figure size dynamically
        squeeze=False,
    )

    heatmap_args = {
        "linewidths": 0.25,
        "linecolor": "0.5",
        "cbar_ax_bbox": [0.97, 0.35, 0.04, 0.3],
        "square": True,
        "cbar": False,
    }

    # 2. Iterate through ALL metrics and plot Nemenyi (Top) and Box Plot (Bottom)
    for i, metric_key in enumerate(all_metrics):
        metric_name = metric_mapping.get(metric_key, metric_key)

        # --- TOP ROW: NEMENYI SIGN PLOT ---
        ax_nemenyi = axes[0, i]

        if metric_name in statistical_results and statistical_results[metric_name].get("rejected", False):
            # Nemenyi plot is only drawn if H0 was rejected
            scores: pd.DataFrame = statistical_results[metric_name]["nemenyi_scores"]

            # Ensure the order of filters matches the data plot order
            scores = scores.reindex(index=filter_groups, columns=filter_groups)

            # scikit_posthocs.sign_plot creates a heatmap
            scikit_posthocs.sign_plot(scores, ax=ax_nemenyi, **heatmap_args)
            ax_nemenyi.set_title(f"Nemenyi ({metric_name})", fontsize=10)
            ax_nemenyi.set_xticklabels(ax_nemenyi.get_xticklabels(), rotation=90)
            ax_nemenyi.set_yticklabels(ax_nemenyi.get_yticklabels(), rotation=0)
            ax_nemenyi.set_xlabel('')
            ax_nemenyi.set_ylabel('')
        else:
            # If Nemenyi is not significant or missing, show a placeholder
            ax_nemenyi.text(0.5, 0.5, "H0 Not Rejected",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax_nemenyi.transAxes,
                            fontsize=12, color='gray')
            ax_nemenyi.set_title(f"Nemenyi ({metric_name})", fontsize=10)
            ax_nemenyi.set_xticks([])
            ax_nemenyi.set_yticks([])
            ax_nemenyi.spines['top'].set_visible(False)
            ax_nemenyi.spines['right'].set_visible(False)
            ax_nemenyi.spines['bottom'].set_visible(False)
            ax_nemenyi.spines['left'].set_visible(False)


        # --- BOTTOM ROW: BOX PLOT ---
        ax_box = axes[1, i]

        metric_df = full_df[full_df["Metric_Key"] == metric_key].copy()

        # Prepare data for Matplotlib: a list of arrays, ensuring correct order
        data_to_plot = [
            metric_df[metric_df["Filter_Group"] == group]["Metric_Value"].values.astype(float)
            for group in filter_groups
        ]

        # Create the Box Plot
        box_plot = ax_box.boxplot(
            data_to_plot,
            tick_labels=filter_groups,
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 1.5},
            flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'red', 'alpha': 0.5}
        )

        colors = plt.cm.Set2.colors[:num_filters]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax_box.set_title(f"Distribution ({metric_name})", fontsize=8)
        ax_box.tick_params(axis='x', rotation=45)

        ax_box.grid(axis='y', linestyle='--', alpha=0.7)


    # 3. Finalize and Save Figure
    fig.tight_layout(pad=5.0)

    try:
        plt.savefig(plot_file, bbox_inches="tight", dpi=300)
        click.echo(f"\nCombined Nemenyi and Box plots saved to: {plot_file}")
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
    multiple=True,
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
    input_pattern: list[str],
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
    file_paths = sorted(
        set(instance for p in input_pattern for instance in glob.glob(p))
    )
    file_paths = [Path(p) for p in file_paths]

    if not file_paths:
        click.echo(
            f"Error: No metrics files found matching the patterns {input_pattern}. Exiting."
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
        avg_df, detailed_results = calculate_aggregated_metrics(
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

    ranking_data = display_and_get_ranking(statistical_results)
    internal_rankings = get_internal_ranking(detailed_results)
    for (filter_name, metric_key), ranking_df in internal_rankings.items():
        click.echo(f"\nFilter: {filter_name}, Metric: {metric_key}")
        print(ranking_df.to_string(float_format="%.3f"))

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

    if plot_file:
        plot_combined_results(
            statistical_results,
            detailed_results,
            METRIC_KEYS,
            plot_file,
            METRIC_MAPPING,
        )


if __name__ == "__main__":
    compare_metrics()
