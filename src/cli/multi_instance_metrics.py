import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import scikit_posthocs
from scipy import stats
import matplotlib.pyplot as plt

from src.cli.filter_param import ClickFilterExpression, FilterExpression

METRIC_KEYS = [
    "hypervolume",
    "epsilon",
    "inverted_generational_distance",
    "r_metric",
]


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


def calculate_average_metrics(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    filters: list[FilterExpression],
    metric_keys: list[str] = METRIC_KEYS,
) -> pd.DataFrame:
    """
    Calculates the average value for all standard metrics, grouped by filter
    and instance, and sorts the results by instance name.

    The resulting DataFrame has:
    - Index: MultiIndex (Instance, Metric_Key)
    - Columns: Filter expressions
    - Values: Average metric value for matched configs
    """

    # results will store data as: { (Instance, Metric_Key): {Filter_Str: Avg_Value} }
    results: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    for file_name, instance_data in all_metrics.items():
        for filter_exp in filters:
            filter_str = str(filter_exp)

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
                    average_value = sum(matched_values) / len(matched_values)
                else:
                    average_value = float("nan")  # Use NaN for no matches

                results[(file_name, metric_key)][filter_str] = average_value

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Instance", "Metric_Key"])
    df = df.sort_index(level="Instance", axis=0)
    df.columns.name = "Filter"

    return df


def perform_nemenyi_test_on_metrics(
    avg_df: pd.DataFrame, metric_keys: list[str], alpha: float = 0.05
) -> dict[str, dict[str, Any]]: # Changed return type to Any for dict values
    """
    Performs the Friedman test and, if rejected, the Nemenyi post-hoc test
    for each specified metric key using the average metrics DataFrame.
    """
    results_by_metric: dict[str, dict[str, Any]] = {}

    for metric_key in metric_keys:
        print(f"\n--- Statistical Comparison for: {metric_key} ---")

        try:
            metric_data = avg_df.xs(metric_key, level="Metric_Key", drop_level=True)
        except KeyError:
            print(f"Skipping {metric_key}: No data found in DataFrame.")
            continue

        ranks = metric_data.rank(axis=1, method="average", ascending=True)

        rank_array = np.asarray(ranks.dropna())

        if rank_array.shape[0] < 2 or rank_array.shape[1] < 2:
            print(
                f"Skipping {metric_key}: Insufficient data ({rank_array.shape[0]} instances, {rank_array.shape[1]} methods)."
            )
            continue

        stat, p = stats.friedmanchisquare(*rank_array.T)
        reject_h0 = p <= alpha

        print(f"Friedman Test: Chi2={stat:.4f}, p-value={p:.4f}")
        print(
            f"Reject H0 (difference exists) at {(1 - alpha) * 100}% confidence? {reject_h0}"
        )

        if not reject_h0:
            results_by_metric[metric_key] = {
                "friedman_p": p,
                "rejected": False,
                "nemenyi_scores": "Not performed (H0 not rejected)",
            }
            continue

        nemenyi_scores = scikit_posthocs.posthoc_nemenyi_friedman(rank_array)

        labels = ranks.columns
        nemenyi_scores = nemenyi_scores.set_axis(labels, axis="columns")
        nemenyi_scores = nemenyi_scores.set_axis(labels, axis="rows")

        results_by_metric[metric_key] = {
            "friedman_p": p,
            "rejected": True,
            "nemenyi_scores": nemenyi_scores,
        }

    return results_by_metric

def plot_nemenyi_scores(results: dict[str, dict[str, Any]], plot_file: Path):
    """
    Generates and saves a sign-plot of the Nemenyi post-hoc scores for each metric
    in a 2x2 grid layout and includes a single, centered legend (colorbar).
    """
    plottable_results = {
        k: v for k, v in results.items() if v.get("rejected", False)
    }
    num_plottable = len(plottable_results)

    if num_plottable == 0:
        click.echo("\nNo significant differences found to plot.", err=True)
        return

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(16, 10)
    )
    
    axes_flat = axes.flatten()

    heatmap_args = {
        'linewidths': 0.25, 
        'linecolor': '0.5', 
        'square': True,
    }

    for ax, (metric_key, result) in zip(axes_flat, plottable_results.items()):
        scores: pd.DataFrame = result["nemenyi_scores"]

        scikit_posthocs.sign_plot(scores, ax=ax, **heatmap_args)
        ax.set_title(f"Nemenyi Post-Hoc Test ({metric_key})", fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    for i in range(num_plottable, nrows * ncols):
        fig.delaxes(axes_flat[i])

    try:
        # Adjust subplot parameters for tight layout and to make room for the colorbar
        # plt.subplots_adjust(right=0.9, wspace=0.5, hspace=0.7)
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
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
def compare_metrics(
    input_pattern: str,
    filters: list[FilterExpression],
    output_file: Path | None,
    plot_file: Path | None,
    alpha: float,
):
    """
    Main function to load, filter, average, and display cross-instance metrics.
    """
    file_paths = [Path(p) for p in glob.glob(input_pattern)]

    if not file_paths:
        click.echo(f"Error: No metrics files found matching the pattern '{input_pattern}'. Exiting.")
        return

    click.echo(f"Found {len(file_paths)} files.")
    click.echo(f"Filters to apply: {[f for f in filters]}")

    all_metrics = load_metrics_data(file_paths)

    if not all_metrics:
        click.echo("Error: Failed to load valid metrics data from any file. Exiting.")
        return
    try:
        avg_df = calculate_average_metrics(all_metrics, filters)
    except Exception as e:
        click.echo(
            f"An error occurred during metric calculation: {e}", err=True
        )
        return

    click.echo("\n--- Averaged Metrics Table (Instance x Metric x Filter) ---")
    print(avg_df.to_string(na_rep="N/A"))

    statistical_results = perform_nemenyi_test_on_metrics(
        avg_df, METRIC_KEYS, alpha=alpha
    )
    
    click.echo("\n--- Nemenyi Post-Hoc Results ---")
    for metric, result in statistical_results.items():
        click.echo(f"\nMetric: {metric}")
        click.echo(f"  Friedman p-value: {result['friedman_p']:.4f}")
        
        if result['rejected']:
            click.echo("  Status: SIGNIFICANT DIFFERENCE FOUND (H0 Rejected).")
            click.echo("  Nemenyi p-value matrix:")
            print(result['nemenyi_scores'].to_string())
        else:
            click.echo("  Status: NO SIGNIFICANT DIFFERENCE (H0 Not Rejected).")

    if output_file:
        try:
            if output_file.suffix.lower() in [".xlsx", ".xls"]:
                with pd.ExcelWriter(output_file) as writer:
                    avg_df.to_excel(writer, sheet_name="Average_Metrics")
                    
                    # Save each Nemenyi matrix to a separate sheet
                    for metric, result in statistical_results.items():
                        if result['rejected']:
                            sheet_name = f"Nemenyi_{metric}"
                            result['nemenyi_scores'].to_excel(writer, sheet_name=sheet_name)
                    
                click.echo(f"\nResults successfully saved to Excel (Avg Metrics + Nemenyi matrices): {output_file}")
            else:
                # Default to saving only the average metrics to CSV
                avg_df.to_csv(output_file)
                click.echo(f"\nAverage metrics saved to: {output_file}")

        except Exception as e:
            click.echo(
                f"\nError: Could not save output file {output_file}: {e}", err=True
            )
            
    if plot_file:
        plot_nemenyi_scores(
            {k: v for k, v in statistical_results.items() if v.get('rejected', False)},
            plot_file
        )


if __name__ == "__main__":
    compare_metrics()
