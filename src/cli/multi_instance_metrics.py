import click
import json
import pandas as pd
import glob

from pathlib import Path
from collections import defaultdict

from src.cli.filter_param import ClickFilterExpression, FilterExpression


def load_metrics_data(file_paths: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    """
    Loads data from multiple JSON files into a nested dictionary structure:
    {file_name: {config_name: {metric_name: value, ...}, ...}, ...}
    """
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    
    for path in file_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                metrics_data = data["metrics"]

                if not isinstance(metrics_data, dict):
                     click.echo(f"Warning: Skipping file {path.name}. Root is not a dictionary.", err=True)
                     continue

                processed_data = {}
                for config_name, metrics in metrics_data.items():
                    if isinstance(metrics, dict):
                        processed_data[config_name] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                    
                all_metrics[path.name] = processed_data
                
        except json.JSONDecodeError:
            click.echo(f"Warning: Skipping file {path.name}. Malformed JSON.", err=True)
        except Exception as e:
            click.echo(f"Warning: Skipping file {path.name}. An unexpected error occurred: {e}", err=True)
            
    return all_metrics

def calculate_average_metrics(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    filters: list[FilterExpression],
    metric_key: str
) -> pd.DataFrame:
    """
    Calculates the average metric value for each filter and each instance (file).

    The resulting DataFrame has:
    - Index: File names (instance names)
    - Columns: Filter expressions
    - Values: Average metric_key value for matched configs
    """
    
    results: dict[str, dict[str, float]] = defaultdict(dict)
    
    for file_name, instance_data in all_metrics.items():
        
        for f in filters:
            matched_values = []
            
            for config_name, metrics in instance_data.items():
                
                if f.is_match(config_name):
                    if metric_key in metrics:
                        matched_values.append(metrics[metric_key])
            
            if matched_values:
                average_value = sum(matched_values) / len(matched_values)
            else:
                average_value = float('nan') # Use NaN for no matches
            
            results[file_name][str(f)] = average_value

    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = "Instance"
    df.columns.name = f"Avg. {metric_key}"
    
    return df


@click.command(help="Compare average metrics across multiple instance files using N filters.")
@click.option(
    "-i",
    "--input-pattern",
    required=True,
    type=str,
    help="Glob pattern for metrics JSON files (e.g., 'metrics_output/*.json')."
)
@click.option(
    "-m",
    "--metric-key",
    required=True,
    type=str,
    help="The specific metric to average (e.g., 'hypervolume', 'epsilon')."
)
@click.option(
    "-f",
    "--filter-expression",
    'filters',  # Store the values in the 'filters' variable
    type=ClickFilterExpression(),
    required=True,
    multiple=True,  # Allows the option to be specified multiple times (N>=1)
    default=[],
    help="Boolean filter expression for config names (e.g., '(vns or nsga2) and 120s').",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to save the resulting CSV or Excel file."
)
def compare_metrics(
    input_pattern: str, 
    metric_key: str, 
    filters: list[FilterExpression],
    output_file: Path | None
):
    """
    Main function to load, filter, average, and display cross-instance metrics.
    """
    click.echo(f"Searching for metrics files using pattern: '{input_pattern}'")
    file_paths = [Path(p) for p in glob.glob(input_pattern)]
    
    if not file_paths:
        click.echo("Error: No metrics files found matching the pattern. Exiting.")
        return
        
    click.echo(f"Found {len(file_paths)} files. Target metric: '{metric_key}'.")
    click.echo(f"Filters to apply: {[f for f in filters]}")
    
    # 1. Load data
    all_metrics = load_metrics_data(file_paths)
    
    if not all_metrics:
        click.echo("Error: Failed to load valid metrics data from any file. Exiting.")
        return
        
    # 2. Calculate averages and build DataFrame
    try:
        df = calculate_average_metrics(all_metrics, filters, metric_key)
    except KeyError as e:
        click.echo(f"Error: The metric key '{e}' was not found in one or more configurations. Check input data.", err=True)
        return
    except Exception as e:
        click.echo(f"An unexpected error occurred during metric calculation: {e}", err=True)
        return

    # 3. Display and save results
    
    click.echo("\n" + "=" * 50)
    click.echo(f"Cross-Instance Average Metrics for: {metric_key}")
    click.echo("=" * 50)
    
    # Print the DataFrame, filling NaN with 'N/A' for display
    print(df.to_string(na_rep='N/A'))
    
    if output_file:
        try:
            if output_file.suffix.lower() == '.csv':
                df.to_csv(output_file)
                click.echo(f"\nResults successfully saved to CSV: {output_file}")
            elif output_file.suffix.lower() in ['.xlsx', '.xls']:
                # Ensure pandas and openpyxl are installed for Excel support
                df.to_excel(output_file)
                click.echo(f"\nResults successfully saved to Excel: {output_file}")
            else:
                click.echo(f"\nWarning: Unsupported output format '{output_file.suffix}'. Saving to CSV by default.")
                df.to_csv(output_file.with_suffix('.csv'))
                
        except Exception as e:
            click.echo(f"\nError: Could not save output file {output_file}: {e}", err=True)


if __name__ == '__main__':
    # Example usage (assuming 'metrics_output/file1.json', 'metrics_output/file2.json', etc. exist):
    # python script_name.py -i 'metrics_output/*.json' -m hypervolume -f 'vns 10s' -f 'config 2' -o comparison.csv
     # Need this import here for the top-level script
    compare_metrics()
