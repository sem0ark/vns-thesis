import glob
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import click

from src.cli.filter_param import ClickFilterExpression, FilterExpression
from src.cli.metrics import display_metrics, plot_runs
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.cli.utils import parse_time_string
from src.vns.abstract import Problem

logger = logging.getLogger()


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)


def _load_runs(
    root_folder: Path,
    problem_name: str,
    instance_name: str,
    include_data=False,
) -> dict[str, list[SavedRun]]:
    """
    Loads saved run files for a given instance, returning the latest version of each unique configuration.
    """
    all_files = root_folder.glob(
        f"{problem_name}_{instance_name}*.json", case_sensitive=False
    )

    runs_by_name = defaultdict(list)
    for file_path in all_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                metadata = Metadata(**data["metadata"])
                if (
                    metadata.instance_name.lower() != instance_name.lower()
                    or metadata.problem_name.lower() != problem_name.lower()
                ):
                    click.echo(
                        f"Unexpected metadata for {file_path}: "
                        f"got ({metadata.problem_name}, {metadata.instance_name}), "
                        f"but expected ({problem_name}, {instance_name})"
                    )
                    del data
                    continue

                solutions = [
                    SavedSolution(
                        objectives=s["objectives"],
                        data=s["data"] if include_data else None,
                    )
                    for s in data["solutions"]
                ]
                del data

            config_name = (
                f"{metadata.name} v{metadata.version} {int(metadata.run_time_seconds)}s"
            )
            run = SavedRun(metadata=metadata, solutions=solutions)

            runs_by_name[config_name].append(run)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            click.echo(f"Skipping malformed or incomplete run file {file_path}: {e}")
            continue

    return runs_by_name


def _filter_runs(
    runs_grouped: dict[str, list[SavedRun]],
    filter_expression: FilterExpression,
    select_latest_only: bool = False,
) -> dict[str, list[SavedRun]]:
    """
    Filters saved run files based on name, run time, creation date.
    """
    runs_filtered = defaultdict(list)

    for config_name, runs in runs_grouped.items():
        if not filter_expression.is_match(config_name):
            continue

        for run in runs:
            runs_filtered[config_name].append(run)

    if select_latest_only:
        runs_filtered = {
            name: [sorted(runs, key=lambda run: run.metadata.date)[-1]]
            for name, runs in runs_filtered.items()
            if runs
        }

    return runs_filtered


def common_options(f):
    """Apply common CLI options for both run and show actions."""

    f = click.option(
        "-i",
        "--instance",
        required=True,
        type=str,
        help="Path pattern (with wildcards) to instance files (for run) or single path (for show).",
    )(f)

    f = click.option(
        "-f",
        "--filter-string",
        type=ClickFilterExpression(),
        default="",
        help="Boolean filter expression for config names (e.g., '(vns or nsga2) and 120s').",
    )(f)

    return f


@dataclass
class RunConfig:
    run_time_seconds: float
    instance_path: Path


class InstanceRunner:
    def __init__(self, config: RunConfig):
        pass

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        """Gives out a prepared set of runner classes."""
        raise NotImplementedError()


class CLI:
    def __init__(
        self,
        problem_name: str,
        storage_folder: Path,
        runners: list[type[InstanceRunner]],
        problem_class: type[Problem],
    ) -> None:
        self.problem_name = problem_name
        self.storage_folder = storage_folder
        self.runners = runners
        self.problem_class = problem_class

        self.storage_folder.mkdir(exist_ok=True, parents=True)

    def _execute_run_logic(self, instance: str, max_time: str, filter_expression: FilterExpression):
        """Contains the logic for running optimizations and saving results."""
        run_time_seconds = parse_time_string(max_time)

        instance_paths = sorted(glob.glob(instance))
        if not instance_paths:
            click.echo(
                f"Warning: No files found matching pattern '{instance}'. Exiting..."
            )
            return

        click.echo(
            f"Running configs for problem {self.problem_name} on {len(instance_paths)} instance(s).\n"
            + "\n".join(instance_paths)
        )

        for instance_path_str in instance_paths:
            configuration = RunConfig(run_time_seconds, Path(instance_path_str))

            problem_configs = {
                config_name: func
                for runner in self.runners
                for config_name, func in runner(configuration).get_variants()
                if filter_expression.is_match(config_name)
            }

            click.echo("-" * 50)
            click.echo(f"Processing instance: {configuration.instance_path}")

            for variant_name, runner in problem_configs.items():
                click.echo(
                    f"Running '{variant_name}' variant on '{self.problem_name}' for {configuration.run_time_seconds} seconds."
                )

                results = runner(configuration)

                instance_name = Path(instance_path_str).stem
                timestamp = results.metadata.date

                destination_path = (
                    self.storage_folder
                    / f"{self.problem_name}_{instance_name} {variant_name} "
                    f"{timestamp.split('.')[0].replace(':', '-')}.json"
                )

                with open(destination_path, "w") as f:
                    json.dump(asdict(results), f)

                print(f"Optimization run data saved to: {destination_path}")

    def _execute_plot_logic(
        self,
        instance: str,
        filter_expression: FilterExpression,
        lines: bool,
    ):
        """Contains the logic for displaying optimization runs."""

        instance_path = Path(instance)
        instance_name = instance_path.stem
        problem_instance = self.problem_class.load(instance)

        click.echo(
            f"Displaying runs for problem {self.problem_name} on instance {instance_name}"
        )

        all_runs = _load_runs(self.storage_folder, self.problem_name, instance_name)
        runs_to_show = _filter_runs(all_runs, filter_expression)

        if not runs_to_show:
            click.echo(f"No runs matched the filters '{filter_expression}'.")
            return

        plot_runs(
            instance_path,
            all_runs,
            runs_to_show,
            objective_names=problem_instance.objective_names,
            plot_lines=lines,
        )

    def _execute_show_logic(
        self,
        instance: str,
        unary: bool,
        coverage: bool,
        filter_expression: FilterExpression,
        output_file: Path | None,
    ):
        """Contains the logic for loading and displaying metrics."""

        instance_paths = sorted(glob.glob(instance))
        if not instance_paths:
            click.echo(
                f"Warning: No files found matching pattern '{instance}'. Exiting..."
            )
            return

        for instance_path_str in instance_paths:
            instance_path = Path(instance_path_str)
            instance_name = instance_path.stem

            click.echo(
                f"Displaying metrics for problem: {self.problem_name} on instance: {instance_name}"
            )

            all_runs = _load_runs(self.storage_folder, self.problem_name, instance_name)
            runs_to_show = _filter_runs(all_runs, filter_expression)

            if not runs_to_show:
                click.echo(f"No runs matched the filters '{filter_expression}'.")
                return

            click.echo("Displaying raw metrics...")
            display_metrics(
                instance_path, all_runs, runs_to_show, unary, coverage, output_file
            )

    def run(self) -> None:
        """Builds and executes the top-level CLI."""

        @click.group(help="CLI for optimization problems.")
        def cli():
            pass

        @common_options
        @cli.command(
            name="run", help="Run optimization configs on problem instance(s)."
        )
        @click.option(
            "-t",
            "--max-time",
            required=True,
            help="Maximum execution time (e.g., 30s, 1h).",
        )
        def run_command(instance: str, filter_string: FilterExpression, max_time: str):
            """
            Executes optimization runs for a specified problem and instance(s).

            Example: cli run knapsack -i 'data/*.json' -t 30s -f 'vns,k1 or vns,k3'
            """
            self._execute_run_logic(instance, max_time, filter_string)

        @cli.command(name="plot", help="Plot saved runs for a given instance.")
        @click.option(
            "-i",
            "--instance",
            required=True,
            type=click.Path(exists=True),
            help="Path to instance file.",
        )
        @click.option(
            "-f",
            "--filter-string",
            default="",
            help="Config name parts to match (e.g., 'vns,k1' will match 'vns k1 type 1', 'k2 vns type 2', etc.)",
        )
        @click.option(
            "--lines/--no-lines",
            is_flag=True,
            default=True,
            help="Connect solution front with points a line (for plotting).",
        )
        def plot_command(
            instance: str,
            filter_expression: FilterExpression,
            lines: bool,
        ):
            """
            Displays metrics for saved runs for a specified problem and instance.

            Example: cli show knapsack -i 'data/instance1.json' -t 30s -f 'vns_k1' --plot
            """
            self._execute_plot_logic(
                instance,
                filter_expression,
                lines,
            )

        @cli.command(name="metrics", help="Show metrics for saved runs.")
        @common_options
        @click.option(
            "--unary",
            is_flag=True,
            help="Displays a table of independent performance metrics for each configuration.",
        )
        @click.option(
            "--coverage",
            is_flag=True,
            help="Displays a table of 1-1 coverage comparisons.",
        )
        @click.option(
            "--export",
            is_flag=True,
            help="Export instance-specific metrics data in a common JSON format for multi-dataset comparison.",
        )
        @click.option(
            "-o",
            "--output-file",
            type=click.Path(path_type=Path),
            default=None,
            help="File path to export metrics (.csv or .xlsx).",
        )
        def metrics_command(
            instance: str,
            filter_expression: FilterExpression,
            unary: bool,
            coverage: bool,
            output_file: Path | None,
        ):
            """
            Displays metrics for saved runs for a specified problem and instance.

            Example: cli metrics knapsack -i 'data/instance1.json' -f 'vns_k1,30s' --plot
            """
            self._execute_show_logic(
                instance,
                unary,
                coverage,
                filter_expression,
                output_file,
            )

        setup_logging()
        cli()
