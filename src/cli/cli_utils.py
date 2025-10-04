import glob
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Callable, Iterable

import click

from src.cli.metrics import display_metrics, plot_runs
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.cli.utils import parse_time_string


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
    all_files = root_folder.glob(f"{problem_name}_{instance_name}*.json")

    runs_by_name = {}
    for file_path in all_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                metadata = Metadata(**data["metadata"])
                solutions = [
                    SavedSolution(
                        objectives=s["objectives"],
                        data=s["data"] if include_data else None,
                    )
                    for s in data["solutions"]
                ]
                del data

            config_name = metadata.name
            run = SavedRun(metadata=metadata, solutions=solutions)

            # Keep only the newest version for each config name
            if (
                config_name not in runs_by_name
                or runs_by_name[config_name][0].metadata.version < metadata.version
            ):
                runs_by_name[config_name] = [run]
            else:
                runs_by_name[config_name].append(run)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            click.echo(f"Skipping malformed or incomplete run file {file_path}: {e}")
            continue

    return runs_by_name


def _filter_runs(
    runs_grouped: dict[str, list[SavedRun]],
    max_time_seconds: float,
    filter_configs: str = "",
    select_latest_only: bool = False,
) -> dict[str, list[SavedRun]]:
    """
    Filters saved run files based on name, run time, creation date.
    """
    runs_filtered = defaultdict(list)
    filter_groups = [
        [f.strip() for f in filter_group.strip().lower().split(",")]
        for filter_group in filter_configs.split(" or ")
    ]

    for config_name, runs in runs_grouped.items():
        if filter_groups and not any(
            all(filter_name in config_name.lower() for filter_name in filter_group)
            for filter_group in filter_groups
        ):
            continue

        for run in runs:
            if abs(run.metadata.run_time_seconds - max_time_seconds) > 1e-3:
                continue

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
        "-t",
        "--max-time",
        required=True,
        help="Maximum execution time (e.g., 30s, 1h).",
    )(f)

    f = click.option(
        "-f",
        "--filter-configs",
        default=None,
        help="Config name parts to match (e.g., 'vns,k1' will match 'vns k1 type 1', 'k2 vns type 2', etc.)",
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


def is_applicable_to_filter(name: str, filter_string: str):
    filter_string = filter_string.strip().lower()
    split_name = name.strip().lower().split()
    return not filter_string or any(
        all(f.strip() in split_name for f in group.split(","))
        for group in filter_string.split(" or ")
    )


class CLI:
    def __init__(
        self,
        problem_name: str,
        storage_folder: Path,
        runners: list[type[InstanceRunner]],
    ) -> None:
        self.problem_name = problem_name
        self.storage_folder = storage_folder
        self.runners = runners

        self.storage_folder.mkdir(exist_ok=True, parents=True)

    def _execute_run_logic(self, instance: str, max_time: str, filter_string: str):
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
                if is_applicable_to_filter(config_name, filter_string)
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

    def _execute_show_logic(
        self,
        instance: str,
        max_time: str,
        filter_configs: str,
        plot: bool,
        headers: str,
        output_file: Path | None,
        lines: bool,
    ):
        """Contains the logic for loading and displaying metrics."""

        run_time_seconds = parse_time_string(max_time)
        instance_path = Path(instance)
        instance_name = instance_path.stem

        click.echo(
            f"Displaying metrics for problem: {self.problem_name} on instance: {instance_name}"
        )

        all_runs = _load_runs(self.storage_folder, self.problem_name, instance_name)
        runs_to_show = _filter_runs(all_runs, run_time_seconds, filter_configs)

        if not runs_to_show:
            click.echo("No runs matched the filters or max-time criteria.")
            return

        if plot:
            click.echo("Plotting metrics...")
            plot_runs(
                instance_path,
                all_runs,
                runs_to_show,
                objective_names=headers,
                lines=lines,
            )
        else:
            click.echo("Displaying raw metrics...")
            display_metrics(instance_path, all_runs, runs_to_show, output_file)

    def run(self) -> None:
        """Builds and executes the top-level CLI."""

        @click.group(help="CLI for optimization problems.")
        def cli():
            pass

        @cli.command(
            name="run", help="Run optimization configs on problem instance(s)."
        )
        @common_options
        def run_command(instance: str, max_time: str, filter_configs: str):
            """
            Executes optimization runs for a specified problem and instance(s).

            Example: cli run knapsack -i 'data/*.json' -t 30s -f 'vns,k1 or vns,k3'
            """
            self._execute_run_logic(instance, max_time, filter_configs)

        @cli.command(name="show", help="Show metrics (table or plot) for saved runs.")
        @common_options
        @click.option(
            "--plot", is_flag=True, help="Displays a plot instead of a metrics table."
        )
        @click.option(
            "--headers",
            default="",
            type=str,
            help="Objective names for plotting/display.",
        )
        @click.option(
            "-o",
            "--output-file",
            type=click.Path(path_type=Path),
            default=None,
            help="File path to export metrics (.csv or .xlsx).",
        )
        @click.option(
            "--lines/--no-lines",
            is_flag=True,
            default=True,
            help="Connect solution front with points a line (for plotting).",
        )
        def show_command(
            instance: str,
            max_time: str,
            filter_configs: str,
            plot: bool,
            headers: str,
            output_file: Path | None,
            lines: bool,
        ):
            """
            Displays metrics for saved runs for a specified problem and instance.

            Example: cli show knapsack -i 'data/instance1.json' -t 30s -f 'vns_k1' --plot
            """
            self._execute_show_logic(
                instance,
                max_time,
                filter_configs,
                plot,
                headers,
                output_file,
                lines,
            )

        setup_logging()
        cli()
