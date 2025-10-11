import glob
import json
import logging
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import click

from src.cli.filter_param import ClickFilterExpression, FilterExpression
from src.cli.metrics import display_metrics, plot_runs
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.cli.utils import NpEncoder, parse_time_string
from src.core.abstract import Problem
from src.vns.acceptance import ParetoFront


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


def _load_run(
    file_path: Path,
    problem_name: str,
    instance_name: str,
    include_data=False,
) -> SavedRun:
    with open(file_path, "r") as f:
        data = json.load(f)
        metadata = Metadata(**data["metadata"])
        metadata.file_path = file_path

        if (
            metadata.instance_name.lower() != instance_name.lower()
            or metadata.problem_name.lower() != problem_name.lower()
        ):
            raise ValueError(
                f"Unexpected metadata for {file_path}: "
                f"got ({metadata.problem_name}, {metadata.instance_name}), "
                f"but expected ({problem_name}, {instance_name})"
            )

        solutions = [
            SavedSolution(
                objectives=s["objectives"],
                data=s["data"] if include_data else None,
            )
            for s in data["solutions"]
        ]
        del data

    return SavedRun(metadata=metadata, solutions=solutions)


def _load_runs(
    root_folder: Path,
    problem_name: str,
    instance_name: str,
    include_data=False,
) -> dict[str, list[SavedRun]]:
    """
    Loads saved run files for a given instance, returning the latest version of each unique configuration.
    """
    all_files = list(
        root_folder.glob(f"{problem_name}_{instance_name}*.json", case_sensitive=False)
    )
    reference_front_file = (
        root_folder / f"reference_front_{problem_name}_{instance_name}.json"
    )
    if reference_front_file.exists() and reference_front_file.is_file():
        all_files.append(reference_front_file)

    runs_by_name = defaultdict(list)
    for file_path in all_files:
        try:
            run = _load_run(file_path, problem_name, instance_name, include_data)
            config_name = f"v{run.metadata.version} {int(run.metadata.run_time_seconds)}s {run.metadata.name}"
            runs_by_name[config_name].append(run)

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
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
        if "REFERENCE-FRONT" in config_name:  # HACK: skip prepared reference front run
            continue

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


def _merge_runs_to_non_dominated_front(runs: list[SavedRun]) -> SavedRun:
    """
    Calculates the reference front by combining and sorting all non-dominated
    solutions from all runs. It can be overridden by a predefined front.
    """

    reference_front = ParetoFront()
    for run in runs:
        for sol in run.solutions:
            reference_front.accept(cast(Any, sol))

    result = SavedRun(
        metadata=deepcopy(runs[0].metadata),
        solutions=cast(Any, reference_front.get_all_solutions()),
    )

    return result


def common_options(f):
    """Apply common CLI options for both run and show actions."""

    f = click.option(
        "-i",
        "--instance",
        required=True,
        multiple=True,
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
        self.runners = runners
        self.problem_class = problem_class

        self.storage_folder = storage_folder
        """Place to store correct runs."""
        self.storage_folder.mkdir(exist_ok=True, parents=True)

        self.quarantine_folder = storage_folder / "incorrect"
        """Place to store incorrect saved runs, which have wrong objective values or infeasible solutions."""
        self.quarantine_folder.mkdir(exist_ok=True, parents=True)

        self.archive_folder = storage_folder / "archive"
        """Place to store incorrect saved runs, which have wrong objective values or infeasible solutions."""
        self.archive_folder.mkdir(exist_ok=True, parents=True)

    def _execute_archive_logic(
        self,
        instance_patterns: list[str],
        filter_expression: FilterExpression,
        move_runs: bool,
    ):
        """
        Validates saved runs, moves valid ones to the archive folder,
        and updates the instance's non-dominated reference front.
        """

        instance_paths = sorted(
            set(instance for p in instance_patterns for instance in glob.glob(p))
        )
        if not instance_paths:
            click.echo(
                f"Warning: No files found matching patterns {instance_patterns}. Exiting..."
            )
            return

        for instance_path_str in instance_paths:
            instance_path = Path(instance_path_str)
            instance_name = instance_path.stem

            reference_front_filename = (
                f"reference_front_{self.problem_name}_{instance_name}.json"
            )
            reference_front_path = self.storage_folder / reference_front_filename

            try:
                problem_instance = self.problem_class.load(instance_path_str)
            except Exception as e:
                click.echo(f"Error loading problem instance {instance_path_str}: {e}")
                continue

            click.echo("-" * 50)
            click.echo(
                f"Archiving runs for problem: {self.problem_name} on instance: {instance_name}"
            )

            # Load ALL runs from storage (must include data for validation/archiving)
            all_runs_grouped = _load_runs(
                self.storage_folder, self.problem_name, instance_name, include_data=True
            )

            runs_to_archive_grouped = _filter_runs(all_runs_grouped, filter_expression)

            if not runs_to_archive_grouped:
                click.echo(
                    f"No runs matched the filters '{filter_expression}' for archiving."
                )
                continue

            valid_runs_for_merge = []

            for config_name, runs in runs_to_archive_grouped.items():
                for run in runs:
                    file_path = run.metadata.file_path
                    if file_path is None:
                        click.echo(
                            f"Error: Could not determine file path for run {config_name}. Skipping."
                        )
                        continue

                    if self._validate_run(run, problem_instance):
                        valid_runs_for_merge.append(run)

                        if move_runs:
                            destination_path = self.archive_folder / file_path.name
                            shutil.move(file_path, destination_path)
                            run.metadata.file_path = destination_path
                            click.echo(
                                f"Moved {run.metadata.instance_name} to {destination_path}"
                            )

            if not valid_runs_for_merge:
                click.echo("Warning: no available valid runs for merge. Exiting.")
                return

            reference_run = valid_runs_for_merge[0]
            if reference_front_path.exists():
                try:
                    reference_run = _load_run(
                        reference_front_path,
                        self.problem_name,
                        instance_name,
                        include_data=True,
                    )
                    click.echo(
                        f"Loaded {len(reference_run.solutions)} solutions from existing reference front."
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    click.echo(
                        f"Warning: Could not load existing reference front at {reference_front_path}: {e}"
                    )

            reference_run = _merge_runs_to_non_dominated_front(
                valid_runs_for_merge + [reference_run]
            )
            reference_run.metadata.name = "REFERENCE-FRONT"
            reference_run.metadata.version = 0
            reference_run.metadata.run_time_seconds = 0

            with open(reference_front_path, "w") as f:
                json.dump(asdict(reference_run), f, cls=NpEncoder)

            click.echo(
                f"Updated reference front saved to: {reference_front_path} ({len(reference_run.solutions)} total solutions)."
            )
            click.echo("-" * 50)

    def _validate_run(
        self,
        run: SavedRun,
        problem_instance: Problem,
    ) -> bool:
        """
        Validates every solution in a run. Moves the file to quarantine if any solution fails.
        Returns True if the run is valid, False otherwise.
        """
        is_valid = True
        file_path = run.metadata.file_path
        if not file_path:
            raise ValueError(f"Missing file_path for run {run.metadata}")

        for i, saved_solution in enumerate(run.solutions):
            if not saved_solution.data:
                click.echo(
                    f"Validation FAILED: Solution #{i} data is missing in {file_path.name}."
                )
                is_valid = False
                break

            solution = problem_instance.load_solution(saved_solution.data)
            if not problem_instance.satisfies_constraints(solution.data):
                click.echo(
                    f"Validation FAILED: Solution #{i} in {file_path.name} is INFEASIBLE."
                )
                is_valid = False
                break

            try:
                calculated_objectives = problem_instance.calculate_objectives(
                    solution.data
                )
            except Exception as e:
                click.echo(
                    f"Validation FAILED: Solution #{i} in {file_path.name} failed objective calculation: {e}",
                )
                is_valid = False
                break

            if len(calculated_objectives) != len(saved_solution.objectives):
                click.echo(
                    f"Validation FAILED: Solution #{i} in {file_path.name} has objective count mismatch "
                    f"({len(saved_solution.objectives)} saved vs {len(calculated_objectives)} calculated)."
                )
                is_valid = False
                break

            for saved_obj, calc_obj in zip(
                saved_solution.objectives, calculated_objectives
            ):
                if abs(saved_obj - calc_obj) > 1e-6:
                    click.echo(
                        f"Validation FAILED: Solution #{i} in {file_path.name} has objective mismatch. "
                        f"Saved: {saved_solution.objectives}, Calculated: {calculated_objectives}"
                    )
                    is_valid = False
                    break

            if not is_valid:
                break

        if not is_valid:
            # Move the file to quarantine folder
            destination_path = self.quarantine_folder / file_path.name
            shutil.move(file_path, destination_path)
            click.echo(f"Run file moved to quarantine: {destination_path}")
            return False

        return True

    def _execute_validate_logic(
        self,
        instance_patterns: list[str],
        filter_expression: FilterExpression,
    ):
        """Contains the logic for validating saved run solutions."""

        instance_paths = sorted(
            set(instance for p in instance_patterns for instance in glob.glob(p))
        )
        if not instance_paths:
            click.echo(
                f"Warning: No files found matching patterns {instance_patterns}. Exiting..."
            )
            return

        for instance_path_str in instance_paths:
            instance_path = Path(instance_path_str)
            instance_name = instance_path.stem

            try:
                problem_instance = self.problem_class.load(instance_path_str)
            except Exception as e:
                click.echo(f"Error loading problem instance {instance_path_str}: {e}")
                continue

            click.echo("-" * 50)
            click.echo(
                f"Validating runs for problem: {self.problem_name} on instance: {instance_name}"
            )

            # Use _load_runs with include_data=True
            all_runs_grouped = _load_runs(
                self.storage_folder, self.problem_name, instance_name, include_data=True
            )

            runs_to_validate_grouped = _filter_runs(all_runs_grouped, filter_expression)

            if not runs_to_validate_grouped:
                click.echo(
                    f"No runs matched the filters '{filter_expression}' for validation."
                )
                continue

            validated_count = 0
            quarantined_count = 0

            for config_name, runs in runs_to_validate_grouped.items():
                for run in runs:
                    file_path = run.metadata.file_path
                    if file_path is None:
                        click.echo(
                            f"Error: Could not determine file path for run {config_name}. Skipping validation."
                        )
                        continue

                    click.echo(f"-> Checking run: {file_path.name}")
                    if self._validate_run(run, problem_instance):
                        validated_count += 1
                    else:
                        quarantined_count += 1

            click.echo(f"Instance {instance_name} Summary:")
            click.echo(f"  Total runs checked: {validated_count + quarantined_count}")
            click.echo(f"  Valid runs: {validated_count}")
            click.echo(f"  Invalid/Quarantined runs: {quarantined_count}")
            click.echo("-" * 50)

    def _execute_run_logic(
        self,
        instance_patterns: list[str],
        max_time: str,
        filter_expression: FilterExpression,
    ):
        """Contains the logic for running optimizations and saving results."""
        run_time_seconds = parse_time_string(max_time)

        instance_paths = sorted(
            set(instance for p in instance_patterns for instance in glob.glob(p))
        )
        if not instance_paths:
            click.echo(
                f"Warning: No files found matching patterns {instance_patterns}. Exiting..."
            )
            return

        click.echo(
            f"Running configs for problem {self.problem_name} on {len(instance_paths)} instance(s).\n"
            + "\n".join(instance_paths)
        )

        for instance_path_str in instance_paths:
            configuration = RunConfig(run_time_seconds, Path(instance_path_str))
            instance_path = Path(instance_path_str)
            instance_name = instance_path.stem

            try:
                problem_instance = self.problem_class.load(instance_path_str)
            except Exception as e:
                click.echo(f"Error loading problem instance {instance_path_str}: {e}")
                continue

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
                timestamp = results.metadata.date

                destination_path = (
                    self.storage_folder
                    / f"{self.problem_name}_{instance_name} {variant_name} "
                    f"{timestamp.split('.')[0].replace(':', '-')}.json"
                )
                results.metadata.file_path = destination_path

                with open(destination_path, "w") as f:
                    json.dump(asdict(results), f, cls=NpEncoder)

                print(f"Optimization run data saved to: {destination_path}")

                self._validate_run(results, problem_instance)

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

    def _execute_metrics_logic(
        self,
        instance_patterns: list[str],
        unary: bool,
        coverage: bool,
        export_to_json: bool,
        filter_expression: FilterExpression,
        output_file: Path | None,
    ):
        """Contains the logic for loading and displaying metrics."""

        instance_paths = sorted(
            set(instance for p in instance_patterns for instance in glob.glob(p))
        )
        if not instance_paths:
            click.echo(
                f"Warning: No files found matching patterns {instance_patterns}. Exiting..."
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
                instance_path,
                all_runs,
                runs_to_show,
                unary,
                coverage,
                export_to_json,
                output_file,
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
        @click.option(
            "--trace",
            is_flag=True,
            default=False,
        )
        def run_command(
            instance: list[str],
            filter_string: FilterExpression,
            max_time: str,
            trace: bool,
        ):
            """
            Executes optimization runs for a specified problem and instance(s).

            Example: script.py run -i 'data/*.json' -t 30s -f 'vns,k1 or vns,k3'
            """
            if trace:
                try:
                    from viztracer import VizTracer

                    with VizTracer(min_duration=20):
                        self._execute_run_logic(instance, max_time, filter_string)
                except ImportError:
                    click.echo(
                        "Warning: to use --trace, please, install viztracer with pip install viztracer"
                    )
            else:
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
            type=ClickFilterExpression(),
            default="",
            help="Boolean filter expression for config names (e.g., '(vns or nsga2) and 120s').",
        )
        @click.option(
            "--lines/--no-lines",
            is_flag=True,
            default=True,
            help="Connect solution front with points a line (for plotting).",
        )
        def plot_command(
            instance: str,
            filter_string: FilterExpression,
            lines: bool,
        ):
            """
            Displays metrics for saved runs for a specified problem and instance.

            Example: script.py show -i 'data/instance1.json' -t 30s -f 'vns_k1' --plot
            """
            self._execute_plot_logic(
                instance,
                filter_string,
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
            instance: list[str],
            filter_string: FilterExpression,
            unary: bool,
            coverage: bool,
            export: bool,
            output_file: Path | None,
        ):
            """
            Displays metrics for saved runs for a specified problem and instance.

            Example: script.py metrics -i 'data/instance1.json' -f 'vns_k1,30s' --plot
            """
            self._execute_metrics_logic(
                instance,
                unary,
                coverage,
                export,
                filter_string,
                output_file,
            )

        @cli.command(
            name="validate", help="Validate saved run solutions for correctness."
        )
        @common_options
        def validate_command(instance: list[str], filter_string: FilterExpression):
            """
            Validates saved solutions against the problem's feasibility and objective functions.
            Invalid run files are moved to the quarantine folder.

            Example: script.py validate -i 'data/*.json' -f 'vns,k1 or nsga2'
            """
            self._execute_validate_logic(instance, filter_string)

        @cli.command(
            name="archive",
            help="Validate and update merged reference front.",
        )
        @common_options
        @click.option(
            "--move",
            is_flag=True,
            help="Additionally move all matched files into archive folder.",
        )
        def archive_command(
            instance: list[str], filter_string: FilterExpression, move: bool
        ):
            """
            Validates saved solutions. Valid runs are moved to the archive folder.
            Solutions from the archived runs are merged into a single reference front file.

            Example: script.py archive -i 'data/*.json' -f 'vns,k1 or nsga2'
            """
            self._execute_archive_logic(instance, filter_string, move)

        setup_logging()
        cli()
