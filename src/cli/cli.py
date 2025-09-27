import glob
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Self

import click
import numpy as np

from src.cli.metrics import display_metrics, plot_runs
from src.cli.shared import Metadata, SavedRun, SavedSolution

BASE = Path(__file__).parent.parent.parent / "runs"
BASE.mkdir(exist_ok=True, parents=True)


def parse_time_string(time_str: str) -> int:
    """Parses a time string like '5s', '2m', '1h' into seconds."""
    if not time_str:
        raise ValueError(
            "Invalid time format. Use digits followed by 's' (seconds), 'm' (minutes), or 'h' (hours)."
        )

    match = re.match(r"(\d+)([smh])", time_str)
    if not match:
        raise ValueError(
            "Invalid time format. Use digits followed by 's' (seconds), 'm' (minutes), or 'h' (hours). "
            "Example: '30s', '5m', '1h'."
        )
    value = int(match.group(1))
    unit = match.group(2)
    return {
        "s": 1,
        "m": 60,
        "h": 3600,
    }[unit] * value


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.bool)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


def _load_runs(
    problem_name: str,
    instance_name: str,
) -> dict[str, list[SavedRun]]:
    """
    Loads saved run files for a given instance, returning the latest version of each unique configuration.
    """
    all_files = BASE.glob(f"{problem_name}_{instance_name}_*.json")

    runs_by_name = {}
    for file_path in all_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            metadata = Metadata(**data["metadata"])
            config_name = metadata.name

            solutions = {
                tuple(s["objectives"]): SavedSolution(objectives=s["objectives"])
                for s in data["solutions"]
            }
            run = SavedRun(metadata=metadata, solutions=list(solutions.values()))

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


class CLI:
    def __init__(self) -> None:
        self.runners: dict[str, list[tuple[str, Callable[[str, float], SavedRun]]]] = {}

    def register_runner(
        self, name: str, configs: list[tuple[str, Callable[[str, float], SavedRun]]]
    ) -> Self:
        self.runners.setdefault(name, []).extend(configs)
        return self

    def run(self) -> None:
        @click.group(help="CLI for optimization problems.")
        def cli():
            pass

        @click.group(help="Show and plot metrics.")
        def show_command():
            pass

        def create_problem_show_command(problem_name):
            """Generates a command group for a specific problem under 'show'."""

            @show_command.group(
                name=problem_name,
                help=f"Show metrics for the '{problem_name}' problem.",
            )
            @click.option(
                "-i",
                "--instance",
                required=True,
                type=click.Path(exists=True),
                help="Path to the problem instance file.",
            )
            @click.option(
                "-t",
                "--max-time",
                help="Filter by maximum execution time (e.g., 30s, 1h).",
                required=True,
            )
            @click.option(
                "-f",
                "--filter-configs",
                help="A comma-separated list of config names or substrings to match.",
                default="",
                type=str,
            )
            @click.pass_context
            def problem_show_group(ctx, instance, max_time, filter_configs):
                # Context object is created and populated here
                ctx.ensure_object(dict)
                ctx.obj["problem_name"] = problem_name
                ctx.obj["instance_path"] = Path(instance)
                ctx.obj["instance_name"] = Path(instance).stem

                ctx.obj["max_time"] = max_time
                ctx.obj["filter_configs"] = filter_configs
                ctx.obj["max_time_seconds"] = parse_time_string(max_time)

            @problem_show_group.command(name="plot", help="Plot the metrics.")
            @click.option(
                "--headers",
                help="A comma-separated list of objective names to replace default Z1, Z2, etc.",
                default="",
                type=str,
            )
            @click.option(
                "--lines/--no-lines",
                help="Connect solution front with points a line.",
                is_flag=True,
                default=True,
            )
            @click.pass_context
            def show_plot(ctx, headers, lines):
                problem_name = ctx.obj["problem_name"]
                instance_name = ctx.obj["instance_name"]
                max_time_seconds = ctx.obj["max_time_seconds"]
                filter_configs = ctx.obj["filter_configs"]

                click.echo("Plotting metrics...")
                all_runs = _load_runs(problem_name, instance_name)
                runs_to_show = _filter_runs(
                    all_runs, max_time_seconds, filter_configs, select_latest_only=True
                )
                plot_runs(
                    ctx.obj["instance_path"],
                    all_runs,
                    runs_to_show,
                    objective_names=headers,
                    lines=lines,
                )

            @problem_show_group.command(name="metrics", help="Display raw metrics.")
            @click.option(
                "-o",
                "--output-file",
                type=click.Path(path_type=Path),
                default=None,
                help="File path (including extension .csv or .xlsx) to export the tables.",
            )
            @click.pass_context
            def show_metrics(ctx, output_file):
                problem_name = ctx.obj["problem_name"]
                instance_name = ctx.obj["instance_name"]
                max_time_seconds = ctx.obj["max_time_seconds"]
                filter_configs = ctx.obj["filter_configs"]

                click.echo("Displaying metrics...")

                all_runs = _load_runs(problem_name, instance_name)
                runs_to_show = _filter_runs(all_runs, max_time_seconds, filter_configs)

                display_metrics(
                    ctx.obj["instance_path"], all_runs, runs_to_show, output_file
                )

            return problem_show_group

        @click.group(help="Run an optimization algorithm for a given problem.")
        def run_command():
            pass

        def create_problem_run_command(problem_name, configs):
            @run_command.command(
                name=problem_name, help=f"Run the '{problem_name}' optimization."
            )
            @click.option(
                "-t",
                "--max-time",
                required=True,
                help="Maximum execution time (e.g., 30s, 1h).",
            )
            @click.option(
                "-f",
                "--filter-configs",
                help="List of config name substrings to match example 'k1,fi or k2,bi'.\n",
                default=None,
            )
            @click.option(
                "-i",
                "--instances",
                required=True,
                type=str,
                help="Path pattern (with wildcards) to problem instance files.",
            )
            def problem_runner(max_time: str, filter_configs: str, instances: str):
                run_time_seconds = parse_time_string(max_time)

                filter_groups = [
                    [f.strip() for f in filter_group.strip().lower().split(",")]
                    for filter_group in (filter_configs or "").split(" or ")
                ]
                configs_filtered = {
                    config_name: runner
                    for config_name, runner in configs
                    if not filter_groups
                    or any(
                        all(
                            filter_name in config_name.lower()
                            for filter_name in filter_group
                        )
                        for filter_group in filter_groups
                    )
                }

                if not configs_filtered:
                    raise click.UsageError(
                        f"Failed to match any runner with filters: {filter_groups}"
                    )

                instance_paths = sorted(glob.glob(instances))
                if not instance_paths:
                    click.echo(
                        f"Warning: No files found matching pattern '{instances}'. Exiting..."
                    )
                    return

                click.echo(
                    f"Running configs for problem: {problem_name} on {len(instance_paths)} instance(s)."
                )

                for instance_path_str in instance_paths:
                    instance_path = Path(instance_path_str)
                    click.echo("-" * 50)
                    click.echo(f"Processing instance: {instance_path}")

                    for config_name, runner in configs_filtered.items():
                        click.echo(
                            f"Running {config_name} on problem '{problem_name}' from instance '{instance_path}' for {run_time_seconds} seconds."
                        )
                        results: SavedRun = runner(instance_path_str, run_time_seconds)
                        instance_name, timestamp = (
                            results.metadata.instance_name,
                            results.metadata.date,
                        )

                        destination_path = (
                            BASE
                            / f"{problem_name}_{instance_name}_{config_name}_{timestamp.split('.')[0].replace(':', '-')}.json"
                        )

                        with open(destination_path, "w") as f:
                            json.dump(asdict(results), f, cls=NpEncoder)

                        print(f"Optimization run data saved to: {destination_path}")

            return problem_runner

        for problem_name, configs in self.runners.items():
            run_command.add_command(create_problem_run_command(problem_name, configs))
            show_command.add_command(create_problem_show_command(problem_name))

        cli.add_command(show_command, name="show")
        cli.add_command(run_command, name="run")
        cli()
