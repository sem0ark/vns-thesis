from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Callable, Self

import click
import numpy as np

from src.cli.metrics import display_metrics, calculate_metrics
from src.cli.plot import plot_runs
from src.cli.shared import SavedRun, SavedSolution, Metadata

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
        if isinstance(o, np.bool):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


def _load_and_filter_runs(
    problem_name: str,
    instance_name: str,
    max_time_seconds: int,
    filter_configs: str = "",
) -> dict[str, list[SavedRun]]:
    """
    Loads and filters saved run files based on criteria, returning the latest
    version of each unique configuration.
    """
    problem_name = "mokp"

    all_files = BASE.glob(f"{problem_name}_{instance_name}_*.json")

    runs_by_name = {}
    filter_groups = [
        [f.strip() for f in filter_group.strip().lower().split(",")]
        for filter_group in filter_configs.split(" or ")
    ]

    for file_path in all_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            metadata = Metadata(**data["metadata"])
            config_name = metadata.name

            if filter_groups and not any(
                all(filter_name in config_name.lower() for filter_name in filter_group)
                for filter_group in filter_groups
            ):
                continue

            if abs(metadata.run_time_seconds - max_time_seconds) > 1e-3:
                continue

            solutions = [
                SavedSolution(objectives=s["objectives"]) for s in data["solutions"]
            ]
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
        @click.option(
            "-p",
            "--problem",
            required=True,
            type=str,
            help="Name of the problem class the instances come from.",
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
        @click.option(
            "-d",
            "--make-objectives-positive",
            help="Define, whether the plotted objectives should be positive or not. It allows to flip objective values that have been negated with the goal of maximization.",
            default=True,
            type=bool,
        )
        @click.pass_context
        def show_command(
            ctx, problem, instance, max_time, filter_configs, make_objectives_positive
        ):
            ctx.ensure_object(dict)
            ctx.obj["problem_name"] = problem
            ctx.obj["instance_path"] = Path(instance)
            ctx.obj["instance_name"] = Path(instance).stem

            ctx.obj["max_time"] = max_time
            ctx.obj["filter_configs"] = filter_configs
            ctx.obj["max_time_seconds"] = parse_time_string(max_time)

            ctx.obj["make_objectives_positive"] = make_objectives_positive

        @show_command.command(name="plot", help="Plot the metrics.")
        @click.pass_context
        def show_plot(ctx):
            problem_name = ctx.obj["problem_name"]
            instance_name = ctx.obj["instance_name"]
            max_time_seconds = ctx.obj["max_time_seconds"]
            filter_configs = ctx.obj["filter_configs"]
            make_objectives_positive = ctx.obj["make_objectives_positive"]

            click.echo("Plotting metrics...")
            runs = _load_and_filter_runs(
                problem_name,
                instance_name,
                max_time_seconds,
                filter_configs,
            )
            plot_runs(ctx.obj["instance_path"], runs, make_objectives_positive)

        @show_command.command(name="metrics", help="Display raw metrics.")
        @click.pass_context
        def show_metrics(ctx):
            problem_name = ctx.obj["problem_name"]
            instance_name = ctx.obj["instance_name"]
            max_time_seconds = ctx.obj["max_time_seconds"]
            filter_configs = ctx.obj["filter_configs"]

            click.echo("Displaying metrics...")

            runs = _load_and_filter_runs(
                problem_name, instance_name, max_time_seconds, filter_configs
            )
            display_metrics(calculate_metrics(ctx.obj["instance_path"], runs))

        @click.group(help="Run an optimization algorithm for a given problem.")
        @click.option(
            "-t",
            "--max-time",
            required=True,
            help="Maximum execution time (e.g., 30s, 1h).",
        )
        @click.option(
            "-i",
            "--instance",
            required=True,
            type=click.Path(exists=True),
            help="Path to the problem instance file.",
        )
        def run_command(max_time, instance):
            pass

        def create_problem_command(problem_name, configs):
            @run_command.command(
                name=problem_name, help=f"Run the '{problem_name}' optimization."
            )
            @click.option(
                "-f",
                "--filter-configs",
                help="List of config name substrings to match example 'k1,fi or k2,bi'.\n",
                default=None,
            )
            @click.pass_context
            def problem_runner(ctx, filter_configs):
                args = ctx.parent.params
                run_time_seconds = parse_time_string(args["max_time"])
                instance = args["instance"]

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

                click.echo(f"Running configs for problem: {problem_name}")
                for config_name, runner in configs_filtered.items():
                    click.echo(
                        f"Running {config_name} on problem '{problem_name}' from instance '{instance}' for {run_time_seconds} seconds."
                    )
                    results: SavedRun = runner(instance, run_time_seconds)
                    instance_name, timestamp = (
                        results.metadata.instance_name,
                        results.metadata.date,
                    )
                    destination_path = (
                        BASE
                        / f"mokp_{instance_name}_{timestamp.split('.')[0].replace(':', '-')}.json"
                    )

                    with open(destination_path, "w") as f:
                        json.dump(asdict(results), f, cls=NpEncoder)

                    print(f"Optimization run data saved to: {destination_path}")

            return problem_runner

        for problem_name, configs in self.runners.items():
            run_command.add_command(create_problem_command(problem_name, configs))

        cli.add_command(show_command, name="show")
        cli.add_command(run_command, name="run")
        cli()
