import json
from pathlib import Path
from typing import Any

import click

BASE = Path(__file__).parent


def parse_dat_file(file_path: Path) -> dict[str, Any]:
    """
    Parses a single multi-objective Set Covering Problem (.dat) file
    into a structured dictionary format.
    """
    with open(file_path, "r") as f:
        text = f.read()

    values = map(lambda v: int(v), text.split())

    try:
        m, n = next(values), next(values)
        costs_obj1 = [next(values) for _ in range(n)]
        costs_obj2 = [next(values) for _ in range(n)]

        sets: list[list[int]] = []
        for _ in range(m):
            num_elements_in_set = next(values)
            sets.append([next(values) for _ in range(num_elements_in_set)])

    except Exception as e:
        raise RuntimeError(f"Error parsing file {file_path}: {e}")

    return {
        "metadata": {
            "problem": "MOSCP",
            "instance": file_path.stem,
            "objectives": 2,
            "constraints": 1,
        },
        "data": {
            "num_items": m,
            "num_sets": n,
            "sets": sets,
            "costs": [
                costs_obj1,
                costs_obj2,
            ],
        },
    }


def parse_knapsack_file(file_path: Path) -> dict[str, Any]:
    """
    Parses a single Multi-Objective Knapsack Problem (.dat) file
    into a structured dictionary format.
    """
    with open(file_path, "r") as f:
        # Filter comments and join lines back
        text = " ".join(
            line.strip() for line in f if line.strip() and not line.startswith("#")
        )

    values = map(lambda v: int(v), text.split())

    try:
        # number of variables (n)
        n = next(values)

        # number of objectives (p = 2)
        p = next(values)
        if p != 2:
            click.echo(f"Warning: Expected 2 objectives for MOKP, found {p}.", err=True)

        # number of constraints (k = 1)
        k = next(values)
        if k != 1:
            click.echo(f"Warning: Expected 1 constraint for MOKP, found {k}.", err=True)

        costs = []
        for _ in range(p):
            costs.append([next(values) for _ in range(n)])

        weights = []
        for _ in range(k):
            weights.append([next(values) for _ in range(n)])

        capacity = next(values)

    except StopIteration:
        raise RuntimeError(
            f"Unexpected end of file while parsing {file_path}. Data may be incomplete."
        )
    except Exception as e:
        raise RuntimeError(f"Error parsing file {file_path}: {e}")

    return {
        "metadata": {
            "problem": "MOKP",
            "instance": file_path.stem,
            "objectives": p,
            "constraints": k,
        },
        "data": {
            "num_items": n,
            "capacity": capacity,
            "costs": costs,
            "weights": weights,
        },
    }


@click.command(
    help="Converts multi-objective Set Covering Problem (.dat) files to a structured JSON format."
)
@click.argument(
    "dat_files", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=BASE,
    help=f"Directory to save the converted JSON files. Default is '{BASE.name}/'.",
)
def convert_scp_cli(dat_files: tuple[Path], output_dir: Path):
    """
    Receives multiple .dat file paths, processes them, and saves the output
    as JSON files in the specified output directory.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    click.echo(f"Starting conversion for {len(dat_files)} file(s).")

    successful_conversions = 0
    for file_path in dat_files:
        output_file_name = f"{file_path.stem}.json"
        output_path = output_dir / output_file_name

        click.echo("-" * 50)
        click.echo(f"Processing: {file_path.name}")

        try:
            json_data = parse_dat_file(file_path)
            with open(output_path, "w") as f:
                json.dump(json_data, f)

            click.echo(f"Successfully converted and saved to: {output_path}")
            successful_conversions += 1

        except Exception as e:
            click.echo(f"ERROR: Failed to process {file_path.name}: {e}", err=True)
            continue

    click.echo("-" * 50)
    click.echo(
        f"Conversion complete. {successful_conversions}/{len(dat_files)} files processed successfully."
    )


if __name__ == "__main__":
    convert_scp_cli()
