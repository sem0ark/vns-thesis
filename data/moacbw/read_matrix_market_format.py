import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Union

import click


def mm_to_json_graph(input_gz_path: Union[str, Path]) -> dict:
    """
    Transforms a gzipped Matrix Market (.mtx.gz) file in coordinate format
    into a JSON graph representation (adjacency list) with 0-based indicies.

    Args:
        input_gz_path: Path to the gzipped Matrix Market file.

    Returns:
        A dictionary conforming to the target JSON structure.
    """
    input_gz_path = Path(input_gz_path)
    graph = defaultdict(set)
    is_symmetric = False
    num_nodes = 0

    with gzip.open(input_gz_path, "rt") as f:
        lines = iter(f)

        # 1. Parse Header and Size
        for line in lines:
            line = line.strip()
            if line.startswith("%%MatrixMarket"):
                header_tokens = line.lower().split()
                # Check the fifth token for the symmetry property
                if len(header_tokens) > 4 and header_tokens[4] == "symmetric":
                    is_symmetric = True
                continue

            if line.startswith("%"):
                continue

            # This is the size line: M N L
            try:
                M, N, L = map(int, line.split())
                if M != N:
                    raise ValueError(
                        "Matrix is not square (M != N), which is required for a simple graph representation."
                    )
                num_nodes = N
                break  # Size line found, break from header loop
            except ValueError:
                # Still looking for the size line, skip non-comment non-header lines
                continue

        # 2. Process Data Lines
        for line in lines:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            # Read I and J (1-based indices). Values are ignored in pattern/graph format.
            try:
                tokens = line.split()
                i = int(tokens[0]) - 1
                j = int(tokens[1]) - 1
            except ValueError as e:
                raise IOError(
                    f"Malformed data line: '{line}' in file {input_gz_path}."
                ) from e

            # Add the edge (I -> J), ignoring self-loops (i == j)
            if i != j:
                graph[i].add(j)

            # Handle symmetry property: add (J -> I) if I != J
            if is_symmetric and i != j:
                graph[j].add(i)

    # 3. Format Output

    # Convert keys to strings (for JSON consistency) and sets to sorted lists
    json_graph = sorted(
        [[node_id, sorted(list(neighbors))] for node_id, neighbors in graph.items()]
    )

    output_data = {
        "metadata": {
            "objectives": 2,  # For this problem we have 2 objective goals
            "weights": 0,  # but no actual constraints or weights
        },
        "data": {"nodes": num_nodes, "graph": json_graph},
    }

    return output_data


@click.group()
def cli():
    """CLI tool for graph file transformations."""
    pass


@cli.command(name="transform")
@click.option(
    "-i",
    "--input-paths",
    required=True,
    type=str,
    help="Path pattern (with wildcards, e.g., 'data/*.mtx.gz') to Matrix Market files.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path("./"),
    help="Directory to save the resulting JSON files. Will be created if it doesn't exist.",
)
def transform_command(input_paths: str, output_dir: Path):
    """
    Transforms gzipped Matrix Market files matching a pattern into a JSON graph format.
    """
    click.echo(f"Searching for files matching pattern: '{input_paths}'")

    # Use glob to find all matching files, handling wildcards
    file_paths = glob.glob(input_paths)

    if not file_paths:
        click.echo(f"No files found matching pattern '{input_paths}'.")
        sys.exit(0)

    click.echo(f"Found {len(file_paths)} file(s).")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for input_path_str in file_paths:
        input_path = Path(input_path_str)
        click.echo(f"Processing {input_path.name}...")

        try:
            # 1. Process file
            json_data = mm_to_json_graph(input_path)

            # 2. Determine output filename
            # Base name: 'file' from 'file.mtx.gz' or 'file.gz'
            output_name = input_path.stem
            if output_name.endswith(".mtx"):
                output_name = Path(output_name).stem  # Remove .mtx if present

            output_file = output_dir / f"{output_name}.json"

            # 3. Save JSON
            with open(output_file, "w") as f:
                json.dump(json_data, f)

            click.echo(f"Successfully saved to {output_file}")
            success_count += 1

        except Exception as e:
            click.echo(f"Skipping {input_path.name} due to an error: {e}", err=True)
            continue

    click.echo("--- Transformation Complete ---")
    click.echo(f"Successfully processed {success_count} / {len(file_paths)} files.")


if __name__ == "__main__":
    cli()
