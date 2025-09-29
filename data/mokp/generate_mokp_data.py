import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _generate_uniform_costs(n_items: int) -> List[int]:
    return [random.randint(1, 100) for _ in range(n_items)]


def _generate_plateau_costs(n_items: int) -> List[int]:
    """
    The length l of a plateau is generated following an uniform distribution
    in {1,...,0.1n} where n is the size of the problem.
    A cost is generated following an uniform distribution
    in {1,...,100} and is repeated l times.

    The weights are generated independently following an uniform distribution in {1,...,100}.
    The principle is repeated until both objective costs are generated.
    """
    costs = []
    max_l = max(1, int(0.1 * n_items))

    while len(costs) < n_items:
        plateau_length = random.randint(1, max_l)
        cost = random.randint(1, 100)

        for _ in range(plateau_length):
            if len(costs) < n_items:
                costs.append(cost)
            else:
                break

    return costs


def _generate_weights(
    n_items: int, n_weights: int
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates weights and capacity based on the tightness ratio W / (Sum{w(i)}) = 0.5.
    """
    all_weights = []
    all_capacities = []

    for _ in range(n_weights):
        weights = [random.randint(1, 100) for _ in range(n_items)]
        capacity = int(sum(weights) / 2)

        all_weights.append(weights)
        all_capacities.append(capacity)

    return all_weights, all_capacities


def generate_mokp_instance(
    n_items: int, n_objectives: int, n_weights: int, problem_type: str
) -> Dict[str, Any]:
    """
    Generates a Multi-objective Knapsack Problem instance based on the specified type.
    """
    if n_objectives < 1:
        raise ValueError("Number of objectives must be at least 1.")

    problem_type = problem_type.upper()

    weights_matrix, capacity_vector = _generate_weights(n_items, n_weights)

    objectives_matrix = []
    cost_generator = (
        _generate_plateau_costs
        if problem_type in ("C", "D")
        else _generate_uniform_costs
    )

    for i in range(n_objectives):
        objectives_matrix.append(cost_generator(n_items))

    if n_objectives >= 2:
        if problem_type == "B":
            # Type B: Obj 1 is uniform, Obj 2 is reverse of Obj 1. Obj i > 2 are uniform.
            objectives_matrix[1] = objectives_matrix[0][::-1]

        elif problem_type == "D":
            # Type D: Obj 2 is plateaus, Obj 1 is reverse of Obj 2. Obj i > 2 are plateaus.
            objectives_matrix[0] = objectives_matrix[1][::-1]

    data = {
        "capacity": capacity_vector,
        "weights": weights_matrix,
        "objectives": objectives_matrix,
    }

    return {
        "metadata": {
            "items": n_items,
            "objectives": n_objectives,
            "constraints": n_weights,
        },
        "data": data,
    }


def _save_instance_to_dat_file(data: Dict[str, Any], filepath: Path):
    """
    Saves the generated MOKP instance data to the original .dat format.
    This function assumes 2 objectives (p=2) and 1 constraint (k=1) for simplicity,
    as the original format description is tied to the bi-objective single-constraint problem.
    It will adapt for the general case but prints a warning for k > 1.
    """
    metadata = data["metadata"]
    d = data["data"]

    n = metadata["items"]
    p = metadata["objectives"]
    k = metadata["constraints"]

    if k > 1:
        print(
            f"Warning: .dat format is typically for single constraint (k=1). Outputting {k} constraints sequentially.",
            file=sys.stderr,
        )

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{p}\n")
        f.write(f"{k}\n")

        for i in range(p):
            f.write(f"\n# Cost for objective {i + 1}\n")
            f.write("\n".join(map(str, d["objectives"][i])) + "\n")

        for i in range(k):
            f.write(f"\n# Weight for constraint {i + 1}\n")
            f.write("\n".join(map(str, d["weights"][i])) + "\n")

        f.write("\n# Total capacity W\n")
        f.write("\n".join(map(str, d["capacity"])) + "\n")

    print(f"Saved instance to {filepath}")


def _save_instance_to_json_file(data: Dict[str, Any], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved instance to {filepath}")


def save_instance_to_file(data: Dict[str, Any], filepath: Path, output_format: str):
    if output_format == "json":
        return _save_instance_to_json_file(data, filepath)

    if output_format == "dat":
        return _save_instance_to_dat_file(data, filepath)

    raise ValueError(f"Unsupported output format: {output_format}")


def setup_cli() -> argparse.ArgumentParser:
    """Sets up the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate Multi-objective Knapsack Problem (MOKP) benchmark instances.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--objectives", "-k", type=int, default=2, help="Number of objectives (k)."
    )
    parser.add_argument(
        "--weights", "-c", type=int, default=1, help="Number of weight constraints (c)."
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["A", "B", "C", "D"],
        required=True,
        help="Problem type based on cost generation:\n"
        "  A: Uniform costs/profits and weights.\n"
        "  B: Uniform, Obj 2 is reverse of Obj 1.\n"
        "  C: Plateau costs/profits, uniform weights.\n"
        "  D: Plateau, Obj 1 is reverse of Obj 2.",
    )
    parser.add_argument(
        "--instances",
        "-n",
        type=int,
        nargs="+",
        required=True,
        help="List of problem sizes (number of items) to generate, e.g., '100 200 500'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/mokp/",
        help="Output directory or a specific file name.",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "dat"],
        default="json",
        help="Output file format (json or dat).",
    )
    return parser


def main():
    parser = setup_cli()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_path = Path(args.output)
    output_ext = f".{args.format}"

    is_single_file_output = output_path.suffix == output_ext

    if is_single_file_output:
        if len(args.instances) != 1:
            print(
                f"Error: When specifying a single output file name ({output_path}), only one instance size (see --help) is allowed.",
                file=sys.stderr,
            )
            sys.exit(1)

        n = args.instances[0]
        instance = generate_mokp_instance(n, args.objectives, args.weights, args.type)
        save_instance_to_file(instance, output_path, args.format)
    else:
        for n in args.instances:
            # Filename generation follows the convention 2KPn-1A (or {k}KPn-{c}{type})
            filename = f"{args.objectives}KP{n}-{args.weights}{args.type}{output_ext}"
            file_path = output_path / filename
            instance = generate_mokp_instance(
                n, args.objectives, args.weights, args.type
            )
            save_instance_to_file(instance, file_path, args.format)


if __name__ == "__main__":
    main()
