from copy import deepcopy
import json
import random
from pathlib import Path
from typing import Dict, Any


def generate_class_c(n: int) -> Dict[str, Any]:
    """
    Generates a 2-objective knapsack problem instance of class C.
    The objectives are built with plateaus of uniform costs.
    """
    objectives_list = [[], []]
    weights = []

    # Generate objective 1
    current_length = 0
    while current_length < n:
        n_repeats = random.randint(1, int(0.1 * n))
        cost = random.randint(1, 100)
        for _ in range(n_repeats):
            if len(objectives_list[0]) < n:
                objectives_list[0].append(cost)
        current_length += n_repeats

    # Generate objective 2
    current_length = 0
    while current_length < n:
        n_repeats = random.randint(1, int(0.1 * n))
        cost = random.randint(1, 100)
        for _ in range(n_repeats):
            if len(objectives_list[1]) < n:
                objectives_list[1].append(cost)
        current_length += n_repeats

    # Generate weights independently
    for _ in range(n):
        weights.append(random.randint(1, 100))

    # W / (Sum{i=1,...n} w(i)) = 0.5
    capacity = int(sum(weights) / 2)

    data = {"capacity": capacity, "weights": [weights], "objectives": objectives_list}

    return {"metadata": {"objectives": 2, "weights": 1}, "data": data}


def generate_class_d(class_c_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a 2-objective knapsack problem instance of class D from a class C instance.
    The first objective is replaced by the reversed second objective.
    """
    class_d_data = deepcopy(class_c_data)

    # Get the second objective and reverse it
    reversed_obj2 = class_d_data["data"]["objectives"][1][::-1]

    # Replace the first objective with the reversed second objective
    class_d_data["data"]["objectives"][0] = reversed_obj2

    return class_d_data


def save_instance_to_file(data: Dict[str, Any], filename: str):
    """
    Saves the generated problem instance data to a JSON file.
    """
    output_dir = Path("./data/mokp")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved instance to {file_path}")


def main():
    """
    Generates and saves a set of Class C and Class D instances.
    """
    problem_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    print("Generating problem instances...")

    for n in problem_sizes:
        class_c_instance = generate_class_c(n)
        filename_c = f"2KP{n}-1C.json"
        save_instance_to_file(class_c_instance, filename_c)

        class_d_instance = generate_class_d(class_c_instance)
        filename_d = f"2KP{n}-1D.json"
        save_instance_to_file(class_d_instance, filename_d)


if __name__ == "__main__":
    main()
