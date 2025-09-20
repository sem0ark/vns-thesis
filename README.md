# Set of components for VNS and MO-VNS Optimization

This set of utilities provides a a set of components for solving single- and multi-objective optimization problems using a variety of Variable Neighborhood Search (VNS) algorithms. The design allows you to mix and match different components to create and test various heuristic approaches, including those tailored for multi-objective problems like the Multi-objective Knapsack Problem (MOKP).

## Components

The toolkit is built on interchangeable modules, enabling the rapid development of new VNS variations. The core components are:

The `VNSOptimizer` class acts as the main orchestrator, tying all the components together to execute the full VNS algorithm. Its main optimization loop follows a classic VNS structure:

1.  **Shaking**: In each iteration `k`, a shake function (e.g., `shake_swap`) is applied to a solution from the current solution archive. The `shake_function`'s role is to generate a new, sufficiently different starting point for the local search.

2.  **Local Search**: The shaken solution is passed to a search function. This can be a single local search operator (`best_improvement`) or a composite VND function that systematically explores multiple neighborhoods.

3.  **Acceptance**: The local optimum found is then evaluated by the acceptance criterion. The solution is either accepted (and added to the archive/buffer) or rejected. If accepted, the process restarts from a new current solution. If rejected, the algorithm proceeds to a larger neighborhood by increasing `k`.


### Local Search Operators

These functions define the neighborhood structure for the local search phase, which aims to find a better solution in the vicinity of the current one. The search functions are wrappers around neighborhood operators, iterating through the neighborhood to find an improving solution.

- `noop`: A null operator for Randomized VNS (RVNS) variants where the local search step is skipped. It simply returns the `initial` solution without any changes.
- `best_improvement(operator)`: Explores the entire neighborhood defined by a `NeighborhoodOperator` and returns the single best solution found. It's a greedy approach that guarantees the best local improvement at the cost of being computationally more expensive.
- `first_improvement(operator)`: Explores the neighborhood and returns the very first solution that improves upon the current one. This can be significantly faster than `best_improvement` because it stops as soon as an improvement is found.
- `first_improvement_quick(operator)`: Similar to `first_improvement` but it's an even more aggressive version. It only checks one level of neighborhood and returns the first improving neighbor. It's designed to be a very fast local search step.
- `composite(search_functions)`: This function allows a **Variable Neighborhood Descent (VND)** strategy. It combines multiple local search functions and applies them sequentially. If a local search finds an improvement, the process restarts from the first search function in the list, effectively functioning as a classic VND.


### Acceptance Criteria

The toolkit includes a set of acceptance criteria designed to manage the solution archive, particularly for multi-objective problems where a Pareto front of non-dominated solutions must be maintained. They determine whether a new candidate solution should be accepted into the archive and influence the overall search behavior.

- `AcceptBatchBigger` & `AcceptBatchSmaller`: These criteria manage an archive of non-dominated solutions, representing the current Pareto front. They are used in the classic VNS framework where the algorithm iterates through each solution in the front before moving to a new one. The archive is updated by combining accepted solutions with the previous non-dominated solutions. `Bigger` is for maximization problems and `Smaller` for minimization.

- `AcceptBeamBigger` & `AcceptBeamSmaller`: These criteria introduce a **beam search-like behavior**. Instead of a strict iteration through the entire front, the algorithm selects a random solution from the archive (buffer). When a new solution is accepted, the archive is updated immediately. This provides a more dynamic, less structured search, similar to a beam search, where the focus is on rapidly improving the quality of the current set of solutions. `Bigger` is for maximization and `Smaller` for minimization.

- `AcceptBatchSkewedBigger` & `AcceptBatchSkewedSmaller`: Implement a **Skewed Acceptance** for Skewed VNS (SVNS). On top of checking for strict Pareto dominance, these criteria also maintain a "skewed front" of solutions that are not strictly non-dominated but are sufficiently different from existing ones. If a new solution is rejected by the main front, but is different enough, it's added to a buffer. The algorithm can then randomly select from this buffer, encouraging exploration of different regions of the search space.

- `AcceptBeamSkewedBigger` & `AcceptBeamSkewedSmaller`: These criteria combine the concepts of **beam search and skewed acceptance**. They behave like the `AcceptBeam` variants, but also maintain a buffer of skewed solutions. This allows for both rapid, random exploration from the main non-dominated front and strategic dives into promising but previously rejected regions of the solution space.

## Getting Started

### Installation

This project uses `uv` for dependency management. To get started, navigate to the project directory and install the required packages.

```bash
uv sync
```

### CLI Functionality

The main command-line interface (CLI) is accessed via `python ./cli.py` and is divided into two primary sub-commands: `show` and `run`.

#### `show`

The `show` command is used to analyze results from previously executed optimization runs. It takes a problem name, an instance file, and a maximum execution time to filter the results.

- `show plot`: Generates an interactive plot showing the Pareto fronts of the filtered runs, including a reference front for comparison.
    ```bash
    uv run python ./cli.py show -i ./data/mokp/2KP50-11.json -t 10s -p mokp plot
    ```
- `show metrics`: Displays a table of performance metrics (e.g., Hypervolume, Spacing, IGD) for the filtered runs, allowing for quantitative comparison.
    ```bash
    uv run python ./cli.py show -i ./data/mokp/2KP50-11.json -t 10s -p mokp metrics
    ```

Both `show` sub-commands can be further refined using the `-f` or `--filter-configs` option. This option accepts a comma-separated list of keywords to match specific algorithm configurations. Multiple groups of filters can be combined using `or`.

#### `run`

The `run` command executes one or more optimization algorithms on a specified problem instance. It requires an instance file and a maximum execution time. It also takes the problem name as a sub-command.

- `run <problem_name>`: Runs all registered configurations for the specified problem.
    ```bash
    uv run python ./cli.py run -i ./data/mokp/2KP50-11.json -t 10s mokp
    ```
- Filtering runs: Similar to the `show` command, you can use the `-f` or `--filter-configs` option to run only a subset of algorithms. This is particularly useful for testing specific configurations.
    ```bash
    uv run python ./cli.py run -i ./data/mokp/2KP50-11.json -t 10s mokp -f "ngsa or spea or k3,batch,noop"
    ```
    This example would execute three sets of runs: all configurations containing "ngsa", all containing "spea", and all containing both "k3" and "batch" and "noop".

Upon completion, the results of each run are saved to a timestamped JSON file in the `./runs` directory, which can then be analyzed with the `show` command.

## Writing and Registering a Custom Problem

To add a new problem and its corresponding algorithms to the CLI, you need to follow a structured approach that integrates with the existing framework. This involves defining the problem, implementing its solutions and evaluation logic, and registering the optimization runners.

### Step 1: Define the Problem and Solution Classes

First, create a new module (e.g., `src/examples/my_problem.py`) to define the problem. Your problem class must inherit from `Problem`, and its solution class must inherit from `Solution`. This ensures compatibility with the VNS framework.

```python
# src/examples/my_problem.py

from src.vns.abstract import Problem, Solution

class MyProblemSolution(Solution):
    ... # Implement data storage and equals() method

class MyProblem(Problem):
    def __init__(self):
        super().__init__(self.evaluate, self.generate_initial_solutions)

    def evaluate(self, solution: MyProblemSolution) -> tuple[float, ...]:
        ... # Implement your objective function logic

    def generate_initial_solutions(self, num_solutions) -> Iterable[MyProblemSolution]:
        ... # Implement logic to create initial solutions

```

### Step 2: Implement the Optimization Runners

Next, create the functions that will act as the optimization runners for your problem. A runner is a function that takes an `instance_path` and `run_time` and returns a `SavedRun` object containing the results. You can implement any algorithm you choose, such as VNS, GA, or a custom heuristic.

```python
# src/examples/my_problem.py

from src.cli.cli import SavedRun
from src.vns.abstract import VNSConfig
# Import your problem and solution classes

def my_custom_solver(instance_path: str, run_seconds: float) -> SavedRun:
    # 1. Load the problem instance from the file
    problem = MyProblem.load(instance_path)

    # 2. Configure your algorithm (e.g., VNS)
    config = VNSConfig(...)

    # 3. Execute the optimization and get the results
    results = run_instance_with_config(run_seconds, problem, config)

    # 4. Return the results in a SavedRun object
    return SavedRun(metadata=..., solutions=...)
```

### Step 3: Register the CLI Runner

Finally, you need to register your new runners with the main CLI application. Create a `register_cli` function in your new module. This function will be called from `cli.py` to add your runners to the command-line interface.

```python
# src/examples/my_problem.py

from typing import Any
from src.cli.cli import CLI

def register_cli(cli: CLI) -> None:
    # Register your solver under a specific problem name
    # The first argument is the problem name, which becomes a subcommand of `run`
    # The second argument is a list of tuples: (runner_name, runner_function)
    cli.register_runner(
        "my_problem",
        [
            ("my_custom_solver_config_A", my_custom_solver),
            ("another_config_B", another_solver),
        ],
    )
```

### Step 4: Add to `cli.py`

The last step is to import your new module and call its registration function in the `if __name__ == "__main__"` block of `cli.py`.

```python
# cli.py

import logging

import src.examples.mokp.ngsa2
import src.examples.mokp.spea2
import src.examples.mokp.vns
import src.examples.my_problem # Import your new module

from src.cli.cli import CLI

# ... setup_logging function

if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    cli = CLI()

    src.examples.mokp.ngsa2.register_cli(cli)
    src.examples.mokp.spea2.register_cli(cli)
    src.examples.mokp.vns.register_cli(cli)
    src.examples.my_problem.register_cli(cli) # Register your new problem

    cli.run()
```

After completing these steps, you will be able to run your new problem's algorithms directly from the command line using a command like `uv run python ./cli.py run -i <instance> -t <time> my_problem`.

