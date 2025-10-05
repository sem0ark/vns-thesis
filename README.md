# Set of components for VNS and MO-VNS Optimization

This set of utilities provides a a set of components for solving single- and multi-objective optimization problems using a variety of Variable Neighborhood Search (VNS) algorithms. The design allows you to mix and match different components to create and test various heuristic approaches, including those tailored for multi-objective problems like the Multi-objective Knapsack Problem (MOKP).

## Components

The toolkit is built on interchangeable modules, enabling the rapid development of new VNS variations. The core components are:

The `VNSOptimizer` class acts as the main orchestrator, tying all the components together to execute the full VNS algorithm. Its main optimization loop follows a classic VNS structure:

1.  **Shaking**: In each iteration `k`, a shake function (e.g., `shake_swap`) is applied to a solution from the current solution archive. The `shake_function`'s role is to generate a new, sufficiently different starting point for the local search.

2.  **Local Search**: The shaken solution is passed to a search function. This can be a single local search operator (`best_improvement`) or a composite VND function that systematically explores multiple neighborhoods.

3.  **Acceptance**: The local optimum found is then evaluated by the acceptance criterion. The solution is either accepted (and added to the archive/buffer) or rejected. If accepted, the process restarts from a new current solution. If rejected, the algorithm proceeds to a larger neighborhood by increasing `k`.


### Local Search Operators

These functions define the neighborhood search phase, which aims to find a strictly better (dominating) solution in the vicinity of the current one. The search functions are wrappers around neighborhood operators, iterating through the neighborhood to find an improving solution. All comparison functions assume **minimization** for all objectives.

#### Single-Objective Local Searches

- **`noop`**: A null operator for Randomized VNS (RVNS) variants where the local search step is skipped. It simply returns the `initial` solution without any changes.
- **`best_improvement(operator)`**: Executes a greedy search. It explores the entire neighborhood defined by a `NeighborhoodOperator` and returns the single **best strictly better** solution found. This guarantees the best local improvement but is computationally expensive.
- **`first_improvement(operator)`**: Implements a repeated descent. It explores the neighborhood and moves to the **first strictly better** near neighbor found. This process repeats until a neighborhood scan yields no improvement. This is typically faster than `best_improvement`.
- **`first_improvement_quick(operator)`**: Executes a **single scan** of the neighborhood and returns the first strictly improving neighbor found. If no improvement is found in this single scan, the original solution is returned. This is designed for a very fast local search step.
- **`composite(search_functions)`**: Implements the **Variable Neighborhood Descent (VND)** strategy. It applies a sequence of local search functions (derived from different neighborhood operators). If a local search finds a **strictly better** solution, the entire sequence restarts from the first function ($\text{VND level } k=0$).

### Acceptance Criteria

The toolkit includes a set of acceptance criteria designed to manage the solution archive, particularly for multi-objective problems where a Pareto front of non-dominated solutions must be maintained. They determine whether a new candidate solution should be accepted into the archive and influence the overall search behavior. All provided implementations assume **minimization** for all objectives.

- **`AcceptBatch`**: This criterion manages the archive (Pareto front) in a **batch-processing style**. The algorithm must **iterate through every solution** in the current front before processing the next one. Accepted solutions are placed into an `upcoming_front`. Once the iteration is complete, the current non-dominated solutions are merged with the accepted ones to form the new front.

- **`AcceptBeam`**: This criterion introduces a **dynamic, beam search-like behavior**. Instead of strict iteration, the algorithm selects a **random solution** from the archive to continue the search. When a new non-dominated solution is found, the archive is **updated immediately** by removing dominated solutions. This provides a more dynamic, less structured search, prioritizing rapid improvement of the current set of solutions.

- **`AcceptBatchSkewed`**: Implements a **Skewed Acceptance** criterion within the batch-processing framework.
    - It maintains a main archive (`front`) and a separate **skewed front** (`skewed_front`).
    - If a new candidate is rejected by the main front, it is then compared against the front using a **skewed objective vector** (the candidate's objectives are skewed by a distance-dependent term). If the skewed candidate is found to be non-dominated, the candidate is added to the `upcoming_skewed_front`.
    - Solutions for the next iteration are selected first from the main front, then from the skewed front. Both fronts are fully iterated and rebuilt in a batch process when exhausted.

- **`AcceptBeamSkewed`**: These criteria combine the concepts of **beam search and skewed acceptance**.
    - It maintains the main non-dominated archive (`front`) and a **buffer** of skewed-accepted solutions (`buffer`).
    - If a new candidate is rejected by the main front, it is compared against the front using a **skewed objective vector** derived from solutions in the *main front*. If this skewed comparison is favorable (not strictly better than the skewed solution's original objective), the candidate is added to the `buffer`.
    - The next solution for the search is chosen **randomly** from the union of the main front and the skewed buffer, encouraging exploration of promising but non-Pareto regions of the solution space.


## Getting Started

### Installation

This project uses `uv` for dependency management. To get started, navigate to the project directory and install the required packages.

```bash
uv sync
```
The updated README description for the CLI functionality, reflecting the new features in `show metrics` (Coverage Matrix and Export options), is below.

### CLI Functionality

The main command-line interface (CLI) is accessed via `python ./cli.py` and is divided into two primary sub-commands: **`show`** and **`run`**.

#### `show`

The `show` command is used to analyze results from previously executed optimization runs. It requires an **instance file** (`-i`), and a **maximum execution time** (`-t`) to filter the results.

- `show plot`: Generates an interactive plot showing the Pareto fronts of the filtered runs, including a reference front for comparison.
    ```bash
    uv run python ./cli.py show mokp -i ./data/mokp/2KP50-11.json -t 10s plot
    ```

- `show metrics`: Displays quantitative performance metrics for the filtered runs. This command now generates **two tables**:
    1.  **Unary Metrics** (e.g., Epsilon, Hypervolume, IGD, R-Metric).
    2.  A **1-to-1 Coverage Matrix** $C(A, B)$ comparing the dominance between all run pairs. Runs that are **completely dominated** by another run are **hidden** from the matrix.

    ```bash
    uv run python ./cli.py show mokp -i ./data/mokp/2KP50-11.json -t 10s mokp metrics
    ```

    The `show metrics` command supports exporting the results:
    - `-o, --output-file <PATH>`: Exports both the Unary Metrics and Coverage Matrix tables to the specified path. It supports **`.csv`** and **`.xlsx`** formats, generating two separate files (e.g., `run_summary_unary.xlsx` and `run_summary_coverage.xlsx`).

    ```bash
    # Example: Exporting metrics to an Excel file
    uv run python ./cli.py show mokp -i ... -t 10s metrics -o ./results/run_summary.xlsx
    ```

Both `show` sub-commands can be further refined using the **`-f`** or **`--filter-configs`** option. This option accepts a boolean expression of "tags", in case any run configuration has multiple space-separated parts, e.g. "vns and k2 or nsga2", filters will match runs with names such as "vns test k2", "k2 vns", "nsga pop_200", etc.

#### `run`

The `run` command executes one or more optimization algorithms on a specified problem instance. It requires an instance file (`-i`) and a maximum execution time (`-t`).

  - `run <problem_name>`: Runs all registered configurations for the specified problem **sequentially**.
    ```bash
    uv run python ./cli.py run -i ./data/mokp/2KP50-11.json -t 10s mokp
    ```
  - Filtering runs: Similar to the `show` command, you can use the `-f` or `--filter-configs` option to run only a subset of algorithms.
    ```bash
    uv run python ./cli.py run -i ./data/mokp/2KP50-11.json -t 10s mokp -f "nsga or spea or k3,batch,noop"
    ```
    This example would execute three sets of runs: all configurations containing "nsga", all containing "spea", and all containing both "k3" and "batch" and "noop".

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

import src.examples.mokp.nsga2
import src.examples.mokp.spea2
import src.examples.mokp.vns
import src.examples.my_problem # Import your new module

from src.cli.cli import CLI

# ... setup_logging function

if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    cli = CLI()

    src.examples.mokp.nsga2.register_cli(cli)
    src.examples.mokp.spea2.register_cli(cli)
    src.examples.mokp.vns.register_cli(cli)
    src.examples.my_problem.register_cli(cli) # Register your new problem

    cli.run()
```

After completing these steps, you will be able to run your new problem's algorithms directly from the command line using a command like `uv run python ./cli.py run -i <instance> -t <time> my_problem`.

