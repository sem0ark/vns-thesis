# Set of components for VNS and MO-VNS Optimization

This set of utilities provides components for solving single- and multi-objective optimization problems using a variety of **Variable Neighborhood Search (VNS)** algorithms. The design is based on abstract interfaces and interchangeable modules, allowing for rapid development and testing of new heuristic variations.

## Components Architecture

The toolkit is built on several key abstract interfaces and one core concrete optimizer:

| Component | Abstract Class/Type | Role |
| :--- | :--- | :--- |
| **Problem Definition** | `Problem[T]` | Defines the environment: loads problem instances, handles constraints, generates initial solutions, and provides the $\mathbf{Z}$ objective evaluation function (assumed to be for **minimization**). |
| **Solution Structure** | `Solution[T]` | A wrapper around problem-specific data (`T`) that links to the `Problem` instance. It provides cached access to the objective vector and defines equality and hashing for comparison. |
| **Archive/Acceptance** | `AcceptanceCriterion[T]` | Defines the **acceptance rule** and manages the **solution archive**. It decides whether to accept a new solution (e.g., based on dominance for MOO or strict improvement for SOO) and provides the current best solution(s) for the next iteration. |
| **Search Function** | `SearchFunction` | A `Callable` (generator) representing the local search strategy (e.g., `best_improvement`, `composite/VND`). It takes a `Solution` and yields new solutions (or `None` for time tracking). |
| **Shaker Function** | `ShakeFunction` | A `Callable` that performs the shake operation. It takes the current solution and the current neighborhood level ($k$) to generate a random starting point for the local search. |

## VNS Optimizer: `ElementwiseVNSOptimizer`

The `ElementwiseVNSOptimizer` class acts as the main orchestrator, implementing the classic VNS optimization loop. It uses a list of `SearchFunction`s to represent the increasing set of neighborhoods.

The main optimization loop follows a classic **Variable Neighborhood Search** (VNS) structure:

1.  **Selection & Initialization**: The process begins by selecting a `current_solution` from the internal archive managed by the `AcceptanceCriterion`. The neighborhood level $k$ is reset to $0$.

2.  **Shaking**: A shake function (`self.shake_function`) is applied to the `current_solution` using the neighborhood level $k+1$ (to access the next larger neighborhood). This generates a new starting point (`shaken_solution`).

3.  **Local Search (Intensification)**: The `shaken_solution` is passed to the local search function selected by the current neighborhood level ($k$): `self.search_functions[k]`. This search yields `intensified_solution`s.

4.  **Acceptance & Update**:
  - Each yielded `intensified_solution` is evaluated by the `acceptance_criterion.accept()`.
  - If **any** local search step leads to an accepted solution (which typically means finding a better local optimum or adding a non-dominated solution to the archive), the `k` level is **reset to 0** (`k = 0`), and the loop proceeds to the next iteration with the improved archive.
  - If the entire local search (intensification) at level $k$ yields **no improvement** leading to an acceptance, the neighborhood level is **increased** (`k = k + 1`).

The process continues until all neighborhood levels are exhausted ($k \ge k_{\max}$), yielding control (`None`) for external time monitoring and repeating with another solution.


Based on the provided code, the file contains a self-contained command-line utility for performing **cross-instance statistical comparison** of optimization algorithm performance metrics. It averages metrics over multiple runs (instances) and uses the **Friedman test** followed by the **Nemenyi post-hoc test** to determine significant differences in algorithm rankings.

Here is the new README section:

### Cross-Instance Analysis: `multi_instance_metrics.py`

The `multi_instance_metrics.py` utility is designed to aggregate and statistically analyze performance metrics across **multiple problem instances** (or multiple runs saved in separate JSON files). It uses non-parametric statistical tests (Friedman and Nemenyi) to rank the performance of different configuration filters.

The core idea is to treat each instance file as a **dataset**, and each filter expression as a **treatment** (algorithm configuration).

#### Usage

The CLI is a single command:

```bash
python multi_instance_metrics.py [OPTIONS]
```

#### Key Options

| Option | Shorthand | Type | Description |
| :--- | :--- | :--- | :--- |
| **`--input-pattern`** | `-i` | `str` | **Required.** A glob pattern to find the JSON files containing the metrics data (e.g., `'results/metrics/*.json'`). Each matched file is treated as a separate instance/dataset. |
| **`--filter-expression`** | `-f` | `str` | **Required (multiple).** The boolean filter to select configurations for comparison. **Each unique filter** defines a method/treatment in the statistical test. Use this option multiple times. |
| **`--filter-name`** | `-n` | `str` | **Optional (multiple).** A user-friendly name for each corresponding filter expression. If not provided, the expressions themselves are used as names. |
| **`--output-file`** | `-o` | `Path` | Optional path to save the aggregated metrics, Nemenyi p-values, and algorithm rankings. Supports **`.csv`** or **`.xlsx`**. |
| **`--plot-file`** | `-p` | `Path` | Optional path to save the **Nemenyi sign-plot** (a heatmap visualizing significant differences). |
| **`--alpha`** | N/A | `float` | Significance level ($\alpha$) for the statistical tests (default: $0.05$). |

#### Example Execution

To compare two groups of algorithms—"VNS-k1" and "NSGA-II"—across all metrics files in the `metrics_output` directory:

```bash
python multi_instance_metrics.py \
    -i 'metrics_output/*.json' \
    -f 'vns and k1 and 30s' -n 'VNS k1 30s run time limit' \
    -f 'nsga2' -n 'NSGA2 All' \
    -f 'spea2' -n 'SPEA2 All' \
    -o 'cross_instance_analysis.xlsx' \
    -p 'nemenyi_plots.png'
```

#### Output and Analysis

The tool performs the following steps:

1.  **Averaging**: For each filter, it calculates the **average** metric value across all matched configurations within **each instance file**.
2.  **Friedman Test**: It applies the non-parametric Friedman test to the **ranks** of the filters across all instances. This determines if there is a statistically significant difference among the compared configurations.
3.  **Nemenyi Post-Hoc Test**: If the Friedman test finds a significant difference ($\text{p-value} \le \alpha$), the Nemenyi test is performed to identify **which specific pairs** of configurations perform significantly differently.
4.  **Ranking**: It generates an overall ranking based on the **average rank** achieved by each filter across all instances.

The results are printed to the console and can be exported:

- **Excel Export (`.xlsx`)**: The recommended output format, which generates multiple sheets:
    - `Average_Metrics`: The raw aggregated average metrics per instance.
    - `Nemenyi <Metric Short Name>`: The p-value matrix for the Nemenyi test (for metrics where H0 was rejected).
    - `Ranking <Metric Short Name>`: The final table of Average Rank and Standard Deviation of the ranks.
- **Plot Export (`-p`)**: Saves a sign-plot (heatmap) that visually indicates which pairs of algorithms are significantly different according to the Nemenyi test.


### Local Search Operators

These functions define the **local search phase** of a Variable Neighborhood Search (VNS) algorithm. They are **higher-order functions** that wrap a provided `NeighborhoodOperator` (or a list of `SearchFunction`s for `composite`) and return a `SearchFunction` (a generator) that executes the search strategy.

All search functions assume **minimization** for the objectives. They support two modes:

1.  **Multi-Objective**: (When `objective_index` is `None`) Comparison is based on **strict Pareto dominance**.
2.  **Single-Objective**: (When `objective_index` is specified) Comparison is based on the single objective value.

The search functions use a Python **generator** (`yield None` for intermediate steps) to allow external control (e.g., stopping based on time limits) over potentially long-running local search loops. The final, best solution found is returned last.


#### Individual Search Strategies

| Function | Primary Strategy | Objective Mode | Key Behavior |
| :--- | :--- | :--- | :--- |
| **`noop`** | **Null Operator** | N/A | Used primarily for **Randomized VNS (RVNS)** variants. It bypasses the local search step, simply yielding and returning the `initial` solution. |
| **`best_improvement(operator)`** | **Greedy Descent** | SOO / MOO | Repeatedly executes a descent until a local optimum is reached. In each iteration, it explores the **entire neighborhood** defined by the `operator` and moves to the single **best strictly better/dominating** solution found. Computationally expensive but exhaustive. |
| **`first_improvement(operator)`** | **Repeated Descent** | SOO / MOO | Repeatedly executes a descent until a local optimum is reached. In each iteration, it scans the neighborhood and moves immediately to the **first strictly better/dominating** neighbor encountered. This is typically much faster than `best_improvement`. |
| **`first_improvement_quick(operator)`**| **Single Scan** | SOO / MOO | Executes a **single scan** of the neighborhood. It returns the first strictly improving neighbor found. If no improvement is found in this single scan, the original solution is yielded and returned. Designed for the fastest possible local search step. |


#### Composite Search Strategy

| Function | Primary Strategy | Behavior | Acceptance Condition |
| :--- | :--- | :--- | :--- |
| **`composite(search_functions)`** | **Variable Neighborhood Descent (VND)** | Implements the VND metaheuristic using a sequence of local search functions (derived from different neighborhood operators, $k=1, 2, \dots, k_{\max}$). | If a search at level $k$ finds a solution ($\mathbf{x}'$) that **strictly dominates** the current solution ($\mathbf{x}$), $\mathbf{x}$ is updated to $\mathbf{x}'$, and the VND process **restarts** at the first neighborhood level ($\mathbf{k=0}$). Otherwise, the level is incremented ($\mathbf{k \leftarrow k+1}$). |


### Acceptance Criteria

The toolkit includes a set of acceptance criteria designed to manage the solution archive, particularly for multi-objective problems where a Pareto front of non-dominated solutions must be maintained. They determine whether a new candidate solution should be accepted into the archive and influence the overall search behavior. All provided implementations assume **minimization** for all objectives.

- **`ParetoFront`**: The fundamental archive structure. It maintains a list of **non-dominated solutions** based on the provided comparison function (defaults to standard Pareto dominance).
    - **Acceptance:** A candidate is accepted if it's not dominated by any current solution. If it dominates any existing solutions, they are pruned immediately.
    - **Selection (`get_one_current_solution`):** Selects a **random solution** from the front for neighborhood exploration.

- **`AcceptBatch`**: A criterion that wraps a `ParetoFront` and implements a **batch-processing style** iteration.
    - **Acceptance:** Updates the underlying `true_front` immediately.
    - **Selection (`get_one_current_solution`):** Iterates through a **snapshot** of the current non-dominated solutions. The algorithm must **process every solution** in the current snapshot before taking a new snapshot of the updated front for the next batch.

- **`AcceptBeamWrapped` (and `AcceptBeamSkewed`):** This criterion introduces a **dynamic, beam search-like behavior** by wrapping two `ParetoFront` archives.
    - **Archives:** Maintains a **Standard Front (`true_front`)** using standard Pareto dominance (for final results) and a **Custom Front (`custom_front`)** using a user-provided comparison (e.g., skewed dominance) for acceptance and selection.
    - **Acceptance:** A candidate is accepted if it's non-dominated by **both** the standard and the custom front criteria.
    - **Selection (`get_one_current_solution`):** Selects a **random solution** from the **Custom Front** (`custom_front`) to continue the search. The `AcceptBeamSkewed` class specializes this by using the **Skewed Acceptance** logic (based on the `make_skewed_comparator` function) for the `custom_front`.

- **`AcceptBatchWrapped` (and `AcceptBatchSkewed`):** This criterion implements a **batch-processing style** by wrapping two `ParetoFront` archives and utilizing a snapshot.
    - **Archives:** Maintains a **Standard Front (`true_front`)** and a **Custom Front (`custom_front`)**.
    - **Acceptance:** A candidate is accepted if it's non-dominated by **both** the standard and the custom front criteria.
    - **Selection (`get_one_current_solution`):** Iterates through a **snapshot** of the solutions from the **Custom Front** (`custom_front`). Once the snapshot is exhausted, a new snapshot is taken. The `AcceptBatchSkewed` class specializes this by using the **Skewed Acceptance** logic for the `custom_front`.

### Skewed Acceptance Mechanism

The **Skewed Acceptance** used in `AcceptBeamSkewed` and `AcceptBatchSkewed` is implemented via the `make_skewed_comparator` function.

This mechanism modifies the standard dominance check by calculating a **skewed objective vector** for the candidate solution $x$ before comparison:

$$f'_{i, y}(x) = f_i(x) - \alpha_i \times \text{distance}(x, y)$$

This skews the objective value **favorably** (since we are minimizing and subtracting a positive penalty) for solutions that are **far** from the existing solutions $y$ in the front, encouraging the search to explore sparsely populated regions of the solution space.

- $f_i(x)$: The true objective value.
- $\alpha_i$: The weight parameter (skewing magnitude).
- $\text{distance}(x, y)$: The distance between the candidate $x$ and a solution $y$ in the current front, measured in the solution space.


## Getting Started

### Installation

This project uses `uv` for dependency management. To get started, navigate to the project directory and install the required packages.

```bash
uv sync
```
The updated README description for the CLI functionality, reflecting the new features in `show metrics` (Coverage Matrix and Export options), is below.

### CLI Functionality

The main command-line interface (CLI) is accessed via a single entry point (e.g., `python ./cli.py`) and utilizes several specialized top-level commands to manage the optimization workflow.

All commands that operate on saved run files accept the following crucial global options:

| Option | Shorthand | Description | Usage |
| :--- | :--- | :--- | :--- |
| `--instance` | `-i` | **Required.** Accepts one or more paths or glob patterns (e.g., `'data/instance*.json'`) to identify the problem instance(s) to process. | `metrics -i 'data/*.json'` |
| `--filter-string` | `-f` | **Optional.** A boolean expression to select specific configuration runs based on "tags" - whitespace separated elements of a run name (e.g., `'vns and not 30s'`). | `plot -f 'k1 or k3'` |

### Advanced Filter Expression Logic

The filter expression (`-f`) now supports a full boolean logic, case-insensitive tags, and operator precedence:

| Feature | Operator | Example | Description |
| :--- | :--- | :--- | :--- |
| **High Precedence** | `NOT` | `not vns` | Negates the tag/expression immediately following it. |
| **Medium Precedence** | `AND` | `vns and k1` | Matches runs containing **all** specified tags. |
| **Low Precedence** | `OR` | `nsga2 or spea` | Matches runs containing **at least one** of the specified tags. |
| **Grouping** | `( )` | `not (k1 or k2)` | Parentheses override standard operator precedence. |

**Example Usage of Filtering:**

```bash
# Run all configurations that have 'vns' AND are NOT tagged with '30s'.
$ python ./cli.py run -i ./data/instance.json -t 60s -f "vns AND NOT 30s"

# Plot all runs tagged with 'k1' OR any run tagged with 'nsga2'.
$ python ./cli.py plot -i ./data/instance.json -f "(k1 or k3) and not batch"
```

### Core Commands

| Command | Purpose | Key Options | Example Usage |
| :--- | :--- | :--- | :--- |
| **`run`** | Executes selected optimization configurations on specified instance(s). | `-t, --max-time`: **Required.** Max execution time per run (e.g., `30s`, `1h`). | `run -i 'data/*.json' -t 10m -f 'vns or nsga2'` |
| **`plot`** | Generates plots of saved Pareto fronts. | `--lines/--no-lines`: Controls whether points are connected by lines (default: `lines`). | `plot -i instance1.json -f 'vns' --no-lines` |
| **`metrics`** | Displays quantitative performance metrics. | `--unary`, `--coverage`: Flags to display the respective metric tables. | `metrics -i instance1.json --unary --coverage` |
| **`validate`** | Checks saved solutions for correctness (feasibility and objective function validity). | *(Uses global options)* | `validate -i 'data/*.json'` |
| **`archive`** | Validates, archives (moves), and merges solutions into a combined **reference Pareto front**. | `--move`: Flag to move matched run files to an archive directory. | `archive -i 'data/*.json' --move` |

### Detail: The `metrics` Command

The `metrics` command is used for quantitative analysis of saved optimization runs.

- `--unary`: Displays a table of **independent performance metrics** (e.g., Epsilon, Hypervolume, IGD, R2 Metric) for each configuration.
- `--coverage`: Displays a table of **1-to-1 coverage comparisons** (e.g., $C(A, B)$) between all selected configurations.
- `--export`: Exports instance-specific metrics data into a common JSON format suitable for cross-instance comparison tools.
- `-o, --output-file <PATH>`: Exports the calculated unary and coverage metrics to a file (supports `.csv` or `.xlsx`). The Excel format (`.xlsx`) is recommended as it can contain both tables.



## Writing and Registering a Custom Problem

To add a new optimization problem (e.g., a new variant of Knapsack or TSP) and its corresponding algorithms to the CLI, you need to follow a structured approach based on inheritance and concrete runner classes.

### Step 1: Define the Problem and Solution Classes

First, define the problem's abstract representation. Your concrete classes must inherit from the framework's abstract types.

1.  **Solution**: Implement a subclass of `Solution[T]`. This class must define how its problem-specific data (`T`, e.g., a `numpy` array for MOKP) is **hashed** (`get_hash`) for archive storage and how it is **serialized** (`to_json_serializable`).
2.  **Problem**: Implement a subclass of `Problem[T]`. This class must define the instance loading (`load`), the **objective evaluation** (`calculate_objectives`), and the **constraint check** (`satisfies_constraints`).

**Example (MOKP):**

```python
# MOKPProblem loads instance data, checks constraints, and returns
# objectives (negated for maximization to fit the framework's minimization)
class MOKPProblem(Problem[np.ndarray]):
    # ... implementation of load, satisfies_constraints, and calculate_objectives
    pass

# _MOKPSolution handles hashing the numpy data and serialization
class _MOKPSolution(Solution[np.ndarray]):
    def get_hash(self) -> int:
        return xxhash.xxh64().update(self.data.tobytes()).intdigest()

    def to_json_serializable(self):
        return self.data.tolist()
```

### Step 2: Implement the Algorithm Runners

For each set of algorithms (e.g., VNS variants, Pymoo algorithms), implement a concrete **`InstanceRunner`** subclass. These classes define the actual algorithms to be tested on the problem.

#### A. The VNS Runner (`VNSInstanceRunner`)

This runner uses the VNS framework components to generate a large list of VNS variations.

- It initializes the `self.problem` instance in its constructor.
- The `get_variants` method uses `itertools.product` to combine different **Acceptance Criteria**, **Local Search Functions**, and **Shake Functions** (and $k$ levels) to automatically generate all VNS configurations.
- The resulting function (`self.make_func`) wraps the VNS execution logic (`run_vns_optimizer`).

#### B. The Pymoo Runner (`PymooInstanceRunner`)

This runner integrates external libraries (like Pymoo) by wrapping the core problem definition.

- It initializes the **framework problem** (`MOKPProblem`) and wraps it inside the **Pymoo-compatible problem class** (`MOKPPymoo`).
- The `get_variants` method generates different **NSGA2/SPEA2** configurations by varying the population size.
- The execution function calls Pymoo's `minimize` and post-processes the results to ensure only non-dominated solutions are saved in the `SavedRun` object.


### Step 3: Register the Runners in the CLI

The final step is to register the problem name and the list of runner classes with the `CLI`. This is done in the entry point file (e.g., your `if __name__ == "__main__":` block).

```python
# The final CLI registration call:
if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    CLI(
        problem_name="MOKP",
        base_path=base,
        # Pass a list of ALL InstanceRunner classes for this problem
        runner_classes=[VNSInstanceRunner, PymooInstanceRunner],
        problem_class=MOKPProblem # The class used to load problem instances
    ).run()
```

After completing these steps, the CLI automatically exposes all configurations defined in both `VNSInstanceRunner` and `PymooInstanceRunner` for the `MOKP` problem.
