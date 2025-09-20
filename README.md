# Set of components for VNS and MO-VNS Optimization

This library provides modular framework for solving (MO) optimization problems using a variety of Variable Neighborhood Search (VNS) and Multi-objective Variable Neighborhood Search (MO-VNS) algorithms. The design allows you to mix and match different components to create and test various heuristic approaches, including those tailored for multi-objective problems like the Multi-objective Knapsack Problem (MOKP).

## Components

The toolkit is built on interchangeable modules, enabling the rapid development of new VNS variations. The core components are:

The `VNSOptimizer` class acts as the main orchestrator, tying all the components together to execute the full VNS algorithm. Its main optimization loop follows a classic VNS structure:
1.  Shaking: In each iteration `k`, a shake function (e.g., `shake_swap`) is applied to the current solution.
2.  Local Search: The shaken solution is passed to a search function. This can be a single local search operator (`best_improvement`) or a composite VND function that systematically explores multiple neighborhoods.
3.  Acceptance: The local optimum found is then evaluated by the acceptance criterion. The solution is either accepted (and added to the archive/buffer) or rejected. If accepted, the process restarts from a new current solution. If rejected, the algorithm proceeds to a larger neighborhood by increasing `k`.

### Local Search Operators

These functions define the neighborhood structure for the local search phase, which aims to find a better solution in the vicinity of the current one.

- `noop`: A null operator for RVNS variants.
- `best_improvement`: Explores the entire neighborhood defined by an operator and returns the single best solution found.
- `first_improvement`: Explores the neighborhood and returns the very first solution that improves upon the current one, which can be faster than `best_improvement`.
- `first_improvement_quick`: `first_improvement` that returns faster by simply returning the first improving neighbor it finds without an inner loop.
- `composite`: This function allows a Variable Neighborhood Descent (VND) strategy, which combines multiple local search functions and applies them sequentially. If a local search finds an improvement, the process restarts from the first search function in the list, basically functioning like VND.


### Acceptance Criteria

The toolkit includes a set of acceptance criteria designed to manage the solution archive, particularly for multi-objective problems where a Pareto front of non-dominated solutions must be maintained.

`AcceptBigger` & `AcceptSmaller`: Allow maintaining archive of non-dominated solutions. They also receive `buffer_size=` that allows more Beam-search like functionality by maintaining a limited queue of previously dominated solutions.

`AcceptSkewedBigger` & `AcceptSkewedSmaller`: Implement a skewed acceptance for Skewed VNS (SVNS). On top of checking for strict Pareto dominance, in case of skewed-dominating each existing solution, it is added to the buffer of size `buffer_size`.


## Getting Started

### Installation

This project uses `uv` for dependency management. To get started, navigate to the project directory and install the required packages.

```bash
uv sync
```
