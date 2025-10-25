from copy import copy
from functools import cached_property
from typing import Any, Iterable, Self, TypeVar

T = TypeVar("T")


class Delta[T]:
    """Represents a single step change in the solution."""

    def apply(self, data: T):
        """Change the data inplace."""
        raise NotImplementedError()

    def revert(self, data: T):
        """Revert the data inplace."""
        raise NotImplementedError()


class Solution[T]:
    """Abstract base class for a solution to a given problem."""

    def __init__(
        self,
        data: T,
        problem: "Problem[T]",
        objectives: tuple[float, ...] | None = None,
        delta: Delta[T] | None = None,
    ) -> None:
        self.data = data
        """Problem-specific representation (list, array, etc.)."""
        self.problem = problem
        """Link to the problem instance."""
        self._delta = delta

        self.objectives = objectives or self.problem.calculate_objectives(self.data)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self._hash == other._hash
        return False

    @cached_property
    def _hash(self) -> int:
        """Returns a hash of a solution to use for comparison.
        Default implementation gives a hash of solution's objectives."""
        return self.get_hash()

    def get_hash(self) -> int:
        """Returns a hash of a solution to use for comparison.
        Default implementation gives a hash of solution's objectives."""
        return hash(self.objectives)

    def get_data_copy(self) -> T:
        return copy(self.data)

    def flatten_solution(self) -> Self:
        """Ensures that any deltas for solution were applied."""
        if self._delta:
            new_data = self.get_data_copy()
            self._delta.apply(new_data)
            return self.__class__(new_data, self.problem, self.objectives)

        return self.__class__(self.data, self.problem, self.objectives)

    def to_json_serializable(self) -> Any:
        return self.data

    @staticmethod
    def from_json_serializable(
        problem: "Problem[T]", serialized_data: Any
    ) -> "Solution[T]":
        raise NotImplementedError()


class Problem[T]:
    """
    Abstract interface for defining an optimization problem.
    Concrete problems (e.g., TSP, knapsack) will implement this.
    """

    def __init__(
        self,
        num_variables: int,
        num_objectives: int,
        num_constraints: int,
        problem_name: str = "",
        objective_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.problem_name = problem_name

        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints

        self.objective_names = objective_names or [
            f"Z{i}" for i in range(1, num_objectives + 1)
        ]
        """Actual names of the objectives used when displaying optimization results."""

    def get_initial_solutions(self, num_solutions: int = 50) -> Iterable["Solution[T]"]:
        """Get a random solution to start with."""
        raise NotImplementedError()

    def calculate_objectives(self, solution_data: T) -> tuple[float, ...]:
        """Objective function for this problem, expected minimization."""
        raise NotImplementedError()

    def satisfies_constraints(self, solution_data: T) -> bool:
        """Checks whether a solution is legit for a given problem."""
        raise NotImplementedError()

    def load_solution(self, saved_solution_data: Any) -> "Solution[T]":
        """Loads problem solution instance from a given serialized data."""
        raise NotImplementedError()

    @staticmethod
    def load(filename: str) -> "Problem[T]":
        """Loads problem instance from a given file path."""
        raise NotImplementedError()


class AcceptanceCriterion[T]:
    """
    Abstract interface for deciding whether to accept a new solution.
    Used to compare and store currently the best solutions found.
    """

    def __init__(self):
        """The archive is now managed internally by the acceptance criterion."""

    def accept(self, candidate: Solution[T]) -> bool:
        """
        Decides whether to accept candidate_solution and updates the internal archive.
        Returns True if the candidate leads to a new "current best" or improves the archive.
        """
        raise NotImplementedError()

    def get_all_solutions(self) -> list[Solution[T]]:
        """Returns the full archive of accepted solutions."""
        raise NotImplementedError()

    def get_one_current_solution(self) -> Solution[T]:
        """Returns a single solution from the archive."""
        raise NotImplementedError()

    def clear(self):
        """Clears the state of the criterion."""
        raise NotImplementedError()


class OptimizerAbstract[T]:
    """Abstract optimizer capable of external control over optimization process."""

    def __init__(
        self,
        name: str,
        version: int,
        problem: Problem[T],
        acceptance_criterion: AcceptanceCriterion[T],
    ) -> None:
        """Init.

        Args:
            name (str): Name of the optimizer configuration, useful for logging, debugging, etc.
            version (int): Version of optimizer configuration, useful for serialization and comparison.
            problem (Problem[T]): Problem instance used for optimization.
        """
        self.name = name
        self.version = version
        self.problem = problem
        self.acceptance_criterion = acceptance_criterion

    def initialize(self) -> None:
        """
        Initializes optimization with default problem's solutions.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Resets optimization process and clears internal state.
        """
        raise NotImplementedError()

    def optimize(self) -> Iterable[bool | None]:
        """
        Runs the optimization process. Yielding None in this case, allows to improve CLi experience to allow interruption in the middle of actual iteration.
        """
        raise NotImplementedError()

    def get_solutions(self) -> list[Solution]:
        """
        Runs the current best set of solutions.
        Returns the best solution found (for single-obj) or the Pareto front (for multi-obj).
        """
        raise NotImplementedError()
