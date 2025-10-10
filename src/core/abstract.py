from enum import Enum
from functools import cached_property
from typing import Any, Iterable, Self, TypeVar

T = TypeVar("T")


class OptimizationDirection(Enum):
    """
    Represents whether a given objective should be minimized or maximized.
    """

    MIN = 1
    MAX = -1


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

    def load_solution(self, saved_solution_data: Any) -> "Solution[T]":
        """Loads problem solution instance from a given serialized data."""
        raise NotImplementedError()

    @staticmethod
    def load(filename: str) -> "Problem[T]":
        """Loads problem instance from a given file path."""
        raise NotImplementedError()


class Solution[T]:
    """Abstract base class for a solution to a given problem."""

    def __init__(
        self,
        data: T,
        problem: Problem[T],
        precomputed_objectives: tuple[float, ...] | None = None,
    ) -> None:
        self.data = data
        """Problem-specific representation (list, array, etc.)."""
        self.problem = problem
        """Link to the problem instance."""
        self._objectives = precomputed_objectives

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self._hash == other._hash
        return False

    @cached_property
    def objectives(self) -> tuple[float, ...]:
        if self._objectives is not None:
            return self._objectives
        return self.calculate_objectives()

    @cached_property
    def _hash(self) -> int:
        """Returns a hash of a solution to use for comparison.
        Default implementation gives a hash of solution's objectives."""
        return self.get_hash()

    def calculate_objectives(self) -> tuple[float, ...]:
        """Objective function for this problem, expected minimization."""
        raise NotImplementedError()

    def satisfies_constraints(self) -> bool:
        """Checks whether a solution is legit for a given problem."""
        raise NotImplementedError()

    def get_hash(self) -> int:
        """Returns a hash of a solution to use for comparison.
        Default implementation gives a hash of solution's objectives."""
        return hash(self.objectives)

    def get_data_copy(self) -> T:
        raise NotImplementedError()

    def get_solution_copy(self) -> Self:
        """Create a copy of a solution with new data."""
        return self.__class__(self.get_data_copy(), self.problem)

    def to_json_serializable(self) -> Any:
        return self.data

    @staticmethod
    def from_json_serializable(
        problem: Problem[T], serialized_data: Any
    ) -> "Solution[T]":
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

    def reset(self) -> None:
        """
        Initializes VNS optimization process and clear internal state.
        """
        raise NotImplementedError()

    def optimize(self) -> Iterable[bool | None]:
        """
        Runs the VNS optimization process. Yielding None in this case, allows to improve CLi experience to allow interruption in the middle of actual iteration.
        """
        raise NotImplementedError()

    def get_solutions(self) -> list[Solution]:
        """
        Runs the current best set of solutions.
        Returns the best solution found (for single-obj) or the Pareto front (for multi-obj).
        """
        raise NotImplementedError()
