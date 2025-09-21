from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Iterable, Self, TypeVar

T = TypeVar("T")


class OptimizationDirection(Enum):
    """
    Represents whether a given objective should be minimized or maximized.
    """

    MIN = 1
    MAX = -1


@dataclass
class Problem[T]:
    """
    Abstract interface for defining an optimization problem.
    Concrete problems (e.g., TSP, knapsack) will implement this.
    """

    objective_function: Callable[["Solution[T]"], tuple[float, ...]]
    """Objective function for this problem, expected minimization."""

    get_initial_solutions: Callable[[], Iterable["Solution[T]"]]
    """Get a random solution to start with."""


@dataclass
class Solution[T]:
    """Abstract base class for a solution to a given problem."""

    data: T
    """Problem-specific representation (list, array, etc.)."""
    problem: Problem[T]
    """Link to the problem instance."""

    def __hash__(self) -> int:
        return self.get_hash()

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.equals(other)
        return False

    @cached_property
    def objectives(self) -> tuple[float, ...]:
        return self.problem.objective_function(self)

    def new(self, data: Any) -> Self:
        return self.__class__(data, self.problem)

    def get_hash(self) -> int:
        """Returns a hash of a solution to use for comparison.
        Default implementation gives a hash of solution's objectives."""
        return hash(self.objectives)

    def equals(self, other: Self) -> bool:
        """Checks whether solutions are the same.
        Default implementation considers solutions as equal if their objective value is the same.
        """
        return all(
            abs(o1 - o2) < 1e-6 for o1, o2 in zip(self.objectives, other.objectives)
        )

    def to_json_serializable(self) -> Any:
        return self.data


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
        raise NotImplementedError

    def get_all_solutions(self) -> list[Solution[T]]:
        """Returns the full archive of accepted solutions."""
        raise NotImplementedError

    def get_one_current_solution(self) -> Solution[T]:
        """Returns a single solution from the archive."""
        raise NotImplementedError

    def clear(self):
        """Clears the state of the criterion."""
        raise NotImplementedError


class VNSOptimizerAbstract[T]:
    """Abstract VNS optimizer, also defines the context passed between program elements."""

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

    def optimize(self) -> Iterable[bool]:
        """
        Runs the VNS optimization process.
        """
        raise NotImplementedError

    def get_solutions(self) -> list[Solution]:
        """
        Runs the current best set of solutions.
        Returns the best solution found (for single-obj) or the Pareto front (for multi-obj).
        """
        raise NotImplementedError


NeighborhoodOperator = Callable[
    [Solution[T], VNSOptimizerAbstract[T]], Iterable[Solution[T]]
]
