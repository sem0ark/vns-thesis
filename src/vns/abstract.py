from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Generic, Iterable, Self, TypeVar

T = TypeVar("T")


@dataclass
class Problem(Generic[T]):
    """
    Abstract interface for defining an optimization problem.
    Concrete problems (e.g., TSP, knapsack) will implement this.
    """

    objective_function: Callable[["Solution[T]"], tuple[float, ...]]
    """Objective function for this problem."""

    get_initial_solutions: Callable[[], Iterable["Solution[T]"]]
    """Get a random solution to start with."""


@dataclass
class Solution(Generic[T]):
    """Abstract base class for a solution to a given problem."""

    data: T  # Problem-specific representation (list, array, etc.)
    problem: Problem[T]

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.equals(other)
        return False

    @cached_property
    def objectives(self) -> tuple[float, ...]:
        return self.problem.objective_function(self)

    def new(self, data: Any) -> Self:
        return self.__class__(data, self.problem)

    def equals(self, other: Self) -> bool:
        """Checks whether solutions of the same type also completely the same.
        Default implementation considers solutions as equal if their objective value is the same.
        """
        return all(
            abs(o1 - o2) < 1e-6 for o1, o2 in zip(self.objectives, other.objectives)
        )

    def to_json_serializable(self) -> Any:
        return self.data


class AcceptanceCriterion(Generic[T]):
    """
    Abstract interface for deciding whether to accept a new solution.
    Used to compare and store currently the best solutions found.
    """

    def __init__(self):
        """The archive is now managed internally by the acceptance criterion."""

    def dominates(
        self, new_solution: Solution[T], current_solution: Solution[T]
    ) -> bool:
        """
        Determines if 'new_solution' is better than 'current_solution' based on Pareto dominance.
        """
        raise NotImplementedError

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


NeighborhoodOperator = Callable[[Solution[T], "VNSConfig"], Iterable[Solution[T]]]
ShakeFunction = Callable[[Solution[T], int, "VNSConfig"], Solution[T]]
SearchFunction = Callable[[Solution[T], "VNSConfig"], Solution[T]]


@dataclass
class VNSConfig(Generic[T]):
    """Configuration for an abstract VNS optimizer."""

    problem: Problem[T]

    search_functions: list[SearchFunction[T]]
    """Neighborhood operators for a given problem, ordered by increasing size/complexity."""

    shake_function: ShakeFunction

    acceptance_criterion: AcceptanceCriterion[Solution[T]]
    name: str = "default_name"
    version: int = 1

    def __post_init__(self):
        if not self.search_functions:
            raise ValueError(
                "At least one neighborhood operator must be provided or defined by the problem."
            )
