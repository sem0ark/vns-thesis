from typing import Iterable

import pytest

from src.vns.local_search import (
    best_improvement,
    composite,
    first_improvement,
    first_improvement_quick,
    noop,
)


class Solution:
    """Mock Solution class with objectives and item data for distance."""

    def __init__(self, objectives: tuple[float, ...], data_id: int = 1):
        self.objectives = objectives
        self.data_id = data_id

    def __eq__(self, other):
        return self.objectives == other.objectives and self.data_id == other.data_id

    def __repr__(self):
        return f"S{self.objectives}(ID{self.data_id})"

    def __hash__(self):
        return hash((self.objectives, self.data_id))


@pytest.fixture
def initial_solution_single():
    return Solution((10.0,), 10)


@pytest.fixture
def initial_solution_multi():
    return Solution((10.0, 10.0), 10)


def mock_operator_single_objective(current: Solution) -> Iterable[Solution]:
    """Yields a sequence of neighbors for single-objective tests."""
    # (10) -> (9) -> (11) -> (8) -> (7)
    if current.data_id == 10:
        yield Solution((9.0,), 9)
        yield Solution((11.0,), 11)
        yield Solution((8.0,), 8)
    elif current.data_id == 9:
        yield Solution((8.0,), 8)
        yield Solution((12.0,), 12)
    elif current.data_id == 8:
        yield Solution((7.0,), 7)
    elif current.data_id == 7:
        yield Solution((10.0,), 10)  # Worse neighbor


def mock_operator_multi_objective(current: Solution) -> Iterable[Solution]:
    """Yields a sequence of neighbors for multi-objective tests."""
    # (10, 10) -> (9, 9) [dominates] -> (11, 9) [non-dominated] -> (8, 8) [dominates]
    if current.data_id == 10:
        yield Solution((9.0, 9.0), 9)  # Dominates (10, 10)
        yield Solution((11.0, 9.0), 11)  # Non-dominated with (10, 10)
        yield Solution((8.0, 8.0), 8)  # Dominates (10, 10) and (9, 9)
    elif current.data_id == 9:
        yield Solution((8.0, 8.0), 8)
    elif current.data_id == 8:
        yield Solution((7.0, 7.0), 7)


def mock_operator_mixed(current: Solution) -> Iterable[Solution]:
    """
    Yields neighbors with a mix of worse, non-dominated, and better solutions.

    (10.0, 10.0) neighbors:
    - S11: (11.0, 11.0) -> WORSE (Rejected)
    - S12: (11.0, 9.0) -> NON-DOMINATED (Rejected in single/multi-obj)
    - S13: (9.0, 9.0) -> STRICTLY BETTER (Accepted)
    """
    if current.data_id == 10:
        # 1. Worse solution
        yield Solution((11.0, 11.0), 11)
        # 2. Non-dominated solution
        yield Solution((11.0, 9.0), 12)
        # 3. Strictly better solution (First Improvement)
        yield Solution((9.0, 9.0), 13)
    elif current.data_id == 13:
        # 4. Strictly better solution
        yield Solution((8.0, 8.0), 14)
    elif current.data_id == 14:
        # 5. Already Local Optimum: only yields worse/non-improving neighbor
        yield Solution((10.0, 10.0), 15)
    # Stops for any other ID


def test_noop_returns_initial_solution(initial_solution_single):
    search_func = noop()
    results = list(search_func(initial_solution_single))
    assert len(results) == 1
    assert results[0] == initial_solution_single


@pytest.mark.parametrize(
    "objective_index, initial_sol, operator, expected_id",
    [
        (
            0,
            Solution((10.0,), 10),
            mock_operator_single_objective,
            9,
        ),  # First is better (9 < 10)
        (
            None,
            Solution((10.0, 10.0), 10),
            mock_operator_multi_objective,
            9,
        ),  # First (9, 9) dominates (10, 10)
        (
            0,
            Solution((7.0,), 7),
            mock_operator_single_objective,
            7,
        ),  # No improvement found, yield initial
    ],
)
def test_first_improvement_quick(objective_index, initial_sol, operator, expected_id):
    search_func = first_improvement_quick(operator, objective_index)
    results = list(search_func(initial_sol))
    assert len(results) == 1
    assert results[0].data_id == expected_id


def test_best_improvement_single_objective_full_descent():
    search_func = best_improvement(mock_operator_single_objective, objective_index=0)
    initial = Solution((10.0,), 10)

    # Expected moves: 10 -> (best of 9, 11, 8) -> 8. Then 8 -> 7. Stop.
    # Neighborhood(10) yields 9, 11, 8. Best is 8.
    # Neighborhood(8) yields 7. Best is 7.
    # Neighborhood(7) yields 10. Best is 7 (no improvement). Stop.

    results = list(search_func(initial))

    # Expected results: [None (10->8), None (8->7), S7]
    assert len(results) == 3
    assert results[0] is None
    assert results[1] is None
    assert results[2].data_id == 7  # Final local optimum


def test_best_improvement_multi_objective_full_descent():
    search_func = best_improvement(mock_operator_multi_objective)
    initial = Solution((10.0, 10.0), 10)

    # Expected moves: 10 -> (best of 9, 11, 8) -> 8. Then 8 -> 7. Stop.
    # N(10) yields S9, S11, S8. S8 dominates S9, S11 is non-dominated by S9. S8 is the 'best' dominator.
    # The comparison only checks if neighbor strictly dominates best_found_in_neighborhood.
    # S9 dominates S10. best = S9.
    # S11 is NON-DOMINATED by S9. best remains S9.
    # S8 dominates S9. best = S8.
    # Final move is 10 -> 8.
    # N(8) yields S7. S7 dominates S8. Final move is 8 -> 7.

    results = list(search_func(initial))

    # Expected results: [None (10->8), None (8->7), S7]
    assert len(results) == 3
    assert results[2].data_id == 7  # Final local optimum


@pytest.mark.parametrize(
    "objective_index, initial_sol, operator, expected_flow_ids",
    [
        # Case 1: Simple Single-Objective Descent (from previous tests)
        (0, Solution((10.0,), 10), mock_operator_single_objective, [10, 9, 8, 7]),
        # Case 2: Multi-Objective Descent (from previous tests)
        (
            None,
            Solution((10.0, 10.0), 10),
            mock_operator_multi_objective,
            [10, 9, 8, 7],
        ),
        # Case 3 (New): Single-Objective, Accepts the third neighbor (9.0 < 10.0)
        (0, Solution((10.0, 10.0), 10), mock_operator_mixed, [10, 13, 14, 14]),
        # Case 4 (New): Multi-Objective, Accepts the third neighbor ((9,9) dominates (10,10))
        (None, Solution((10.0, 10.0), 10), mock_operator_mixed, [10, 13, 14, 14]),
        # Case 5 (New): Local Optimum from the start
        (0, Solution((9.0, 9.0), 14), mock_operator_mixed, [14, 14]),
    ],
)
def test_first_improvement_iterative_descent(
    objective_index, initial_sol, operator, expected_flow_ids
):
    """Test the full descent path of the first_improvement strategy."""
    for result in first_improvement(operator, objective_index)(initial_sol):
        if result is not None:
            final_solution = result
            break

    assert final_solution.data_id == expected_flow_ids[-1], (
        "Final solution ID mismatch."
    )


@pytest.mark.parametrize(
    "objective_index, initial_sol, operator, expected_flow_ids",
    [
        # Case B1: Single-Objective Descent. 10 -> 8 (best) -> 7 (best). Stop.
        (0, Solution((10.0,), 10), mock_operator_single_objective, [10, 8, 7, 7]),
        # Case B2: Multi-Objective Descent. 10 -> 8 (best dominator) -> 7. Stop.
        (
            None,
            Solution((10.0, 10.0), 10),
            mock_operator_multi_objective,
            [10, 8, 7, 7],
        ),
        # Case B3: Single-Objective Mixed. 10 -> 13 (obj=9, best improvement) -> 14 (obj=8). Stop.
        (0, Solution((10.0, 10.0), 10), mock_operator_mixed, [10, 13, 14, 14]),
        # Case B4: Multi-Objective Mixed. 10 -> 13 (best dominator) -> 14. Stop.
        (None, Solution((10.0, 10.0), 10), mock_operator_mixed, [10, 13, 14, 14]),
        # Case B5: Local Optimum from the start. (S14 only yields S15 (worse)).
        (0, Solution((8.0, 8.0), 14), mock_operator_mixed, [14, 14]),
    ],
)
def test_best_improvement_iterative_descent(
    objective_index, initial_sol, operator, expected_flow_ids
):
    """Test the full descent path of the best_improvement strategy."""

    final_solution = None
    for result in best_improvement(operator, objective_index)(initial_sol):
        if result is not None:
            final_solution = result
            break

    assert final_solution.data_id == expected_flow_ids[-1], (
        "Final solution ID mismatch."
    )


@pytest.mark.parametrize(
    "objective_index, initial_sol, operator, expected_flow_ids",
    [
        # Case Q1: Single-Objective, First neighbor is an improvement (10 -> 9).
        (0, Solution((10.0,), 10), mock_operator_single_objective, [10, 9]),
        # Case Q2: Multi-Objective, First neighbor is an improvement (10 -> 9).
        (None, Solution((10.0, 10.0), 10), mock_operator_multi_objective, [10, 9]),
        # Case Q3: Mixed neighborhood, Third neighbor is the first improvement (10 -> 13).
        # S11 (11) is worse, S12 (11,9) is non-dominated, S13 (9,9) is improvement.
        (None, Solution((10.0, 10.0), 10), mock_operator_mixed, [10, 13]),
        # Case Q4: Local Optimum (Single-Objective 7 only has 10 as neighbor - worse).
        (0, Solution((7.0,), 7), mock_operator_single_objective, [7, 7]),
        # Case Q5: Local Optimum (S14 only has S15 (worse) as neighbor).
        (0, Solution((8.0, 8.0), 14), mock_operator_mixed, [14, 14]),
    ],
)
def test_first_improvement_quick_one_shot(
    objective_index, initial_sol, operator, expected_flow_ids
):
    """Test the single-shot behavior of first_improvement_quick."""

    # 1. Execute the search function
    final_solution = None
    for result in first_improvement_quick(operator, objective_index)(initial_sol):
        if result is not None:
            final_solution = result
            break

    assert final_solution.data_id == expected_flow_ids[-1], (
        "Final solution ID mismatch."
    )


@pytest.mark.parametrize(
    "initial_sol, operator, expected_flow_ids",
    [
        # Case S1: Simple Single-Objective Descent (10 -> 9 -> 8 -> 7)
        (Solution((10.0,), 10), mock_operator_single_objective, [10, 9, 8, 7]),
        # Case S2: Mixed neighborhood, First Improvement: 10 -> 13 (9.0 < 10.0) -> 14 (8.0 < 9.0). Stop.
        # Note: Solutions must have multiple objectives for mock_operator_mixed, but only index 0 is used.
        (Solution((10.0, 10.0), 10), mock_operator_mixed, [10, 13, 14, 14]),
        # Case S3: Local Optimum from the start (S14 only yields S15 (10.0), which is worse than 8.0).
        (Solution((8.0, 8.0), 14), mock_operator_mixed, [14, 14]),
    ],
)
def test_first_improvement_single_objective_descent(
    initial_sol, operator, expected_flow_ids
):
    """Test the full descent path of first_improvement using only objective index 0."""
    # Explicitly set objective_index=0
    search_func = first_improvement(operator, objective_index=0)

    final_solution = None
    for result in search_func(initial_sol):
        if result is not None:
            final_solution = result
            break

    assert final_solution.data_id == expected_flow_ids[-1], (
        "Final solution ID mismatch."
    )


# Search functions (defined outside the test function for clarity)
VND_LEVELS = [
    first_improvement(mock_operator_mixed, objective_index=None),
    best_improvement(mock_operator_multi_objective, objective_index=None),
    first_improvement(mock_operator_mixed, objective_index=None),
]


@pytest.mark.parametrize(
    "initial_sol, search_functions, expected_final_id",
    [
        (Solution((10.0, 10.0), 10), VND_LEVELS, 14),
        (Solution((10.0, 10.0), 10), [VND_LEVELS[0]], 14),
        (Solution((11.0, 11.0), 11), [VND_LEVELS[0]], 11),
        (Solution((8.0, 8.0), 14), VND_LEVELS, 14),
        (Solution((11.0, 11.0), 11), VND_LEVELS, 11),
    ],
)
def test_composite_vnd_flow(initial_sol, search_functions, expected_final_id):
    """
    Tests the full VND flow, ensuring correct level resetting and incrementing
    based on the strict dominance comparison, and asserting the final solution.
    """
    vnd_search_func = composite(search_functions)

    # Run the generator to get the final solution
    results = list(vnd_search_func(initial_sol))
    final_solution = results[-1]

    # Assert the Final Solution ID
    assert final_solution.data_id == expected_final_id, (
        f"Expected final ID {expected_final_id}, got {final_solution.data_id}"
    )
