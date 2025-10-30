from typing import Any

import pytest

from src.vns.acceptance import (
    AcceptBatch,
    AcceptBatchWrapped,
    AcceptBeam,
    ComparisonResult,
    compare_solutions_better,
    is_dominating_min,
)


class MockSolution:
    """A mock class for Solution, only storing objectives."""

    def __init__(self, objectives: tuple[float, ...], data: Any = None):
        self.objectives = objectives
        self.data = data  # Used to simulate a solution's state for distance metric


@pytest.mark.parametrize(
    ["candidate", "existing", "expected"],
    [
        pytest.param(
            (1.0, 2.0, 3.0), (2.0, 3.0, 4.0), ComparisonResult.STRICTLY_BETTER
        ),
        pytest.param((5.0, 10.0), (10.0, 10.0), ComparisonResult.STRICTLY_BETTER),
        pytest.param((10.0, 5.0), (10.0, 10.0), ComparisonResult.STRICTLY_BETTER),
        pytest.param(
            (1.0, 2.0), (1.0 + 1e-5, 2.0 + 1e-5), ComparisonResult.STRICTLY_BETTER
        ),
        pytest.param((10.0, 5.0), (5.0, 10.0), ComparisonResult.NON_DOMINATED),
        pytest.param((10.0, 10.0), (5.0, 5.0), ComparisonResult.WORSE),
        pytest.param((10.0, 10.0), (10.0, 10.0), ComparisonResult.NON_DOMINATED),
    ],
)
def test_is_dominating_min(candidate, existing, expected):
    assert is_dominating_min(candidate, existing) == expected


# ==========================================================================
# Acceptance crirerion tests
# ==========================================================================


class Solution:
    """Mock Solution class with objectives and item data for distance."""

    def __init__(self, objectives: tuple[float, ...], data_id: Any = 1):
        self.objectives = objectives
        self.data_id = data_id

    def __eq__(self, other):
        return self.objectives == other.objectives and self.data_id == other.data_id

    def __repr__(self):
        return f"S{self.objectives}(ID {self.data_id})"

    def __hash__(self):
        return hash((self.objectives, self.data_id))


@pytest.mark.parametrize(
    "initial_front_data, candidate_data, expected_front_data, expected_acceptance",
    [
        # Case 1: Empty front - Candidate is always accepted.
        pytest.param([], (10.0, 10.0), [(10.0, 10.0)], True, id="Empty_Front_Accept"),
        # Case 2: Candidate is STICKTLY DOMINATED (WORSE) - Rejected. Front unchanged.
        pytest.param(
            [(5.0, 5.0)],
            (10.0, 10.0),
            [(5.0, 5.0)],
            False,
            id="Candidate_Worse_Rejected",
        ),
        pytest.param(
            [(5.0, 15.0), (15.0, 5.0)],
            (20.0, 20.0),
            [(5.0, 15.0), (15.0, 5.0)],
            False,
            id="Candidate_Worse_PF_Rejected",
        ),
        # Case 3: Candidate DOMINATES STICKTLY BETTER - Accepted. Dominated solution pruned.
        pytest.param(
            [(10.0, 10.0)], (5.0, 5.0), [(5.0, 5.0)], True, id="Candidate_Better_Prune"
        ),
        pytest.param(
            [(5.0, 15.0), (15.0, 5.0), (20.0, 20.0)],
            (10.0, 10.0),
            [(5.0, 15.0), (15.0, 5.0), (10.0, 10.0)],
            True,
            id="Candidate_Better_Prune_One",
        ),
        # Case 4: Candidate is NON-DOMINATED (added). Front grows.
        pytest.param(
            [(10.0, 10.0)],
            (5.0, 15.0),
            [(10.0, 10.0), (5.0, 15.0)],
            True,
            id="Candidate_NonDominated_Added",
        ),
        pytest.param(
            [(5.0, 15.0), (15.0, 5.0)],
            (10.0, 10.0),
            [(5.0, 15.0), (15.0, 5.0), (10.0, 10.0)],
            True,
            id="Candidate_NonDominated_PF_Added",
        ),
        # Case 5: Candidate is IDENTICAL - Rejected. Front unchanged.
        pytest.param(
            [(10.0, 10.0)],
            (10.0, 10.0),
            [(10.0, 10.0)],
            False,
            id="Candidate_Identical_Rejected",
        ),
        pytest.param(
            [(5.0, 15.0), (15.0, 5.0)],
            (5.0, 15.0),
            [(5.0, 15.0), (15.0, 5.0)],
            False,
            id="Candidate_Identical_PF_Rejected",
        ),
        # Case 6: Mixed Pruning (Candidate dominates one, is non-dominated by another).
        pytest.param(
            [(5.0, 15.0), (20.0, 20.0)],
            (10.0, 10.0),
            [(5.0, 15.0), (10.0, 10.0)],
            True,
            id="Candidate_Mixed_Prune_One",
        ),
        # Case 7: Candidate dominates multiple solutions.
        pytest.param(
            [(10.0, 10.0), (12.0, 12.0), (0.0, 15.0)],
            (7.0, 7.0),
            [(7.0, 7.0), (0.0, 15.0)],
            True,
            id="Candidate_Dominates_Multiple",
        ),
    ],
)
def test_accept_beam_acceptance_logic(
    initial_front_data: list[tuple[float, ...]],
    candidate_data: tuple[float, ...],
    expected_front_data: list[tuple[float, ...]],
    expected_acceptance: bool,
):
    """
    Test the core logic of AcceptBeam.accept for various domination scenarios.
    """

    criterion = AcceptBeam()
    criterion.front.solutions = [Solution(obj) for obj in initial_front_data]
    candidate = Solution(candidate_data)
    actual_acceptance = criterion.accept(candidate)
    actual_front_data = [s.objectives for s in criterion.get_all_solutions()]

    assert actual_acceptance == expected_acceptance

    expected_set = set(expected_front_data)
    actual_set = set(actual_front_data)

    assert actual_set == expected_set, (
        f"\nInitial: {initial_front_data}\nCandidate: {candidate_data}\n"
        f"Expected Front: {expected_front_data}\nActual Front: {actual_front_data}"
    )


def test_accept_updates_true_front_and_initial_snapshot():
    """Test that accept updates the live front and takes an initial snapshot if the batch is empty."""
    criterion = AcceptBatch()
    # same objectives, but different solutions
    candidate_1 = Solution((10.0,), 1)
    candidate_2 = Solution((10.0,), 2)
    # duplicate
    candidate_3_duplicate = Solution((10.0,), 1)

    # 1. Accept unique solution (true_front is empty -> snapshot taken)
    accepted_1 = criterion.accept(candidate_1)
    assert accepted_1 is True
    # Check true front (live archive)
    assert len(criterion.true_front.solutions) == 1
    assert candidate_1 in criterion.true_front.solutions
    # Check snapshot (should have been created immediately after first accept)
    assert criterion.front_snapshot == [candidate_1]

    # 2. Accept another unique solution (front_snapshot is NOT empty -> no new snapshot taken)
    accepted_2 = criterion.accept(candidate_2)
    assert accepted_2 is True
    # Check true front (live archive)
    assert len(criterion.true_front.solutions) == 2
    assert candidate_2 in criterion.true_front.solutions
    # Check snapshot (remains unchanged)
    assert criterion.front_snapshot == [candidate_1]

    # 3. Reject a duplicate solution (Mock ParetoFront rejects duplicates)
    accepted_3 = criterion.accept(candidate_3_duplicate)
    assert accepted_3 is False
    # True front size remains 2
    assert len(criterion.true_front.solutions) == 2


def test_get_one_current_solution_iterates_through_snapshot_wrapped():
    """Test that solutions are popped from the front_snapshot list until empty."""
    criterion = AcceptBatchWrapped(compare_solutions_better)
    s1, s2, s3 = Solution((10.0,), 1), Solution((20.0,), 2), Solution((30.0,), 3)

    # Manually set the snapshot for testing iteration logic
    criterion.front_snapshot = [s1, s2, s3]

    # Pop returns the last element
    sol_3 = criterion.get_one_current_solution()
    assert sol_3.data_id == 3
    assert len(criterion.front_snapshot) == 2

    sol_2 = criterion.get_one_current_solution()
    assert sol_2.data_id == 2
    assert len(criterion.front_snapshot) == 1

    sol_1 = criterion.get_one_current_solution()
    assert sol_1.data_id == 1
    assert not criterion.front_snapshot  # Snapshot is now empty


def test_take_snapshot_makes_correct_snapshot_wrapped():
    """Test that solutions are popped from the front_snapshot list until empty."""
    criterion = AcceptBatchWrapped(compare_solutions_better)
    s1, s2, s3 = Solution((10.0,), 1), Solution((20.0,), 2), Solution((30.0,), 3)

    # Taking multiple snapshots should not change the snapshot without changes in the front
    criterion.true_front.solutions = []
    criterion.custom_front.solutions = [s1, s2, s3]
    criterion._take_snapshot()
    assert criterion.front_snapshot == [s1, s2, s3]

    criterion.true_front.solutions = []
    criterion.custom_front.solutions = [s3, s2, s1]
    criterion._take_snapshot()
    assert criterion.front_snapshot == [s3, s2, s1]

    criterion.true_front.solutions = []
    criterion.custom_front.solutions = [s1, s2, s3]
    criterion._take_snapshot()
    assert criterion.front_snapshot == [s1, s2, s3]


def test_get_one_current_solution_triggers_snapshot_when_empty_wrapped():
    """Test that an empty front_snapshot triggers a call to _take_snapshot from true_front."""
    criterion = AcceptBatchWrapped(compare_solutions_better)

    # 1. Setup true_front (live archive)
    criterion.custom_front.solutions.extend(
        [Solution((100.0,), 100), Solution((200.0,), 200)]
    )

    # 2. Ensure current front_snapshot is empty
    criterion.front_snapshot = []

    # 3. Call get_one_current_solution - this must trigger _take_snapshot()
    sol_pop = criterion.get_one_current_solution()

    # Check that snapshot was taken and one solution was popped
    assert len(criterion.front_snapshot) == 1

    # The returned solution must be one of the original two
    assert sol_pop.data_id in [100, 200]

    # Check that the remaining solution is the other one in the snapshot
    remaining_id = 100 if sol_pop.data_id == 200 else 200
    assert criterion.front_snapshot[0].data_id == remaining_id


def test_get_one_current_solution_raises_value_error_when_empty_wrapped():
    """Test the ValueError when both true_front and front_snapshot are empty."""
    criterion = AcceptBatchWrapped(compare_solutions_better)
    criterion.custom_front.clear()  # Ensure custom_front is empty
    criterion.front_snapshot.clear()  # Ensure snapshot is empty

    with pytest.raises(
        ValueError, match="Archive is empty and all solutions have been processed."
    ):
        criterion.get_one_current_solution()


def test_clear_resets_both_internal_structures_wrapped():
    """Test that clear resets both the front_snapshot and the true_front archive."""
    criterion = AcceptBatchWrapped(compare_solutions_better)

    # Populate both structures
    criterion.true_front.accept(Solution((1.0,), 1))
    criterion.custom_front.accept(Solution((1.0,), 1))
    criterion.front_snapshot.append(Solution((2.0,), 2))

    assert criterion.true_front.solutions
    assert criterion.custom_front.solutions
    assert criterion.front_snapshot

    criterion.clear()

    assert not criterion.true_front.solutions
    assert not criterion.custom_front.solutions
    assert not criterion.front_snapshot
