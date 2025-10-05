from typing import Any, Callable
import pytest

from src.vns.acceptance import (
    AcceptBatch,
    AcceptBatchWrapped,
    ComparisonResult,
    ParetoFront,
    is_dominating_min,
    make_skewed_comparator,
)


class MockSolution:
    """A mock class for Solution, only storing objectives."""

    def __init__(self, objectives: tuple[float, ...], data: Any = None):
        self.objectives = objectives
        self.data = data  # Used to simulate a solution's state for distance metric


def mock_distance_fixed(sol1: MockSolution, sol2: MockSolution) -> float:
    return 10.0


def mock_distance_zero(sol1: MockSolution, sol2: MockSolution) -> float:
    return 0.0


def mock_distance_from_data(sol1: MockSolution, sol2: MockSolution) -> float:
    if sol1.data is not None and sol2.data is not None:
        return sol1.data[0]
    return 5.0


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


# Scenarios for testing skewed acceptance logic
@pytest.mark.parametrize(
    "alpha, new_obj, existing_obj, distance_metric, expected_result, debug_info",
    [
        # Scenario 1: No skewing (distance=0) -> acts like standard dominance
        pytest.param(
            [0.1, 0.1],
            (10.0, 10.0),
            (10.0, 10.0),
            mock_distance_zero,
            ComparisonResult.NON_DOMINATED,
            "Standard: Non-dominated (no skew)",
        ),
        # Scenario 2: Perfect dominance (no skew)
        pytest.param(
            [0.1, 0.1],
            (9.0, 9.0),
            (10.0, 10.0),
            mock_distance_zero,
            ComparisonResult.STRICTLY_BETTER,
            "Standard: Strictly Better (no skew)",
        ),
        # Scenario 3: Skewing enables acceptance (Skewed Objectives: [10 - 0.1*10, 10 - 0.1*10] = [9.0, 9.0])
        pytest.param(
            [0.1, 0.1],
            (10.0, 10.0),
            (9.5, 9.5),
            mock_distance_fixed,
            ComparisonResult.STRICTLY_BETTER,
            "Skewed: Worse solution accepted",
        ),
        # Scenario 4: Skewing NOT enough to enable acceptance (Skewed Objectives: [10 - 0.01*10, 10 - 0.01*10] = [9.9, 9.9])
        pytest.param(
            [0.01, 0.01],
            (10.0, 10.0),
            (9.5, 9.5),
            mock_distance_fixed,
            ComparisonResult.WORSE,
            "Skewed: Worse solution rejected",
        ),
        # Scenario 5: Skewing makes a non-dominated solution strictly better
        # True: (10, 20) vs (20, 10) -> NON_DOMINATED
        # Skewed: [10 - 0.1*10, 20 - 0.1*10] = [9.0, 19.0]
        # Compare: [9.0, 19.0] vs [20.0, 10.0] -> WORSE (since 19.0 > 10.0 and 9.0 < 20.0)
        pytest.param(
            [0.1, 0.1],
            (10.0, 20.0),
            (20.0, 10.0),
            mock_distance_fixed,
            ComparisonResult.NON_DOMINATED,
            "Skewed: Non-dominated remains non-dominated/worse",
        ),
        # Scenario 6: Skewing confirms a worse solution is worse
        # True: (15, 15) vs (10, 10) -> WORSE
        # Skewed: [15 - 0.1*10, 15 - 0.1*10] = [14.0, 14.0]
        # Compare: [14.0, 14.0] vs [10.0, 10.0] -> WORSE
        pytest.param(
            [0.1, 0.1],
            (15.0, 15.0),
            (10.0, 10.0),
            mock_distance_fixed,
            ComparisonResult.WORSE,
            "Skewed: Worse remains worse",
        ),
        # Scenario 7: Testing with data-driven distance (Distance = 2.0, alpha = [1, 0])
        # True: (10, 10) vs (9, 10) -> WORSE
        # Skewed: [10 - 1*2, 10 - 0*2] = [8.0, 10.0]
        # Compare: [8.0, 10.0] vs [9.0, 10.0] -> STRICTLY_BETTER (8.0 < 9.0, 10.0 == 10.0)
        pytest.param(
            [1.0, 0.0],
            (10.0, 10.0),
            (9.0, 10.0),
            mock_distance_from_data,
            ComparisonResult.STRICTLY_BETTER,
            "Data-driven Skewed: Accepted (only obj1 skewed)",
            id="data-driven-skew-acceptance",
            marks=pytest.mark.xfail(
                reason="Need to pass distance via solution object, mocking needed"
            ),
        ),
    ],
)
def test_make_skewed_comparator_logic(
    alpha, new_obj, existing_obj, distance_metric, expected_result, debug_info
):
    comparator = make_skewed_comparator(alpha, distance_metric)

    distance_value = 0.0
    if distance_metric == mock_distance_fixed:
        distance_value = 10.0
    elif distance_metric == mock_distance_from_data:
        distance_value = 2.0

    candidate = MockSolution(new_obj, data=(distance_value,))
    existing = MockSolution(existing_obj, data=(0.0,))

    result = comparator(candidate, existing)
    assert result == expected_result


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

    criterion = ParetoFront()
    criterion.front = [Solution(obj) for obj in initial_front_data]
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
    assert len(criterion.true_front.front) == 1
    assert candidate_1 in criterion.true_front.front
    # Check snapshot (should have been created immediately after first accept)
    assert criterion.front_snapshot == [candidate_1]

    # 2. Accept another unique solution (front_snapshot is NOT empty -> no new snapshot taken)
    accepted_2 = criterion.accept(candidate_2)
    assert accepted_2 is True
    # Check true front (live archive)
    assert len(criterion.true_front.front) == 2
    assert candidate_2 in criterion.true_front.front
    # Check snapshot (remains unchanged)
    assert criterion.front_snapshot == [candidate_1]

    # 3. Reject a duplicate solution (Mock ParetoFront rejects duplicates)
    accepted_3 = criterion.accept(candidate_3_duplicate)
    assert accepted_3 is False
    # True front size remains 2
    assert len(criterion.true_front.front) == 2


def test_get_one_current_solution_iterates_through_snapshot():
    """Test that solutions are popped from the front_snapshot list until empty."""
    criterion = AcceptBatch()
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


def test_take_snapshot_makes_correct_snapshot():
    """Test that solutions are popped from the front_snapshot list until empty."""
    criterion = AcceptBatch()
    s1, s2, s3 = Solution((10.0,), 1), Solution((20.0,), 2), Solution((30.0,), 3)

    # Taking multiple snapshots should not change the snapshot without changes in the front
    criterion.true_front.front = [s1, s2, s3]
    criterion._take_snapshot()
    assert criterion.front_snapshot == [s1, s2, s3]

    criterion.true_front.front = [s3, s2, s1]
    criterion._take_snapshot()
    assert criterion.front_snapshot == [s3, s2, s1]

    criterion.true_front.front = [s1, s2, s3]
    criterion._take_snapshot()
    assert criterion.front_snapshot == [s1, s2, s3]


def test_get_one_current_solution_triggers_snapshot_when_empty():
    """Test that an empty front_snapshot triggers a call to _take_snapshot from true_front."""
    criterion = AcceptBatch()

    # 1. Setup true_front (live archive)
    criterion.true_front.front.extend(
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


def test_get_one_current_solution_raises_value_error_when_empty():
    """Test the ValueError when both true_front and front_snapshot are empty."""
    criterion = AcceptBatch()
    criterion.true_front.clear()  # Ensure true_front is empty
    criterion.front_snapshot.clear()  # Ensure snapshot is empty

    with pytest.raises(
        ValueError, match="Archive is empty and all solutions have been processed."
    ):
        criterion.get_one_current_solution()


def test_clear_resets_both_internal_structures():
    """Test that clear resets both the front_snapshot and the true_front archive."""
    criterion = AcceptBatch()

    # Populate both structures
    criterion.true_front.accept(Solution((1.0,), 1))
    criterion.front_snapshot.append(Solution((2.0,), 2))

    assert criterion.true_front.front
    assert criterion.front_snapshot

    criterion.clear()

    assert not criterion.true_front.front
    assert not criterion.front_snapshot


def mock_custom_comparator(
    candidate: Solution, existing: Solution
) -> ComparisonResult:
    # A simple mock comparator, not strictly used for logic here, but required for init
    return ComparisonResult.NON_DOMINATED

class MockParetoFront:
    """Mock ParetoFront to track calls and store solutions for return."""
    def __init__(self, comparison_function: Callable = None):
        self.compare_solutions = comparison_function
        self.accepted_candidates = []
        self.clear_called = False
        # Solutions the front will "report" when asked
        self._solutions_to_report: list[Solution] = []

    def accept(self, candidate: Solution) -> bool:
        """Simulate acceptance. Always accepts unless data_id is 99."""
        self.accepted_candidates.append(candidate)
        if candidate.data_id == 99:
            return False # Simulate rejection
        
        # Add to solutions to report if accepted (simulating Pareto archive growth)
        if candidate not in self._solutions_to_report:
            self._solutions_to_report.append(candidate)
        return True

    def get_all_solutions(self) -> list[Solution]:
        """Returns all solutions, used by _take_snapshot and get_all_solutions."""
        return self._solutions_to_report[:]

    def get_one_current_solution(self) -> Solution:
        raise NotImplementedError("AcceptBatchWrapped does not use this method.")

    def clear(self):
        self.clear_called = True
        self.accepted_candidates.clear()
        self._solutions_to_report.clear()

def test_init_sets_up_fronts():
    """Test that true_front uses default comparison and custom_front uses the provided function."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)

    # Check that the default (true) front was initialized without the custom comparator
    assert criterion.true_front.compare_solutions is None

    # Check that the custom front was initialized with the custom comparator
    assert criterion.custom_front.compare_solutions is mock_custom_comparator

    # Check that the snapshot starts empty
    assert criterion.front_snapshot == []


def test_accept_updates_both_fronts_and_returns_custom_result():
    """Test that both internal fronts receive the candidate and the return value comes from custom_front."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)
    candidate_accepted = Solution((1.0,), 101)
    candidate_rejected = Solution((2.0,), 99)  # Mock front rejects ID 99

    # Test Accepted Case
    # Note: Snapshot is empty, but accept doesn't trigger snapshot in this clean version.
    result_accepted = criterion.accept(candidate_accepted)
    assert result_accepted is True
    assert candidate_accepted in criterion.true_front.accepted_candidates
    assert candidate_accepted in criterion.custom_front.accepted_candidates

    # Test Rejected Case
    result_rejected = criterion.accept(candidate_rejected)
    assert result_rejected is False
    assert candidate_rejected in criterion.true_front.accepted_candidates
    assert candidate_rejected in criterion.custom_front.accepted_candidates

    # Check that snapshot is still empty, as no solution has been requested yet
    assert not criterion.front_snapshot


def test_get_one_current_solution_triggers_initial_snapshot():
    """Test that the first call to get_one_current_solution takes a snapshot if the batch is empty."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)
    s1, s2 = Solution((10.0,), 1), Solution((20.0,), 2)

    # Manually populate the custom front (live archive)
    criterion.custom_front._solutions_to_report = [s1, s2]

    # Ensure snapshot is empty
    criterion.front_snapshot.clear()

    # First call must trigger snapshot and pop the first item
    solution_pop = criterion.get_one_current_solution()

    # Check that snapshot was taken (2 items - 1 popped = 1 remaining)
    assert len(criterion.front_snapshot) == 1

    # Check that the popped solution is from the front
    assert solution_pop in [s1, s2]


def test_get_one_current_solution_iterates_through_snapshot():
    """Test that solutions are popped from the front_snapshot list until empty."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)
    s1, s2, s3 = Solution((10.0,), 1), Solution((20.0,), 2), Solution((30.0,), 3)

    # Manually set the snapshot for iteration testing (simulating a prior _take_snapshot)
    criterion.front_snapshot = [s1, s2, s3]

    # Pop returns the last element (LIFO on the list)
    sol_3 = criterion.get_one_current_solution()
    assert sol_3.data_id == 3
    assert len(criterion.front_snapshot) == 2

    sol_2 = criterion.get_one_current_solution()
    assert sol_2.data_id == 2
    assert len(criterion.front_snapshot) == 1

    sol_1 = criterion.get_one_current_solution()
    assert sol_1.data_id == 1
    assert not criterion.front_snapshot  # Snapshot is now empty


def test_get_one_current_solution_refreshes_snapshot_when_exhausted():
    """Test that snapshot is refreshed from the custom_front when the old one is empty."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)
    s_old = Solution((10.0,), "old")
    s_new = Solution((1.0,), "new")

    # 1. Setup initial state and exhaust the snapshot
    criterion.custom_front._solutions_to_report = [s_old]
    criterion.front_snapshot = [s_old]
    _ = criterion.get_one_current_solution()
    assert not criterion.front_snapshot

    # 2. Add a new solution to the live custom front
    criterion.custom_front._solutions_to_report.append(
        s_new
    )  # Now custom front is [s_old, s_new]

    # 3. Request a new solution (triggers new snapshot: [s_old, s_new])
    sol_pop_1 = criterion.get_one_current_solution()
    assert len(criterion.front_snapshot) == 1  # One remaining

    # The popped solution should be the LIFO one from the new snapshot, s_new
    assert sol_pop_1 is s_new

    # 4. Pop the last remaining solution
    sol_pop_2 = criterion.get_one_current_solution()
    assert sol_pop_2 is s_old
    assert not criterion.front_snapshot


def test_get_one_current_solution_raises_value_error_when_empty_wrapped():
    """Test the ValueError when both custom_front and front_snapshot are empty."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)
    criterion.custom_front.clear()  # Ensure the source front is empty
    criterion.front_snapshot.clear()  # Ensure snapshot is empty

    with pytest.raises(ValueError, match="Archive is empty."):
        criterion.get_one_current_solution()


def test_get_all_solutions_comes_from_true_front():
    """Test that the true Pareto front is requested from the true_front archive."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)
    true_front_solutions = [Solution((5.0,), 5), Solution((6.0,), 6)]

    # Set the return values for the true front
    criterion.true_front._solutions_to_report = true_front_solutions

    # Check that the result matches the true_front's list
    all_solutions = criterion.get_all_solutions()
    assert all_solutions == true_front_solutions
    # Ensure it's a copy
    assert all_solutions is not criterion.true_front._solutions_to_report


def test_clear_resets_all_internal_structures():
    """Test that clear resets all three internal structures."""
    criterion = AcceptBatchWrapped(mock_custom_comparator)

    # Populate all structures
    criterion.true_front.accept(Solution((1.0,), 1))
    criterion.custom_front.accept(Solution((2.0,), 2))
    criterion.front_snapshot.append(Solution((3.0,), 3))

    # Reset call trackers for clarity
    criterion.true_front.clear_called = False
    criterion.custom_front.clear_called = False

    criterion.clear()

    # Verify both clear methods were called
    assert criterion.true_front.clear_called is True
    assert criterion.custom_front.clear_called is True

    # Verify the snapshot list is empty
    assert not criterion.front_snapshot
