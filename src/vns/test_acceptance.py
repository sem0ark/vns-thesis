import pytest

from src.vns.acceptance import (
    AcceptBatch,
    AcceptBeam,
    AcceptBeamSkewed,
    ComparisonResult,
    is_dominating_min,
)


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


def mock_distance_metric(s1: Solution, s2: Solution) -> float:
    """Distance based on the difference in data_id (for simple control)."""
    return int(s1.data_id != s2.data_id)


def create_initial_skewed_instance(
    initial_front_data: list[tuple[float, int]], alpha: list[float]
):
    """Creates a fresh AcceptBeamSkewed instance with the front initialized."""
    instance = AcceptBeamSkewed(alpha=alpha, distance_metric=mock_distance_metric)
    # Front members are accepted via a mock process for setup
    for obj, data_id in initial_front_data:
        instance.front.append(Solution(obj, data_id))
    return instance


@pytest.mark.parametrize(
    "initial_front_data, candidate_data, alpha, expected_front_data, expected_buffer_data, expected_acceptance",
    [
        # Case 1: Standard Pareto Acceptance (Accepted by super())
        # F = (10, 10, ID=1). C = (5, 5, ID=15). C strictly dominates F.
        # Front: [(5.0, 5.0, 15)], Buffer: []
        pytest.param(
            [(10.0, 10.0, 1)],
            (5.0, 5.0, 15),
            [0.1, 0.1],
            [(5.0, 5.0, 15)],
            [],
            True,
            id="Standard_Accept_Dominates_One",
        ),
        # Case 2: Standard Pareto Rejection (Rejected by super(), C is dominated by F)
        # F = (5, 5, ID=5). C = (10, 10, ID=10). F strictly dominates C.
        # Front: [(5.0, 5.0, 5)], Buffer: []
        pytest.param(
            [(5.0, 5.0, 5)],
            (10.0, 10.0, 10),
            [0.1, 0.1],
            [(5.0, 5.0, 5)],
            [],
            False,
            id="Standard_Reject_C_Dominated",
        ),
        pytest.param(
            [(10.0, 10.0, 1)],
            (11.0, 9.0, 11),
            [0.1, 0.1],
            [(10.0, 10.0, 1), (11.0, 9.0, 11)],
            [],
            True,
            id="Standard_Accept_NonDominated",
        ),
        pytest.param(
            [(10.0, 10.0, 1)],
            (10.01, 10.0, 11),
            [0.1, 0.1],
            [(10.0, 10.0, 1)],
            [(10.01, 10.0, 11)],
            True,
            id="Skewed_Accept_NonDominated_To_Buffer",
        ),
        pytest.param(
            [(10.0, 10.0, 1)],
            (10.01, 10.01, 11),
            [0.0, 0.0],
            [(10.0, 10.0, 1)],
            [],
            False,
            id="Skewed_Reject_Dominated_Completely",
        ),
        pytest.param(
            [(10.0, 10.0, 1)],
            (20.0, 20.0, 11),
            [0.1, 0.1],
            [(10.0, 10.0, 1)],
            [],
            False,
            id="Skewed_Reject_Dominated_Completely",
        ),
    ],
)
def test_accept_beam_skewed_logic(
    initial_front_data: list[tuple[float, float, int]],  # (Obj1, Obj2, ID)
    candidate_data: tuple[float, float, int],  # (obj1, obj2, id)
    alpha: list[float],
    expected_front_data: list[tuple[float, float, int]],
    expected_buffer_data: list[tuple[float, float, int]],
    expected_acceptance: bool,
):
    """
    Test the core acceptance logic of AcceptBeamSkewed.
    """

    criterion = AcceptBeamSkewed(alpha=alpha, distance_metric=mock_distance_metric)

    for obj1, obj2, data_id in initial_front_data:
        criterion.front.append(Solution((obj1, obj2), data_id))

    candidate = Solution((candidate_data[0], candidate_data[1]), candidate_data[2])
    actual_acceptance = criterion.accept(candidate)
    assert actual_acceptance == expected_acceptance, candidate_data

    actual_front_data = [
        s.objectives + (s.data_id,) for s in criterion.get_all_solutions()
    ]
    assert set(actual_front_data) == set(expected_front_data), (
        f"\nFront Mismatch. Actual: {actual_front_data}, Expected: {expected_front_data}"
    )

    actual_buffer_data = [s.objectives + (s.data_id,) for s in criterion.skewed_buffer]
    assert set(actual_buffer_data) == set(expected_buffer_data), (
        f"\nBuffer Mismatch. Actual: {actual_buffer_data}, Expected: {expected_buffer_data}"
    )


def test_clear_resets_fronts_beam():
    criterion = AcceptBeam()
    criterion.accept(Solution((1.0,), 1))

    assert criterion.front

    criterion.clear()

    assert not criterion.front


def test_clear_resets_fronts_beam_skewed():
    criterion = AcceptBeamSkewed(alpha=[0.1], distance_metric=mock_distance_metric)
    criterion.accept(Solution((1.0,), 1))
    criterion.accept(Solution((1.01,), 11))

    assert criterion.front
    assert criterion.skewed_buffer

    criterion.clear()

    assert not criterion.front
    assert not criterion.skewed_buffer


def test_accept_updates_upcoming_front():
    """Test that accept always delegates to and updates the upcoming_front."""
    criterion = AcceptBatch()
    candidate_1 = Solution((10.0,), 1)
    candidate_2 = Solution((20.0,), 2)
    candidate_3_duplicate = Solution((10.0,), 1)

    # 1. Accept unique solution
    accepted_1 = criterion.accept(candidate_1)
    assert accepted_1 is True
    assert (
        candidate_1 in criterion.front
    )  # was added to the upcoming front and swapped immediately
    assert not criterion.upcoming_front.front  # Should be empty after first swap

    # After the first accept, the front is empty, so a swap occurs immediately.
    assert criterion.front == [candidate_1]
    assert not criterion.upcoming_front.front

    # 2. Accept another unique solution (front is NOT empty now)
    accepted_2 = criterion.accept(candidate_2)
    assert accepted_2 is True
    assert candidate_2 in criterion.upcoming_front.front
    assert criterion.front == [candidate_1]  # Front remains unchanged

    # 3. Reject a duplicate solution
    accepted_3 = criterion.accept(candidate_3_duplicate)
    assert accepted_3 is False
    assert len(criterion.upcoming_front.front) == 1  # Only candidate_2 remains


def test_get_one_current_solution_iterates_through_front():
    """Test that solutions are popped from the current front until empty."""
    criterion = AcceptBatch()

    criterion.front = [Solution((10.0,), 1), Solution((20.0,), 2), Solution((30.0,), 3)]

    sol_3 = criterion.get_one_current_solution()
    assert sol_3.data_id == 3
    assert len(criterion.front) == 2

    sol_2 = criterion.get_one_current_solution()
    assert sol_2.data_id == 2
    assert len(criterion.front) == 1

    sol_1 = criterion.get_one_current_solution()
    assert sol_1.data_id == 1
    assert not criterion.front  # Front is now empty


def test_get_one_current_solution_triggers_swap_when_empty():
    """Test that an empty front triggers a swap with the upcoming front."""
    criterion = AcceptBatch()

    # 1. Setup upcoming_front
    upcoming_data = [Solution((100.0,), 100), Solution((200.0,), 200)]
    criterion.upcoming_front.front.extend(upcoming_data)

    # 2. Ensure current front is empty
    criterion.front = []

    # 3. Call get_one_current_solution - this must trigger _swap_fronts()
    sol_pop = criterion.get_one_current_solution()

    # Check that swap occurred
    assert len(criterion.front) == 1  # One solution popped, one remains
    assert not criterion.upcoming_front.front

    # Check that a solution from the UPCOMING front was returned
    assert sol_pop.data_id in [100, 200]

    # Check that the remaining solution is in the current front
    remaining_id = 100 if sol_pop.data_id == 200 else 200
    assert criterion.front[0].data_id == remaining_id


def test_get_one_current_solution_unreachable():
    """Test the ValueError when both front and upcoming_front are empty."""
    criterion = AcceptBatch()
    criterion.front = []
    criterion.upcoming_front.front = []

    with pytest.raises(ValueError, match="Unreachable"):
        criterion.get_one_current_solution()


def test_clear_resets_both_fronts():
    """Test that clear resets both the current front and the upcoming front archive."""
    criterion = AcceptBatch()

    # Populate both fronts
    criterion.front = [Solution((1.0,), 1)]
    criterion.upcoming_front.front.append(Solution((2.0,), 2))

    assert criterion.front
    assert criterion.upcoming_front.front

    criterion.clear()

    assert not criterion.front
    assert not criterion.upcoming_front.front
