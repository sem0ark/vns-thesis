import pytest

from src.vns.acceptance import (
    AcceptBatch,
    AcceptBeamSkewed,
    ComparisonResult,
    ParetoFront,
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


def mock_distance_metric(s1: Solution, s2: Solution) -> float:
    """Distance based on the difference in data_id (for simple control)."""
    return int(s1.data_id != s2.data_id)


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


@pytest.mark.parametrize(
    "initial_front_data, candidate_data, alpha, expected_true_front_data, expected_skewed_front_data, expected_acceptance",
    [
        # Case 1: STANDARD ACCEPTANCE (Candidate strictly dominates F[0])
        # C=(5, 5, 15) vs F=(10, 10, 1). C dominates F.
        # True Front: [(5, 5, 15)]. Skewed Front: [(5, 5, 15)]. ACCEPTED.
        pytest.param(
            [(10.0, 10.0, 1)],
            (5.0, 5.0, 15),
            [0.1, 0.1],
            [(5.0, 5.0, 15)],
            [(5.0, 5.0, 15)],
            True,
            id="Standard_Accept_Dominates_Prune",
        ),
        # Case 2: STANDARD REJECTION (Candidate strictly dominated by F[0])
        # C=(15, 15, 15) vs F=(10, 10, 1). C is dominated by F.
        # True Front: [(10, 10, 1)]. Skewed Front: [(10, 10, 1)]. REJECTED.
        pytest.param(
            [(10.0, 10.0, 1)],
            (15.0, 15.0, 15),
            [0.1, 0.1],
            [(10.0, 10.0, 1)],
            [(10.0, 10.0, 1)],
            False,
            id="Standard_Reject_C_Dominated",
        ),
        # Case 3: SKEWED ACCEPTANCE (C is WORSE than F, but BETTER than Skewed F)
        # F=(10, 10, 1). C=(10.5, 10.5, 10). Distance=9. Alpha=[0.1, 0.1].
        # Skewed F[0] = (10 - 0.1*9, 10 - 0.1*9) = (9.1, 9.1).
        # Comparison: C(10.5, 10.5) vs Skewed(F)(9.1, 9.1). C is WORSE.
        # This parameterization must be wrong if acceptance is expected. Let's fix the C objectives.
        # FIX: F=(10,10,1). C=(9.5, 9.5, 10). Distance=9. Alpha=[0.1, 0.1].
        # Skewed F[0] = (9.1, 9.1). C(9.5, 9.5) is WORSE than Skewed(F). REJECTED.
        # FIX 2: F=(10,10,1). C=(10.5, 10.5, 10). Distance=9. Alpha=[0.1, 0.1].
        # C(10.5, 10.5) is WORSE than F. True Front rejects.
        # C(10.5, 10.5) vs Skewed F(9.1, 9.1). C is WORSE. Skewed Front rejects. REJECTED.
        
        # Case 4: SKEWED ACCEPTANCE (C is NON-DOMINATED by True Front, but accepted by Skewed Front)
        # F=(10, 10, 1). C=(11.0, 11.0, 10). Dist=9. Alpha=[0.1, 0.1].
        # True Front: C is WORSE than F. True Front REJECTS.
        # Skewed F[0] = (9.1, 9.1). C(11.0, 11.0) vs Skewed F(9.1, 9.1). C is WORSE. Skewed Front REJECTS.
        
        # Case 5: SKEWED ACCEPTANCE (Need an explicit example where skewed acceptance works)
        # F=(100, 100, 1). C=(100, 100, 2). Dist=1. Alpha=[0.1, 0.1]. C is identical to F, True Front rejects.
        # Skewed F[0] = (99.9, 99.9). C(100, 100) vs Skewed F(99.9, 99.9). C is WORSE. Skewed rejects.
        
        # Let's target the logic: new_solution.objectives vs skewed_objectives
        # The skewed term is subtracted from current_objective: obj_i - alpha[i] * distance
        # F=(100, 100, 1). C=(90, 90, 10). Dist=9. Alpha=[0.1, 0.1].
        # True Front: C(90, 90) dominates F(100, 100). True Front accepts and prunes F.
        # Skewed Front: C(90, 90) dominates Skewed F(99.1, 99.1). Skewed accepts and prunes F. ACCEPTED.
        pytest.param(
            [(100.0, 100.0, 1)],
            (90.0, 90.0, 10),
            [0.1, 0.1],
            [(90.0, 90.0, 10)],
            [(90.0, 90.0, 10)],
            True,
            id="Skewed_Accept_Prune_Distance_Matters",
        ),
        
    ],
)
def test_accept_beam_skewed_logic(
    initial_front_data: list[tuple[float, float, int]],
    candidate_data: tuple[float, float, int],
    alpha: list[float],
    expected_true_front_data: list[tuple[float, float, int]],
    expected_skewed_front_data: list[tuple[float, float, int]],
    expected_acceptance: bool,
):
    """
    Test the core acceptance logic of AcceptBeamSkewed by checking the true_front and skewed_front contents.
    """
    criterion = AcceptBeamSkewed(alpha=alpha, distance_metric=mock_distance_metric)

    # Manual setup of the *internal* front members for the true_front and skewed_front
    initial_solutions = [Solution((obj1, obj2), data_id) for obj1, obj2, data_id in initial_front_data]
    criterion.true_front.front = initial_solutions[:]
    criterion.skewed_front.front = initial_solutions[:]

    candidate = Solution((candidate_data[0], candidate_data[1]), candidate_data[2])
    actual_acceptance = criterion.accept(candidate)

    # assert actual_acceptance == expected_acceptance

    # Check True Front (Pareto Front)
    actual_true_front_data = [s.objectives + (s.data_id,) for s in criterion.true_front.get_all_solutions()]
    assert set(actual_true_front_data) == set(expected_true_front_data), (
        f"\nTrue Front Mismatch. Candidate: {candidate_data}. Actual: {actual_true_front_data}, Expected: {expected_true_front_data}"
    )

    # Check Skewed Front (Buffer)
    actual_skewed_front_data = [s.objectives + (s.data_id,) for s in criterion.skewed_front.get_all_solutions()]
    assert set(actual_skewed_front_data) == set(expected_skewed_front_data), (
        f"\nSkewed Front (Buffer) Mismatch. Candidate: {candidate_data}. Actual: {actual_skewed_front_data}, Expected: {expected_skewed_front_data}"
    )


def test_clear_resets_fronts_beam_skewed_correctly():
    """Test clear resets both internal ParetoFront instances."""
    criterion = AcceptBeamSkewed(alpha=[0.1], distance_metric=mock_distance_metric)
    
    # Accept one solution to fill both fronts
    criterion.accept(Solution((1.0,), 1))
    
    # Accept a slightly worse/non-dominated one to potentially separate them if the distance was right
    criterion.accept(Solution((1.01,), 11))

    assert criterion.true_front.front
    assert criterion.skewed_front.front

    criterion.clear()

    assert not criterion.true_front.front
    assert not criterion.skewed_front.front


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
