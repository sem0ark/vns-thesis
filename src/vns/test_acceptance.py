from unittest.mock import Mock

import pytest

from src.vns.abstract import MOKPSolution
from src.vns.acceptance import (TakeBigger, TakeBiggerSkewed, TakeSmaller,
                                TakeSmallerSkewed, dominates_maximize,
                                dominates_minimize)


class TestDominatesMinimize:
    def test_strict_dominance_2d(self):
        assert dominates_minimize((5.0, 5.0), (10.0, 10.0), 0.0) is True

    def test_strict_dominance_3d(self):
        assert dominates_minimize((1.0, 2.0, 3.0), (2.0, 3.0, 4.0), 0.0) is True

    def test_partial_dominance_better_one_equal_other(self):
        assert dominates_minimize((5.0, 10.0), (10.0, 10.0), 0.0) is True

    def test_partial_dominance_equal_one_better_other(self):
        assert dominates_minimize((10.0, 5.0), (10.0, 10.0), 0.0) is True

    def test_partial_dominance_with_small_epsilon(self):
        assert dominates_minimize((1.0, 2.0), (1.0 + 1e-7, 2.0 + 1e-7), 0.0) is False

    def test_non_dominance_worse_on_one(self):
        assert dominates_minimize((10.0, 5.0), (5.0, 10.0), 0.0) is False

    def test_non_dominance_all_worse(self):
        assert dominates_minimize((10.0, 10.0), (5.0, 5.0), 0.0) is False

    def test_non_dominance_all_equal(self):
        assert dominates_minimize((10.0, 10.0), (10.0, 10.0), 0.0) is False

    def test_buffer_allows_worsening_but_no_strict_improvement(self):
        assert dominates_minimize((10.0, 12.0), (10.0, 10.0), 5.0) is False

    def test_buffer_allows_worsening_with_strict_improvement(self):
        assert dominates_minimize((5.0, 12.0), (10.0, 10.0), 5.0) is True

    def test_buffer_exhausted(self):
        assert dominates_minimize((5.0, 16.0), (10.0, 10.0), 5.0) is False

    def test_buffer_just_enough(self):
        assert dominates_minimize((5.0, 15.0), (10.0, 10.0), 5.1) is True

    def test_buffer_multiple_worsening(self):
        assert dominates_minimize((10.0, 11.0, 12.0), (10.0, 10.0, 10.0), 3.0) is False

    def test_buffer_multiple_worsening_with_improvement(self):
        assert dominates_minimize((9.0, 11.0, 12.0), (10.0, 10.0, 10.0), 3.1) is True

    def test_buffer_multiple_worsening_exhausted_with_improvement(self):
        assert dominates_minimize((9.0, 11.0, 13.0), (10.0, 10.0, 10.0), 3.0) is False

    def test_value_error_different_lengths(self):
        with pytest.raises(
            ValueError,
            match="Objective vectors must have the same number of objectives.",
        ):
            dominates_minimize((1.0, 2.0), (1.0, 2.0, 3.0), 0.0)


class TestDominatesMaximize:
    def test_strict_dominance_2d(self):
        assert dominates_maximize((10.0, 10.0), (5.0, 5.0), 0.0) is True

    def test_strict_dominance_3d(self):
        assert dominates_maximize((2.0, 3.0, 4.0), (1.0, 2.0, 3.0), 0.0) is True

    def test_partial_dominance_better_one_equal_other(self):
        assert dominates_maximize((10.0, 5.0), (5.0, 5.0), 0.0) is True

    def test_partial_dominance_equal_one_better_other(self):
        assert dominates_maximize((5.0, 10.0), (5.0, 5.0), 0.0) is True

    def test_partial_dominance_with_small_epsilon(self):
        assert dominates_maximize((1.0 + 1e-7, 2.0 + 1e-7), (1.0, 2.0), 0.0) is False

    def test_non_dominance_worse_on_one(self):
        assert dominates_maximize((5.0, 10.0), (10.0, 5.0), 0.0) is False

    def test_non_dominance_all_worse(self):
        assert dominates_maximize((5.0, 5.0), (10.0, 10.0), 0.0) is False

    def test_non_dominance_all_equal(self):
        assert dominates_maximize((10.0, 10.0), (10.0, 10.0), 0.0) is False

    def test_buffer_allows_worsening_but_no_strict_improvement(self):
        assert dominates_maximize((10.0, 8.0), (10.0, 10.0), 5.0) is False

    def test_buffer_allows_worsening_with_strict_improvement(self):
        assert dominates_maximize((15.0, 8.0), (10.0, 10.0), 5.0) is True

    def test_buffer_exhausted(self):
        assert dominates_maximize((15.0, 4.0), (10.0, 10.0), 5.0) is False

    def test_buffer_just_enough(self):
        assert dominates_maximize((15.0, 5.0), (10.0, 10.0), 5.1) is True

    def test_buffer_multiple_worsening(self):
        assert dominates_maximize((10.0, 9.0, 8.0), (10.0, 10.0, 10.0), 3.0) is False

    def test_buffer_multiple_worsening_with_improvement(self):
        assert dominates_maximize((11.0, 9.0, 8.0), (10.0, 10.0, 10.0), 3.1) is True

    def test_buffer_multiple_worsening_exhausted_with_improvement(self):
        assert dominates_maximize((11.0, 9.0, 7.0), (10.0, 10.0, 10.0), 3.0) is False

    def test_value_error_different_lengths(self):
        with pytest.raises(
            ValueError,
            match="Objective vectors must have the same number of objectives.",
        ):
            dominates_maximize((1.0, 2.0), (1.0, 2.0, 3.0), 0.0)


class TestTakeSmaller:
    @pytest.fixture
    def setup_taker(self):
        taker = TakeSmaller(buffer_size=3)
        return taker

    def test_initial_accept(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        assert taker.accept(s1) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1
        assert len(taker.buffer) == 0

    def test_accept_dominating_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(5.0, 5.0))  # Dominates s1
        assert taker.accept(s2) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s2
        assert len(taker.buffer) == 1  # s1 should be in buffer

    def test_accept_dominated_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(5.0, 5.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(10.0, 10.0))  # Dominated by s1
        assert taker.accept(s2) is False  # Archive does not change for the better
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1
        assert len(taker.buffer) == 0  # s2 is just discarded as not relevant

    def test_accept_non_dominated_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 5.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(5.0, 10.0))  # Non-dominated by s1
        assert taker.accept(s2) is True
        assert len(taker.archive) == 2
        assert s1 in taker.archive and s2 in taker.archive
        assert len(taker.buffer) == 0

    def test_accept_duplicate_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        taker.accept(s1)
        s_duplicate = Mock(
            data="s_dup", objectives=(10.0, 10.0)
        )  # Same objectives, different data
        assert (
            taker.accept(s_duplicate) is False
        )  # Archive should not change if same objectives and data (or effectively same solution)
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1

    def test_buffer_full(self, setup_taker):
        taker = TakeSmaller(buffer_size=1)  # Buffer size 1
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(5.0, 5.0))  # Dominates s1
        taker.accept(s2)  # s1 goes to buffer
        assert len(taker.buffer) == 1
        assert taker.buffer[0] == s1

        s3 = Mock(data="s3", objectives=(2.0, 2.0))  # Dominates s2
        taker.accept(s3)  # s2 goes to buffer, s1 should be pushed out
        assert len(taker.buffer) == 1
        assert taker.buffer[0] == s2  # s1 was pushed out by s2

    def test_get_all_solutions(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 5.0))
        s2 = Mock(data="s2", objectives=(5.0, 10.0))
        s3 = Mock(data="s3", objectives=(1.0, 1.0))  # Dominates s1, s2

        taker.accept(s1)
        taker.accept(s2)
        taker.accept(s3)

        assert len(taker.get_all_solutions()) == 1  # Only s3 should be in archive
        assert taker.get_all_solutions()[0] == s3

    def test_get_one_current_solution(self, setup_taker):
        taker = setup_taker
        with pytest.raises(ValueError, match="No solutions available"):
            taker.get_one_current_solution()  # Test when empty

        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        taker.accept(s1)
        assert taker.get_one_current_solution() == s1  # Should return s1

        s2 = Mock(data="s2", objectives=(5.0, 5.0))
        taker.accept(s2)
        assert (
            taker.get_one_current_solution() == s2
        )  # Should return s2 (which replaced s1)

        # Add a non-dominated one so archive has multiple
        s3 = Mock(data="s3", objectives=(4.0, 6.0))
        taker.accept(s3)
        # The specific returned solution from get_one_current_solution will vary based on internal logic (random.choice for example)
        # For our mock, it just takes the first, so this test might need adjustment if random is truly implemented.
        assert (
            taker.get_one_current_solution() in taker.archive
            or taker.get_one_current_solution() in taker.buffer
        )  # More robust check for the simplified get_one_current_solution


class TestTakeSmallerSkewed:
    @pytest.fixture
    def setup_skewed_taker(self):
        # Dummy distance metric for testing
        def dummy_distance(s1: MOKPSolution, s2: MOKPSolution) -> float:
            # Simple Manhattan distance on objectives for test purposes
            return sum(abs(o1 - o2) for o1, o2 in zip(s1.objectives, s2.objectives))

        taker = TakeSmallerSkewed(
            alpha=0.5, distance_metric=dummy_distance, buffer_size=3
        )
        return taker

    def test_initial_accept(self, setup_skewed_taker):
        taker = setup_skewed_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        assert taker.accept(s1) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1
        assert len(taker.buffer) == 0

    def test_accept_dominating_solution(self, setup_skewed_taker):
        taker = setup_skewed_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(5.0, 5.0))  # Dominates s1
        assert taker.accept(s2) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s2
        assert len(taker.buffer) == 1  # s1 is moved to buffer by super().accept

    def test_accept_skewed_acceptable_not_dominating(self, setup_skewed_taker):
        taker = setup_skewed_taker
        s_current = Mock(data="curr", objectives=(10.0, 10.0))
        taker.accept(s_current)  # Archive: [s_current]

        # Candidate is not strictly dominating but is "skewed acceptable"
        # Let's say s_candidate = (11, 9)
        # current = (10, 10)
        # dominates_minimize(s_candidate_obj, s_current_obj, buffer_val) where buffer_val = alpha * dist(s_candidate, s_current)
        # s_candidate worse on obj1 (11>10), better on obj2 (9<10) -> non-dominated in standard sense (False for dominates)
        # dist((11,9), (10,10)) = |11-10| + |9-10| = 1 + 1 = 2
        # buffer_val = 0.5 * 2 = 1.0
        # dominates_minimize((11,9), (10,10), 1.0)
        #   obj0: 11 > 10, buffer -= 1.0 -> buffer = 0.0
        #   obj1: 9 < 10, at_least_one_strictly_better = True
        # Returns True. So (11,9) dominates (10,10) with a buffer of 1.0.

        s_candidate = Mock(data="skewed", objectives=(11.0, 9.0))

        # Before this call, archive has only s_current.
        # super().accept(s_candidate) will check if s_candidate dominates s_current.
        # dominates_minimize((11,9), (10,10), 0.0) -> False (neither dominates for buffer=0)
        # So super().accept will add s_candidate to archive, archive becomes [s_current, s_candidate]. Return True.
        # Then the skewed logic is applied.

        # Test 1: Candidate is non-dominated for 0.0 buffer (standard Pareto)
        # s_current=(10,5), s_candidate=(5,10)
        s_current_nondom = Mock(data="curr_nd", objectives=(10.0, 5.0))
        taker = TakeSmallerSkewed(
            alpha=0.5, distance_metric=setup_skewed_taker.distance_metric, buffer_size=3
        )
        taker.accept(s_current_nondom)  # Archive: [s_current_nondom]

        s_candidate_nondom = Mock(data="cand_nd", objectives=(5.0, 10.0))
        assert (
            taker.accept(s_candidate_nondom) is True
        )  # Standard non-dominated added to archive
        assert len(taker.archive) == 2
        assert s_current_nondom in taker.archive and s_candidate_nondom in taker.archive
        assert len(taker.buffer) == 0  # No standard dominated solutions yet

        # Test 2: Candidate is only skewed-acceptable (not standard dominated/dominating or non-dominated)
        # Reset taker
        taker = TakeSmallerSkewed(
            alpha=0.5, distance_metric=setup_skewed_taker.distance_metric, buffer_size=3
        )
        s_base = Mock(data="base", objectives=(10.0, 10.0))
        taker.accept(s_base)  # Archive: [s_base]

        # Candidate: (11.0, 9.0)
        # vs (10.0, 10.0) with buffer=0.0 -> False (not Pareto dominating)
        # So super().accept(s_skewed) will return False and not change archive directly (it's dominated by nothing but also doesn't dominate s_base,
        # it would be added if it were strictly non-dominated, but it isn't, so it depends on implementation details of TakeSmaller.accept)
        # My TakeSmaller.accept adds non-dominated ones. (11,9) vs (10,10) is non-dominated. So it would be added to archive.
        # This implies the skewed logic as implemented in the given code (check against archive and add to buffer)
        # is for very strong candidates.

        # Let's create a scenario where skewed behavior is visible:
        # Candidate is dominated by base, but accept-able with buffer.
        # base=(5,5), candidate=(6,6)
        # dominates_minimize((6,6), (5,5), 0.0) -> False (6>5, 6>5)
        # dist((6,6), (5,5)) = 2.0. alpha*dist = 0.5*2.0 = 1.0
        # dominates_minimize((6,6), (5,5), 1.0)
        #   obj0: 6 > 5, buffer -= 1.0 -> buffer = 0.0
        #   obj1: 6 > 5, buffer -= 1.0 -> buffer = -1.0. Returns False. Not skewed acceptable.

        # Need a case where buffer_value allows acceptance of a seemingly "worse" solution
        # (new_objective[i] > current_objective[i] is allowed within buffer)
        # and there's a strict improvement somewhere.
        # This is where the interpretation of the provided `dominates_minimize` becomes key.
        # The `dominates_minimize` function with a buffer can return True even if `new_objective` is *worse* on some dimensions,
        # as long as the cumulative "worsening" is within the buffer, AND there is *at least one strictly better* dimension.

        # Example for skewed acceptance:
        # Initial archive: [s_a=(10,10)]
        # Candidate: s_b=(9,12)
        # Standard dominates_minimize((9,12), (10,10), 0.0) -> False (9<10, 12>10; non-dominated)
        # So s_b would be added to archive by super().accept. Archive: [s_a, s_b]. Buffer: []. Return True.
        # Then skewed check:
        # dominates_buffered((9,12), (10,10), alpha*dist((9,12), (10,10)) = 0.5 * (|9-10|+|12-10|) = 0.5 * (1+2) = 1.5)
        #   obj0: 9 < 10, at_least_one_strictly_better = True
        #   obj1: 12 > 10, buffer -= 2.0 -> buffer = -0.5. Returns False.
        # So `all_dom_buffered` would be False. `s_b` is NOT added to buffer by skewed logic.
        # This means the given skewed logic with `all()` is very restrictive.

        # Let's test the case where `all_dom_buffered` IS true.
        # Candidate dominates ALL solutions in the archive *with a buffer*.
        # Archive: [s1=(10,10), s2=(12,8)] (non-dominated pair)
        # Candidate: s_cand=(5,5) (dominates both s1, s2 without buffer)
        # super().accept(s_cand) -> archive = [s_cand], buffer = [s1, s2]. Returns True.
        # Now, check `all(self.dominates_buffered(s_cand.obj, sol.obj, alpha*dist(s_cand, sol)))`
        # vs s1: dominates_buffered((5,5), (10,10), 0.5*10) = dominates_minimize((5,5),(10,10),5.0) -> True
        # vs s2: dominates_buffered((5,5), (12,8), 0.5*10) = dominates_minimize((5,5),(12,8),5.0) -> True
        # Both are true. So `s_cand` is added to buffer.

        taker = setup_skewed_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        s2 = Mock(data="s2", objectives=(12.0, 8.0))
        taker.accept(s1)  # Archive: [s1]
        taker.accept(s2)  # Archive: [s1, s2] (assuming non-dominated)
        assert len(taker.archive) == 2
        assert len(taker.buffer) == 0

        s_cand_strong = Mock(
            data="strong", objectives=(5.0, 5.0)
        )  # Strongly dominates s1 and s2
        assert taker.accept(s_cand_strong) is True
        assert len(taker.archive) == 1  # Only s_cand_strong
        assert taker.archive[0] == s_cand_strong
        assert (
            len(taker.buffer) == 3
        )  # s1, s2 from super.accept, AND s_cand_strong from skewed logic

        # Test where candidate is not accepted into archive but is skewed-acceptable enough for buffer
        taker = TakeSmallerSkewed(
            alpha=0.5, distance_metric=setup_skewed_taker.distance_metric, buffer_size=3
        )
        s_base_obj = (10.0, 10.0)
        s_base = Mock(data="base", objectives=s_base_obj)
        taker.accept(s_base)  # Archive: [s_base]

        # s_cand_skewed is slightly worse on one, better on another, but dominated by s_base with buffer=0
        # For this test, let's craft a case where standard `dominates_minimize(new, current, 0.0)` is False (candidate is worse)
        # but `dominates_minimize(new, current, buffer_val)` is True.
        # current=(10,10), new=(11,9). New is worse on first, better on second.
        # dominates_minimize((11,9),(10,10),0.0) is False. So s_cand_skewed will NOT be added to archive by super.
        # Its objective is (11,9). Original objective (10,10).
        # distance = |11-10| + |9-10| = 1 + 1 = 2
        # buffer_val = 0.5 * 2 = 1.0
        # dominates_minimize((11,9),(10,10),1.0)
        #   (11>10) -> buffer -= 1.0 (becomes 0.0)
        #   (9<10) -> at_least_one_strictly_better = True
        # Returns True. So s_cand_skewed "dominates" with buffer.

        s_cand_skewed = Mock(data="skewed_only", objectives=(11.0, 9.0))
        # Initial archive: [s_base]. s_base dominates s_cand_skewed w/ buffer 0
        assert (
            taker.accept(s_cand_skewed) is False
        )  # Not accepted into archive by standard dominance
        assert len(taker.archive) == 1  # Archive remains [s_base]
        assert (
            len(taker.buffer) == 1
        )  # s_cand_skewed should be added to buffer by the skewed logic.


class TestTakeBigger:
    @pytest.fixture
    def setup_taker(self):
        taker = TakeBigger(buffer_size=3)
        return taker

    def test_initial_accept(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        assert taker.accept(s1) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1

    def test_accept_dominating_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(5.0, 5.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(10.0, 10.0))  # Dominates s1 (maximization)
        assert taker.accept(s2) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s2
        assert len(taker.buffer) == 1  # s1 in buffer

    def test_accept_dominated_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(5.0, 5.0))  # Dominated by s1
        assert taker.accept(s2) is False
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1
        assert len(taker.buffer) == 0

    def test_accept_non_dominated_solution(self, setup_taker):
        taker = setup_taker
        s1 = Mock(data="s1", objectives=(10.0, 5.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(5.0, 10.0))  # Non-dominated
        assert taker.accept(s2) is True
        assert len(taker.archive) == 2
        assert s1 in taker.archive and s2 in taker.archive
        assert len(taker.buffer) == 0


class TestTakeBiggerSkewed:
    @pytest.fixture
    def setup_skewed_taker(self):
        def dummy_distance(s1: MOKPSolution, s2: MOKPSolution) -> float:
            return sum(abs(o1 - o2) for o1, o2 in zip(s1.objectives, s2.objectives))

        taker = TakeBiggerSkewed(
            alpha=0.5, distance_metric=dummy_distance, buffer_size=3
        )
        return taker

    def test_initial_accept(self, setup_skewed_taker):
        taker = setup_skewed_taker
        s1 = Mock(data="s1", objectives=(10.0, 10.0))
        assert taker.accept(s1) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s1
        assert len(taker.buffer) == 0

    def test_accept_dominating_solution(self, setup_skewed_taker):
        taker = setup_skewed_taker
        s1 = Mock(data="s1", objectives=(5.0, 5.0))
        taker.accept(s1)
        s2 = Mock(data="s2", objectives=(10.0, 10.0))  # Dominates s1
        assert taker.accept(s2) is True
        assert len(taker.archive) == 1
        assert taker.archive[0] == s2
        assert len(taker.buffer) == 1  # s1 in buffer

    def test_accept_skewed_acceptable_not_dominating(self, setup_skewed_taker):
        taker = setup_skewed_taker
        s_base = Mock(data="base", objectives=(10.0, 10.0))
        taker.accept(s_base)  # Archive: [s_base]

        # Candidate: (9.0, 11.0)
        # vs (10.0, 10.0) with buffer=0.0 -> False (9<10, 11>10; non-dominated in standard sense)
        # So super().accept will add s_cand_skewed to archive.

        # Scenario where skewed logic adds to buffer because it dominates *all* in archive with buffer:
        taker = setup_skewed_taker
        s1 = Mock(data="s1", objectives=(5.0, 5.0))
        s2 = Mock(data="s2", objectives=(3.0, 7.0))
        taker.accept(s1)  # Archive: [s1]
        taker.accept(s2)  # Archive: [s1, s2] (non-dominated pair)
        assert len(taker.archive) == 2
        assert len(taker.buffer) == 0

        # Candidate: (10.0, 10.0) (dominates both s1, s2 without buffer)
        # super().accept(s_cand_strong) -> archive = [s_cand_strong], buffer = [s1, s2]. Returns True.
        # Now, check `all(self.dominates_buffered(s_cand.obj, sol.obj, alpha*dist(s_cand, sol)))`
        # vs s1: dominates_buffered((10,10), (5,5), 0.5*10) = dominates_maximize((10,10),(5,5),5.0) -> True
        # vs s2: dominates_buffered((10,10), (3,7), 0.5*10) = dominates_maximize((10,10),(3,7),5.0) -> True
        # Both are true. So `s_cand_strong` is added to buffer.

        s_cand_strong = Mock(
            data="strong", objectives=(10.0, 10.0)
        )  # Strongly dominates s1 and s2
        assert taker.accept(s_cand_strong) is True
        assert len(taker.archive) == 1  # Only s_cand_strong
        assert taker.archive[0] == s_cand_strong
        assert (
            len(taker.buffer) == 3
        )  # s1, s2 from super.accept, AND s_cand_strong from skewed logic

        # Test where candidate is not accepted into archive but is skewed-acceptable enough for buffer (Maximization)
        taker = TakeBiggerSkewed(
            alpha=0.5, distance_metric=setup_skewed_taker.distance_metric, buffer_size=3
        )
        s_base_obj = (10.0, 10.0)
        s_base = Mock(data="base", objectives=s_base_obj)
        taker.accept(s_base)  # Archive: [s_base]

        # s_cand_skewed is slightly better on one, worse on another, but not dominated by s_base with buffer=0
        # current=(10,10), new=(9,11). New is worse on first, better on second.
        # dominates_maximize((9,11),(10,10),0.0) is False. So s_cand_skewed will NOT be added to archive by super.
        # distance = |9-10| + |11-10| = 1 + 1 = 2
        # buffer_val = 0.5 * 2 = 1.0
        # dominates_maximize((9,11),(10,10),1.0)
        #   (9<10) -> buffer -= 1.0 (becomes 0.0)
        #   (11>10) -> at_least_one_strictly_better = True
        # Returns True. So s_cand_skewed "dominates" with buffer.

        s_cand_skewed = Mock(data="skewed_only", objectives=(9.0, 11.0))
        assert (
            taker.accept(s_cand_skewed) is False
        )  # Not accepted into archive by standard dominance
        assert len(taker.archive) == 1  # Archive remains [s_base]
        assert (
            len(taker.buffer) == 1
        )  # s_cand_skewed should be added to buffer by the skewed logic.
