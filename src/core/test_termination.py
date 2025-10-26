import pytest
from typing import Optional, Iterable, Generator
from unittest.mock import Mock, patch

from src.core.abstract import AcceptanceCriterion, OptimizerAbstract
from src.core.termination import (
    ChainedOptimizer,
    StoppableOptimizer,
    terminate_max_iterations,
    terminate_max_no_improvement,
    terminate_noop,
    terminate_time_based,
)


class Solution:
    def __init__(self, objectives):
        self.objectives = objectives


@pytest.fixture
def mock_optimizer_process() -> Generator[Optional[bool], None, None]:
    """A mock optimization generator that runs 'forever' (until StopIteration)."""
    # 3 internal steps, 1 failed cycle, 1 internal step, 1 success cycle, 1 failed cycle...
    sequence = [
        None,
        None,
        None,
        False,
        None,
        True,
        False,
        False,
        False,
        True,
        False,
        None,
    ]

    def generator():
        while True:
            for result in sequence:
                yield result

    return generator()


@pytest.fixture
def mock_optimizer():
    """A mock OptimizerAbstract instance."""
    mock_ac = Mock(spec=AcceptanceCriterion)
    mock_problem = Mock()
    mock_problem.get_initial_solutions.return_value = []

    mock_opt = Mock(spec=OptimizerAbstract)
    mock_opt.name = "TestOpt"
    mock_opt.version = 1
    mock_opt.problem = mock_problem
    mock_opt.acceptance_criterion = mock_ac
    mock_opt.reset.side_effect = lambda: mock_ac.clear()
    mock_opt.optimize.return_value = [
        None,
        None,
        None,
        False,
        None,
        True,
        False,
        False,
        False,
        True,
        False,
        None,
    ]

    # Need a separate mock for ChainedOptimizer tests, as it relies on specific instance behavior
    class MockVNS(OptimizerAbstract):
        def __init__(self, name="VNS"):
            super().__init__(name, 1, mock_problem)
            self.optimize_calls = 0

        def optimize(self) -> Iterable[Optional[bool]]:
            self.optimize_calls += 1
            # A simple finite stream for chaining
            yield None
            yield False
            yield True
            yield False

        def initialize(self):
            pass

        def reset(self):
            pass

        def get_solutions(self):
            return []

        def add_solutions(self, solutions):
            pass

    return mock_opt, MockVNS


def test_terminate_noop(mock_optimizer_process):
    """Tests that terminate_noop runs the stream unchanged."""
    # Call the fixture to get the generator (iterator)
    process = terminate_noop()(mock_optimizer_process)
    # Use list slicing or zip for cleaner iteration
    results = [res for _, res in zip(range(7), process)]
    assert results == [None, None, None, False, None, True, False]


def test_terminate_time_based_validity():
    """Tests ValueError for non-positive time."""
    with pytest.raises(ValueError):
        # Must call the inner 'optimize' function
        terminate_time_based(0)


@patch("time.time")
def test_terminate_time_based_stops(mock_time):
    """Tests that terminate_time_based stops the generator after the specified time."""
    max_time = 10.0

    # Mock time to control the flow: Start at 100.0, end at 110.1 (total elapsed 10.1 > 10.0)
    mock_time.side_effect = [
        100.0,  # start_time
        105.0,  # Check 1 (yield True)
        109.9,  # Check 2 (yield True)
        110.1,  # Check 3 (Break condition met after yield)
    ]

    # A simple process that yields True indefinitely
    def infinite_process():
        while True:
            yield True

    # Run the termination wrapper
    process = terminate_time_based(max_time)(infinite_process())
    results = list(process)

    # We expect 3 yields before the break.
    assert len(results) == 3
    assert results == [True, True, True]


def test_terminate_max_iterations_validity():
    """Tests ValueError for non-positive iterations."""
    with pytest.raises(ValueError):
        # Must call the inner 'optimize' function
        terminate_max_iterations(0)


def test_terminate_max_iterations_stops_correctly(mock_optimizer_process):
    """Tests that terminate_max_iterations stops exactly after the specified VNS cycles."""

    # Call the fixture to get the generator (iterator)
    process = terminate_max_iterations(3)(mock_optimizer_process)
    results = list(process)

    assert len(results) == 7
    assert results == [None, None, None, False, None, True, False]


def test_terminate_max_no_improvement_stops_correctly(mock_optimizer_process):
    """Tests that terminate_max_no_improvement stops after N consecutive False results."""

    # Call the fixture to get the generator (iterator)
    process = terminate_max_no_improvement(3)(mock_optimizer_process)
    results = list(process)

    assert len(results) == 9
    assert results == [None, None, None, False, None, True, False, False, False]


def test_stoppable_optimizer_wrapping(mock_optimizer):
    """Tests that StoppableOptimizer correctly wraps and uses the inner optimizer's stream."""
    # mock_opt is the Mock object, not the fixture function
    mock_opt, _ = mock_optimizer

    # Use a termination that stops after 3 iterations
    stoppable_opt = StoppableOptimizer(mock_opt, terminate_max_iterations(3))

    # Test optimize method
    results = list(stoppable_opt.optimize())

    # The results should be the truncated stream
    assert len(results) == 7
    assert results == [None, None, None, False, None, True, False]

    # Test delegation of reset
    stoppable_opt.reset()
    mock_opt.reset.assert_called_once()


def test_chained_optimizer_basic_chaining(mock_optimizer):
    """Tests that ChainedOptimizer runs phases sequentially and shares the AC."""
    _, MockVNS = mock_optimizer

    # Setup mocks for the chain
    mock_ac = Mock(spec=AcceptanceCriterion)
    mock_problem = Mock()
    # Ensure problem initialization works
    mock_problem.get_initial_solutions.return_value = [Solution([1]), Solution([2])]

    stoppable_A = StoppableOptimizer(MockVNS(name="A"), terminate_max_iterations(1))
    stoppable_B = StoppableOptimizer(MockVNS(name="B"), terminate_max_iterations(2))

    # The chain itself
    chain = ChainedOptimizer([stoppable_A, stoppable_B])

    chain.reset()
    chain.initialize()

    chain.problem = mock_problem

    mock_ac.clear = Mock()
    mock_ac.accept = Mock()

    results = list(chain.optimize())

    assert len(results) == 5
    assert results == [None, False, None, False, True]
