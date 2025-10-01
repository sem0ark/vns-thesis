import numpy as np
import pytest

from src.examples.moacbw.problem import MOACBWProblem, _MOACBWSolution


@pytest.fixture
def example_problem() -> MOACBWProblem:
    return MOACBWProblem(
        7,
        [
            (0, [1, 2, 6]),
            (1, [0]),
            (2, [0, 5]),
            (3, [4, 5]),
            (4, [3]),
            (5, [2, 3]),
            (6, [0]),
        ],
    )


def test_example_problem(example_problem: MOACBWProblem):
    antibandwidth = 2
    cutwidth = 5
    antibandwidth_obj, cutwidth_obj = _MOACBWSolution(
        np.array([0, 3, 2, 6, 5, 1, 4]), example_problem
    ).objectives

    assert antibandwidth_obj < 0
    assert int(-antibandwidth_obj) == antibandwidth

    assert cutwidth_obj > 0
    assert int(cutwidth_obj) == cutwidth


def test_example_problem_precise(example_problem: MOACBWProblem):
    antibandwidth = 2 + sum([2, 3, 2, 3, 2, 5, 5]) / 49
    cutwidth = 5 + sum([3, 5, 5, 4, 2, 1, 0]) / 49
    antibandwidth_obj, cutwidth_obj = _MOACBWSolution(
        np.array([0, 3, 2, 6, 5, 1, 4]), example_problem
    ).objectives

    assert antibandwidth_obj < 0
    assert -antibandwidth_obj == pytest.approx(antibandwidth)

    assert cutwidth_obj > 0
    assert cutwidth_obj == pytest.approx(cutwidth)


def test_solution_distance_different(example_problem: MOACBWProblem):
    sol1 = _MOACBWSolution(np.array([0, 3, 2, 6, 5, 1, 4]), example_problem)
    sol2 = _MOACBWSolution(np.array([4, 1, 5, 2, 6, 3, 0]), example_problem)
    assert example_problem.calculate_solution_distance(sol1, sol2) == pytest.approx(1.0)


def test_solution_distance_same(example_problem: MOACBWProblem):
    sol1 = _MOACBWSolution(np.array([0, 3, 2, 6, 5, 1, 4]), example_problem)
    assert example_problem.calculate_solution_distance(sol1, sol1) == pytest.approx(0.0)


def test_solution_distance_partial(example_problem: MOACBWProblem):
    sol1 = _MOACBWSolution(np.array([0, 3, 2, 6, 5, 1, 4]), example_problem)
    sol2 = _MOACBWSolution(np.array([0, 3, 2, 6, 4, 1, 5]), example_problem)
    assert example_problem.calculate_solution_distance(sol1, sol2) == pytest.approx(
        2 / 7
    )
