import numpy as np
import pytest

from src.problems.mokp.problem import MOKPProblem, _MOKPSolution


@pytest.fixture
def example_problem_1d() -> MOKPProblem:
    return MOKPProblem(weights=[[1, 2, 3, 4, 5]], profits=[[5, 4, 3, 2, 1]], capacity=3)


@pytest.fixture
def example_problem_2d() -> MOKPProblem:
    return MOKPProblem(
        weights=[[1, 2, 3, 4, 5]],
        profits=[[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]],
        capacity=3,
    )


@pytest.fixture
def example_problem_2d2d() -> MOKPProblem:
    return MOKPProblem(
        weights=[[1, 2, 3, 4, 5], [1, 2, 3, 2, 1]],
        profits=[[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]],
        capacity=[3, 3],
    )


@pytest.mark.parametrize(
    ["solution_data", "is_feasible"],
    [
        pytest.param(np.array([1, 0, 0, 0, 0], dtype=int), True, id="Weight: 1 <= 3"),
        pytest.param(np.array([1, 1, 0, 0, 0], dtype=int), True, id="Weight: 3 <= 3"),
        pytest.param(
            np.array([1, 1, 1, 0, 0], dtype=int), False, id="Weight: 6 > 3 (Infeasible)"
        ),
    ],
)
def test_is_feasible_1d(
    example_problem_1d: MOKPProblem, solution_data: np.ndarray, is_feasible: bool
):
    assert example_problem_1d.satisfies_constraints(solution_data) == is_feasible


@pytest.mark.parametrize(
    ["solution_data", "is_feasible"],
    [
        pytest.param(
            np.array([1, 0, 0, 0, 0], dtype=int), True, id="Weights: [1, 1] <= [3, 3]"
        ),
        pytest.param(
            np.array([0, 1, 0, 0, 0], dtype=int), True, id="Weights: [2, 2] <= [3, 3]"
        ),
        pytest.param(
            np.array([1, 1, 1, 0, 0], dtype=int),
            False,
            id="Weights: [6, 6] > [3, 3] (Infeasible)",
        ),
        pytest.param(
            np.array([0, 1, 1, 0, 0], dtype=int),
            False,
            id="Weights: [5, 5] > [3, 3] (Infeasible)",
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 1], dtype=int),
            False,
            id="Weights: [5, 1] > [3, 3] (Infeasible)",
        ),
    ],
)
def test_is_feasible_2d2d(
    example_problem_2d2d: MOKPProblem, solution_data: np.ndarray, is_feasible: bool
):
    assert example_problem_2d2d.satisfies_constraints(solution_data) == is_feasible


@pytest.mark.parametrize(
    ["solution_data", "objectives"],
    [
        pytest.param(np.array([1, 0, 0, 0, 0], dtype=int), (-5.0,), id="Feasible 1"),
        pytest.param(np.array([1, 1, 0, 0, 0], dtype=int), (-9.0,), id="Feasible 2"),
        pytest.param(np.array([1, 1, 1, 0, 0], dtype=int), (12.0,), id="Infeasible"),
    ],
)
def test_evaluate_1d(
    example_problem_1d: MOKPProblem,
    solution_data: np.ndarray,
    objectives: tuple[float, ...],
):
    assert _MOKPSolution(solution_data, example_problem_1d).objectives == pytest.approx(
        objectives
    )


@pytest.mark.parametrize(
    ["solution_data", "objectives"],
    [
        pytest.param(np.array([1, 1, 0, 0, 0], dtype=int), (-9.0, -3.0), id="Feasible"),
        pytest.param(
            np.array([1, 1, 1, 0, 0], dtype=int), (12.0, 6.0), id="Infeasible 1"
        ),
    ],
)
def test_evaluate_2d(
    example_problem_2d: MOKPProblem,
    solution_data: np.ndarray,
    objectives: tuple[float, ...],
):
    assert _MOKPSolution(solution_data, example_problem_2d).objectives == pytest.approx(
        objectives
    )


@pytest.mark.parametrize(
    ["solution_data", "objectives"],
    [
        pytest.param(np.array([0, 1, 0, 0, 0], dtype=int), (-4.0, -2.0), id="Feasible"),
        pytest.param(
            np.array([1, 1, 1, 0, 0], dtype=int), (12.0, 6.0), id="Infeasible 1"
        ),
        pytest.param(
            np.array([0, 1, 1, 0, 0], dtype=int), (7.0, 5.0), id="Infeasible 2"
        ),
    ],
)
def test_evaluate_2d2d(
    example_problem_2d2d: MOKPProblem,
    solution_data: np.ndarray,
    objectives: tuple[float, ...],
):
    assert _MOKPSolution(
        solution_data, example_problem_2d2d
    ).objectives == pytest.approx(objectives)


def test_calculate_solution_distance(example_problem_1d: MOKPProblem):
    problem = example_problem_1d
    N = 5  # Number of items

    sol1_data = np.array([1, 1, 0, 0, 0], dtype=int)
    sol2_data = np.array([0, 1, 0, 0, 0], dtype=int)  # 1 difference (item 0)
    sol3_data = np.array([0, 0, 1, 1, 1], dtype=int)  # 5 differences

    sol1 = _MOKPSolution(sol1_data, problem)
    sol2 = _MOKPSolution(sol2_data, problem)
    sol3 = _MOKPSolution(sol3_data, problem)

    # Identical solutions: 0 differences
    assert MOKPProblem.calculate_solution_distance(sol1, sol1) == pytest.approx(0.0)

    # One difference: 1/5 = 0.2
    assert MOKPProblem.calculate_solution_distance(sol1, sol2) == pytest.approx(1 / N)
    assert MOKPProblem.calculate_solution_distance(sol2, sol1) == pytest.approx(1 / N)

    # All differences: 5/5 = 1.0
    assert MOKPProblem.calculate_solution_distance(sol1, sol3) == pytest.approx(5 / N)
