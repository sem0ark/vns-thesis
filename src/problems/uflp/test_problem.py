import numpy as np
import pytest

from src.problems.uflp.problem import MOUFLPProblem, _MOUFLPSolution

NUM_C = 3
NUM_F = 4
FIXED_COSTS_DATA = [[100, 200, 50, 150], [10, 20, 5, 15]]
CUSTOMER_COSTS_DATA = [
    [[10, 20, 30, 40], [50, 10, 80, 20], [5, 5, 5, 100]],
    [[2, 4, 6, 8], [10, 2, 16, 4], [1, 1, 1, 20]],
]


@pytest.fixture
def uflp_instance() -> MOUFLPProblem:
    return MOUFLPProblem(
        num_customers=NUM_C,
        num_facilities=NUM_F,
        fixed_costs=FIXED_COSTS_DATA,
        customer_costs=CUSTOMER_COSTS_DATA,
    )


def test_initialization_and_objective_count(uflp_instance):
    """Test if the problem initializes correctly and objective count is right."""
    assert uflp_instance.num_fixed_obj == 2
    assert uflp_instance.num_assignment_obj == 2
    assert uflp_instance.num_objectives == 4
    assert uflp_instance.num_variables == NUM_F
    assert uflp_instance.num_customers == NUM_C
    assert uflp_instance.fixed_costs.shape == (2, 4)
    assert uflp_instance.assignment_costs.shape == (2, 3, 4)


def test_satisfies_constraints():
    """Test the feasibility check (must have at least one facility open)."""
    problem = MOUFLPProblem(3, 4, FIXED_COSTS_DATA, CUSTOMER_COSTS_DATA)

    assert problem.satisfies_constraints(np.array([True, False, False, False]))
    assert problem.satisfies_constraints(np.array([True, False, True, False]))
    assert problem.satisfies_constraints(np.array([True, True, True, True]))
    assert not problem.satisfies_constraints(np.array([False, False, False, False]))


def test_objective_calculation_feasible_solution_f1_f2_open(uflp_instance):
    """Test objective calculation for solution [1, 1, 0, 0] (F1 and F2 open)."""
    solution_data = np.array([True, True, False, False])

    # --- Expected Fixed Costs ---
    # F1 open: 100 + 200 = 300
    # F2 open: 10 + 20 = 30

    # --- Expected Assignment Costs ---
    # F1 & F2 assignment costs (C x 2 matrix):
    # C1: [10, 20] -> Min =10
    # C2: [50, 10] -> Min =10
    # C3: [5, 5]   -> Min = 5
    # Total Assign Obj 1: 10 + 10 + 5 = 25

    # F1 & F2 assignment costs for Obj 2:
    # C1: [2, 4] -> Min=2
    # C2: [10, 2] -> Min=2
    # C3: [1, 1] -> Min=1
    # Total Assign Obj 2: 2 + 2 + 1 = 5

    expected_objectives = [300, 30, 25, 5]
    actual_objectives = uflp_instance.calculate_objectives(solution_data)
    assert actual_objectives == pytest.approx(expected_objectives)


def test_objective_calculation_feasible_solution_all_open(uflp_instance):
    """Test objective calculation for solution [1, 1, 1, 1]."""
    solution_data = np.array([True, True, True, True])

    expected_objectives = [500, 50, 25, 5]
    actual_objectives = uflp_instance.calculate_objectives(solution_data)
    assert actual_objectives == pytest.approx(expected_objectives)


def test_objective_calculation_feasible_solution_f3_open(uflp_instance):
    """Test objective calculation for solution [0, 0, 1, 0] (F3 open)."""
    solution_data = np.array([False, False, True, False])

    # --- Expected Fixed Costs ---
    # F1 open: 50
    # F2 open: 5
    expected_fixed = [50.0, 5.0]

    # --- Expected Assignment Costs ---
    # F3 assignment costs (C x 1 matrix):
    # C1: [30] -> Min=30
    # C2: [80] -> Min=80
    # C3: [5]  -> Min=5
    # Total Assign Obj 1: 30 + 80 + 5 = 115
    expected_assign_1 = 115.0

    # F3 assignment costs for Obj 2:
    # C1: [6]
    # C2: [16]
    # C3: [1]
    # Total Assign Obj 2: 6 + 16 + 1 = 23
    expected_assign_2 = 23.0

    expected_objectives = expected_fixed + [expected_assign_1, expected_assign_2]

    actual_objectives = uflp_instance.calculate_objectives(solution_data)

    assert len(actual_objectives) == 4
    assert actual_objectives == pytest.approx(expected_objectives)


def test_infeasible_solution_penalty(uflp_instance):
    """Test that solutions with no open facilities are correctly penalized."""
    bad_solution = np.array([False, False, False, False])
    good_solution = np.array([False, True, False, False])
    assert np.all(
        uflp_instance.calculate_objectives(bad_solution)
        > uflp_instance.calculate_objectives(good_solution)
    )


def test_initial_solutions_creation(uflp_instance):
    """Test that initial solutions are created correctly and are feasible."""
    num_solutions_to_create = 5
    solutions = list(uflp_instance.get_initial_solutions(num_solutions_to_create))

    assert len(solutions) == num_solutions_to_create

    for sol in solutions:
        assert isinstance(sol, _MOUFLPSolution)
        assert sol.data.shape == (uflp_instance.num_variables,)
        assert uflp_instance.satisfies_constraints(sol.data)
