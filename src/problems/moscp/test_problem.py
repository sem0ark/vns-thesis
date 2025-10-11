import numpy as np
import pytest
from src.problems.moscp.problem import MOSCPProblem, _MOSCPSolution


@pytest.fixture
def moscp_example_problem() -> MOSCPProblem:
    """
    Example MO-SCP instance:
    - Items (U): {1, 2, 3} (num_items = 3)
    - Sets (S): {S1, S2, S3, S4} (num_sets = 4)
    - Objectives: Z1 (Cost 1), Z2 (Cost 2)

    Set Definitions (0-based sets, 1-based items in coverage_data for loading):
    | Set | Covers Items (1-based) | Cost Z1 | Cost Z2 |
    |-----|------------------------|---------|---------|
    | S1  | {1, 2}                 | 10      | 1       |
    | S2  | {2, 3}                 | 5       | 2       |
    | S3  | {1}                    | 3       | 8       |
    | S4  | {3}                    | 20      | 1       |

    Coverage Data (index is item_idx, value is list of covering set indices 1-based):
    Item 0 (1): Covered by {S1, S3}
    Item 1 (2): Covered by {S1, S2}
    Item 2 (3): Covered by {S2, S4}
    """

    num_items = 3
    num_sets = 4

    # coverage_data: list of lists, where index i is item i, and value is list of covering sets (1-based)
    coverage_data = [
        [1, 3],  # Item 1 (0-idx) covered by S1, S3
        [1, 2],  # Item 2 (1-idx) covered by S1, S2
        [2, 4],  # Item 3 (2-idx) covered by S2, S4
    ]

    # costs: list of costs per objective, per set (index i is objective i)
    # costs[0] = [Cost Z1 for S1, S2, S3, S4]
    # costs[1] = [Cost Z2 for S1, S2, S3, S4]
    costs = [
        [10, 5, 3, 20],  # Costs for Z1
        [1, 2, 8, 1],  # Costs for Z2
    ]

    return MOSCPProblem(coverage_data, costs, num_items, num_sets)


def test_moscp_problem_initialization(moscp_example_problem: MOSCPProblem):
    """Test the dimensions and internal structures are set up correctly."""
    problem = moscp_example_problem
    assert problem.num_variables == 4  # num_sets
    assert problem.num_objectives == 2
    assert problem.num_items == 3

    # Check unpacked coverage matrix shape (num_items x num_sets)
    assert problem.coverage_unpacked.shape == (3, 4)
    # Check costs shape (num_objectives x num_sets)
    assert problem.costs.shape == (2, 4)

    # Check a specific cost: Cost Z2 for S3 (set index 2) should be 8
    assert problem.costs[1, 2] == 8
    # Check coverage of Item 1 (idx 0): Should be covered by S1 (0) and S3 (2)
    assert problem.coverage_unpacked[0].tolist() == [1, 0, 1, 0]


def test_moscp_solution_feasible(moscp_example_problem: MOSCPProblem):
    """Test a known feasible solution (e.g., S1 + S2 covers all: {1,2} U {2,3} = {1,2,3})."""
    # Selection: S1=1, S2=1, S3=0, S4=0
    selection = [1, 1, 0, 0]
    sol_data = np.array(selection).astype(bool)

    # Costs:
    # Z1: S1(10) + S2(5) = 15
    # Z2: S1(1) + S2(2) = 3
    expected_objectives = (15.0, 3.0)

    solution = _MOSCPSolution(sol_data, moscp_example_problem)

    assert moscp_example_problem.satisfies_constraints(sol_data) is True
    assert solution.objectives == pytest.approx(expected_objectives)


def test_moscp_solution_infeasible(moscp_example_problem: MOSCPProblem):
    """Test an infeasible solution (e.g., S1 covers {1, 2}, misses {3})."""
    # Selection: S1=1, S2=0, S3=0, S4=0 (Item 3 is not covered)
    selection = [1, 0, 0, 0]
    sol_data = np.array(selection).astype(bool)

    # Infeasible solutions are penalized by setting cost to x100
    expected_objectives = (1000.0, 100.0)

    solution = _MOSCPSolution(sol_data, moscp_example_problem)

    assert moscp_example_problem.satisfies_constraints(sol_data) is False
    assert solution.objectives == pytest.approx(expected_objectives)


def test_moscp_solution_distance_different(moscp_example_problem: MOSCPProblem):
    """Test distance calculation (Hamming distance on unpacked data)."""
    # Sol1: [1, 1, 0, 0]
    sol1_data = np.array([1, 1, 0, 0])
    sol1 = _MOSCPSolution(sol1_data, moscp_example_problem)

    # Sol2: [0, 1, 1, 0] (S1 vs S3, two differences)
    sol2_data = np.array([0, 1, 1, 0])
    sol2 = _MOSCPSolution(sol2_data, moscp_example_problem)

    # Differences at indices 0 (1 vs 0) and 2 (0 vs 1). Total size is 4.
    # Distance: 2 / 4 = 0.5
    assert moscp_example_problem.calculate_solution_distance(
        sol1, sol2
    ) == pytest.approx(0.5)


def test_moscp_solution_distance_same(moscp_example_problem: MOSCPProblem):
    """Test distance calculation for identical solutions."""
    sol1_data = np.array([1, 0, 1, 0])
    sol1 = _MOSCPSolution(sol1_data, moscp_example_problem)
    assert moscp_example_problem.calculate_solution_distance(
        sol1, sol1
    ) == pytest.approx(0.0)


def test_moscp_solution_serialization(moscp_example_problem: MOSCPProblem):
    """Test packing and unpacking via serialization methods."""
    # A known selection vector
    unpacked_selection = [1, 1, 0, 0]
    sol_data_packed = np.array(unpacked_selection)

    sol = _MOSCPSolution(sol_data_packed, moscp_example_problem)

    # Serialization
    serialized = sol.to_json_serializable()
    assert serialized == unpacked_selection

    # Deserialization (uses from_json_serializable)
    reloaded_sol = _MOSCPSolution.from_json_serializable(
        moscp_example_problem, serialized
    )

    # Check that the reloaded packed data matches the original packed data
    assert np.array_equal(reloaded_sol.data, sol.data)
    # Check that the objectives are the same
    assert reloaded_sol.objectives == pytest.approx(sol.objectives)
