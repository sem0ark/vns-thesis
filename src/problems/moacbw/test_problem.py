import numpy as np
import pytest

from src.problems.moacbw.problem import MOACBWProblem, _MOACBWSolution


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
    antibandwidth_obj, cutwidth_obj = _MOACBWSolution(
        np.array([0, 3, 2, 6, 5, 1, 4]), example_problem
    ).objectives

    assert antibandwidth_obj < 0
    assert -antibandwidth_obj == 2

    assert cutwidth_obj > 0
    assert cutwidth_obj == 5


def test_example_problem_precise(example_problem: MOACBWProblem):
    solution_data = np.array([0, 3, 2, 6, 5, 1, 4])

    assert example_problem.get_antibandwidth_values(solution_data) == [
        2,
        3,
        2,
        3,
        2,
        5,
        5,
    ]
    assert example_problem.get_cutwidth_values(solution_data) == [3, 5, 5, 4, 2, 1]


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


@pytest.fixture
def path_graph_fixture():
    """A simple 4-node path graph: 0-1-2-3"""
    return MOACBWProblem(
        4,
        [
            (0, [1]),
            (1, [0, 2]),
            (2, [1, 3]),
            (3, [2]),
        ],
    )


@pytest.fixture
def star_graph_fixture():
    """A 5-node star graph with center 0: 0 connects to 1, 2, 3, 4."""
    return MOACBWProblem(
        5,
        [
            (0, [1, 2, 3, 4]),
            (1, [0]),
            (2, [0]),
            (3, [0]),
            (4, [0]),
        ],
    )


@pytest.fixture
def disconnected_graph_fixture():
    """A 4-node graph with one edge 0-1 and two isolated nodes 2, 3."""
    return MOACBWProblem(
        4,
        [
            (0, [1]),
            (1, [0]),
            (2, []),
            (3, []),
        ],
    )


def test_cutwidth_path_graph_perfect_ordering(path_graph_fixture):
    """
    Test the Path Graph 0-1-2-3 with the canonical ordering [0, 1, 2, 3].
    Expected cuts:
    C(1): Cut after 0: L={0}, R={1,2,3}. Edge (0,1) crosses. Cut = 1.
    C(2): Cut after 1: L={0,1}, R={2,3}. Edge (1,2) crosses. Cut = 1.
    C(3): Cut after 2: L={0,1,2}, R={3}. Edge (2,3) crosses. Cut = 1.
    Expected result: [1, 1, 1]
    """
    solution_data = np.array([0, 1, 2, 3])
    expected = [1, 1, 1]

    assert path_graph_fixture.get_cutwidth_values(solution_data) == expected


def test_cutwidth_path_graph_bad_ordering(path_graph_fixture):
    """
    Test the Path Graph 0-1-2-3 with a non-canonical ordering [0, 2, 1, 3].
    Expected cuts:
    C(1): Cut after 0: L={0}, R={2,1,3}. Edges (0,1) crosses. Cut = 1.
    C(2): Cut after 2: L={0,2}, R={1,3}. Edges (0,1), (2,1), (2,3) cross. Cut = 3.
    C(3): Cut after 1: L={0,2,1}, R={3}. Edge (2,3) crosses. Cut = 1.
    Expected result: [1, 3, 1]
    """
    solution_data = np.array([0, 2, 1, 3])
    expected = [1, 3, 1]

    assert path_graph_fixture.get_cutwidth_values(solution_data) == expected


def test_cutwidth_star_graph_optimal_ordering(star_graph_fixture):
    """
    Test the Star Graph 0-1, 0-2, 0-3, 0-4 with center 0 placed last: [1, 2, 3, 4, 0].
    Expected cuts (N=5):
    C(1): Cut after 1: L={1}, R={2,3,4,0}. Edge (1,0) crosses. Cut = 1.
    C(2): Cut after 2: L={1,2}, R={3,4,0}. Edges (1,0), (2,0) cross. Cut = 2.
    C(3): Cut after 3: L={1,2,3}, R={4,0}. Edges (1,0), (2,0), (3,0) cross. Cut = 3.
    C(4): Cut after 4: L={1,2,3,4}, R={0}. Edges (1,0)...(4,0) cross. Cut = 4.
    Expected result: [1, 2, 3, 4]
    """
    solution_data = np.array([1, 2, 3, 4, 0])
    expected = [1, 2, 3, 4]

    assert star_graph_fixture.get_cutwidth_values(solution_data) == expected


def test_cutwidth_disconnected_graph(disconnected_graph_fixture):
    """
    Test the Disconnected Graph with ordering [2, 3, 0, 1].
    Edges: (0, 1). Nodes 2, 3 are isolated.
    Cuts (N=4):
    C(1): Cut after 2: L={2}, R={3,0,1}. No edges cross. Cut = 0.
    C(2): Cut after 3: L={2,3}, R={0,1}. No edges cross. Cut = 0.
    C(3): Cut after 0: L={2,3,0}, R={1}. Edge (0,1) crosses. Cut = 1.
    Expected result: [0, 0, 1]
    """
    solution_data = np.array([2, 3, 0, 1])
    expected = [0, 0, 1]

    assert disconnected_graph_fixture.get_cutwidth_values(solution_data) == expected


def test_antibandwidth_path_graph_perfect_ordering(path_graph_fixture):
    """
    Test the Path Graph 0-1-2-3 with the canonical ordering [0, 1, 2, 3].
    The output should be ordered by position: [AB(0), AB(1), AB(2), AB(3)].

    Permutation pi: 0->0, 1->1, 2->2, 3->3.
    Node 0: Neighbors {1}. AB(0) = |pi(0) - pi(1)| = |0 - 1| = 1.
    Node 1: Neighbors {0, 2}. AB(1) = min(|1-0|, |1-2|) = 1.
    Node 2: Neighbors {1, 3}. AB(2) = min(|2-1|, |2-3|) = 1.
    Node 3: Neighbors {2}. AB(3) = |3 - 2| = 1.

    Output by position: [AB(0), AB(1), AB(2), AB(3)]
    Expected result: [1, 1, 1, 1]
    """
    solution_data = np.array([0, 1, 2, 3])
    expected = [1, 1, 1, 1]

    assert path_graph_fixture.get_antibandwidth_values(solution_data) == expected


def test_antibandwidth_path_graph_reversal_ordering(path_graph_fixture):
    """
    Test the Path Graph 0-1-2-3 with the reverse ordering [3, 2, 1, 0].

    Permutation pi: 3->0, 2->1, 1->2, 0->3.
    AB values by Node ID:
    Node 0: Neighbors {1}. AB(0) = |pi(0) - pi(1)| = |3 - 2| = 1.
    Node 1: Neighbors {0, 2}. AB(1) = min(|2-3|, |2-1|) = 1.
    Node 2: Neighbors {1, 3}. AB(2) = min(|1-2|, |1-0|) = 1.
    Node 3: Neighbors {2}. AB(3) = |0 - 1| = 1.

    Output by position (Node ID at position i):
    Pos 0 (Node 3): AB(3) = 1
    Pos 1 (Node 2): AB(2) = 1
    Pos 2 (Node 1): AB(1) = 1
    Pos 3 (Node 0): AB(0) = 1
    Expected result: [1, 1, 1, 1]
    """
    solution_data = np.array([3, 2, 1, 0])
    expected = [1, 1, 1, 1]

    assert path_graph_fixture.get_antibandwidth_values(solution_data) == expected


def test_antibandwidth_star_graph_center_first(star_graph_fixture):
    """
    Test the Star Graph (center 0) with ordering [0, 1, 2, 3, 4].

    Permutation pi: 0->0, 1->1, 2->2, 3->3, 4->4.
    AB values by Node ID:
    Node 0 (Center): N(0)={1,2,3,4}. AB(0) = min(|0-1|, |0-2|, |0-3|, |0-4|) = 1.
    Node 1 (Leaf): N(1)={0}. AB(1) = |1 - 0| = 1.
    Node 2 (Leaf): N(2)={0}. AB(2) = |2 - 0| = 2.
    Node 3 (Leaf): N(3)={0}. AB(3) = |3 - 0| = 3.
    Node 4 (Leaf): N(4)={0}. AB(4) = |4 - 0| = 4.

    Output by position (Node ID at position i):
    Pos 0 (Node 0): AB(0) = 1
    Pos 1 (Node 1): AB(1) = 1
    Pos 2 (Node 2): AB(2) = 2
    Pos 3 (Node 3): AB(3) = 3
    Pos 4 (Node 4): AB(4) = 4
    Expected result: [1, 1, 2, 3, 4]
    """
    solution_data = np.array([0, 1, 2, 3, 4])
    expected = [1, 1, 2, 3, 4]

    assert star_graph_fixture.get_antibandwidth_values(solution_data) == expected


def test_antibandwidth_star_graph_center_last(star_graph_fixture):
    """
    Test the Star Graph (center 0) with ordering [1, 2, 3, 4, 0].

    Permutation pi: 1->0, 2->1, 3->2, 4->3, 0->4.
    AB values by Node ID:
    Node 0 (Center): N(0)={1,2,3,4}. AB(0) = min(|4-0|, |4-1|, |4-2|, |4-3|) = 1.
    Node 1 (Leaf): N(1)={0}. AB(1) = |0 - 4| = 4.
    Node 2 (Leaf): N(2)={0}. AB(2) = |1 - 4| = 3.
    Node 3 (Leaf): N(3)={0}. AB(3) = |2 - 4| = 2.
    Node 4 (Leaf): N(4)={0}. AB(4) = |3 - 4| = 1.

    Output by position (Node ID at position i):
    Pos 0 (Node 1): AB(1) = 4
    Pos 1 (Node 2): AB(2) = 3
    Pos 2 (Node 3): AB(3) = 2
    Pos 3 (Node 4): AB(4) = 1
    Pos 4 (Node 0): AB(0) = 1
    Expected result: [4, 3, 2, 1, 1]
    """
    solution_data = np.array([1, 2, 3, 4, 0])
    expected = [4, 3, 2, 1, 1]

    assert star_graph_fixture.get_antibandwidth_values(solution_data) == expected


def test_antibandwidth_disconnected_graph(disconnected_graph_fixture):
    """
    Test the Disconnected Graph with ordering [2, 3, 0, 1].
    Edges: (0, 1). Nodes 2, 3 are isolated. AB of isolated nodes is usually 0
    or excluded, but here we assume the corrected code initializes isolated node AB to 0.

    Permutation pi: 2->0, 3->1, 0->2, 1->3.
    AB values by Node ID:
    Node 0: Neighbors {1}. AB(0) = |pi(0) - pi(1)| = |2 - 3| = 1.
    Node 1: Neighbors {0}. AB(1) = |pi(1) - pi(0)| = |3 - 2| = 1.
    Node 2: Neighbors {}. AB(2) = 0 (due to initialization/continue logic).
    Node 3: Neighbors {}. AB(3) = 0.

    Output by position (Node ID at position i):
    Pos 0 (Node 2): AB(2) = 0
    Pos 1 (Node 3): AB(3) = 0
    Pos 2 (Node 0): AB(0) = 1
    Pos 3 (Node 1): AB(1) = 1
    Expected result: [0, 0, 1, 1]
    """
    solution_data = np.array([2, 3, 0, 1])
    expected = [0, 0, 1, 1]

    assert (
        disconnected_graph_fixture.get_antibandwidth_values(solution_data) == expected
    )
