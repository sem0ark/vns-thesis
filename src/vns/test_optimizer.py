from unittest.mock import MagicMock
from src.vns.optimizer import ElementwiseVNSOptimizer


class Solution:
    def __init__(self, data_id: int):
        self.data_id = data_id
    def __repr__(self): return f"S{self.data_id}"
    def __eq__(self, other): return self.data_id == other.data_id
    def __hash__(self): return hash(self.data_id)


def test_vns_initialization():
    """Test initial setup: clearing criterion and adding initial solutions."""
    
    optimizer = ElementwiseVNSOptimizer(
        name="VNS",
        version=1,
        problem=MagicMock(),
        search_functions=[lambda *args: [] for _ in range(3)],
        shake_function=lambda sol, k, vns: Solution(data_id=sol.data_id + k),
        acceptance_criterion=MagicMock(),
    )
    optimizer.problem.get_initial_solutions.return_value = [Solution(data_id=100)]
    optimizer.acceptance_criterion.get_one_current_solution.return_value = Solution(data_id=100)

    for _ in zip(range(100), optimizer.optimize()):
        continue
    
    optimizer.acceptance_criterion.clear.assert_called_once()
    optimizer.problem.get_initial_solutions.assert_called_once()

    optimizer.acceptance_criterion.accept.assert_called_once_with(Solution(data_id=100))
    print(optimizer.acceptance_criterion.accept.call_args)
