import itertools
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.cli.problem_cli import CLI, InstanceRunner, RunConfig
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.cli.vns_runner import run_vns_optimizer
from src.core.abstract import OptimizerAbstract
from src.core.termination import terminate_time_based
from src.problems.moacbw.problem import MOACBWProblem, MOACBWProblemPymoo
from src.problems.moacbw.vns import shake_swap, swap_limited_op, swap_op
from src.vns.acceptance import AcceptBatch
from src.vns.local_search import (
    best_improvement,
    first_improvement,
    noop,
)
from src.vns.optimizer import VNSOptimizer
from src.vns_extensions.skewed_vns import (
    AcceptBatchSkewedV1,
    AcceptBatchSkewedV2,
    AcceptBatchSkewedV3,
    AcceptBatchSkewedV4,
)
from src.vns_extensions.variable_formulation_vns import (
    AcceptBatchVFS,
)


class VNSInstanceRunner(InstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.problem = MOACBWProblem.load(str(config.instance_path))

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        acceptance_criteria = [
            ("batch", AcceptBatch()),
            (
                "skewed_v1 a5",
                AcceptBatchSkewedV1(
                    [5.0, 5.0], self.problem.calculate_solution_distance
                ),
            ),
            (
                "skewed_v2 a5",
                AcceptBatchSkewedV2(
                    [5.0, 5.0], self.problem.calculate_solution_distance
                ),
            ),
            (
                "skewed_v3 a5",
                AcceptBatchSkewedV3(
                    [5.0, 5.0], self.problem.calculate_solution_distance
                ),
            ),
            (
                "skewed_v4 a5",
                AcceptBatchSkewedV4(
                    [5.0, 5.0], self.problem.calculate_solution_distance
                ),
            ),
            (
                "vfs",
                AcceptBatchVFS(
                    self.problem.calculate_objectives_max,
                    self.problem.calculate_objectives_sum,
                ),
            ),
        ]

        local_search_functions = [
            ("noop", noop()),
            *[
                (f"{search_name} {op_name}", search_func_factory(op_func))
                for (
                    (search_name, search_func_factory),
                    (op_name, op_func),
                ) in itertools.product(
                    [
                        ("BI", best_improvement),
                        ("FI", first_improvement),
                    ],
                    [("op_swap", swap_op), ("op_short_swap", swap_limited_op)],
                )
            ],
        ]
        shake_functions = [
            ("shake_swap", shake_swap),
        ]

        for (
            (acc_name, acc_criteria),
            (search_name, search_func),
            (shake_name, shake_func),
            k,
        ) in itertools.product(
            acceptance_criteria, local_search_functions, shake_functions, range(1, 8)
        ):
            common_name = "BVNS"
            if "composite_" in search_name:
                common_name = "GVNS"
            elif "noop" in search_name:
                common_name = "RVNS"
            elif "skewed" in acc_name:
                common_name = "SVNS"
            elif "vfs" in acc_name:
                common_name = "VFS"

            config_name = (
                f"vns {common_name} {acc_name} {search_name} k{k} {shake_name}"
            )

            optimizer = VNSOptimizer(
                problem=self.problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acc_criteria,
                shake_function=shake_func,
                name=config_name,
                version=18,
            )

            yield config_name, self.make_func(optimizer)

    def make_func(self, optimizer: OptimizerAbstract):
        def run(config: RunConfig):
            solutions = run_vns_optimizer(
                optimizer, terminate_time_based(config.run_time_seconds)
            )

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=optimizer.name,
                    version=optimizer.version,
                    problem_name="moacbw",
                    instance_name=config.instance_path.stem,
                ),
                solutions=[
                    SavedSolution(sol.objectives, sol.to_json_serializable())
                    for sol in solutions
                ],
            )

        return run


class PymooInstanceRunner(InstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.problem = MOACBWProblemPymoo(MOACBWProblem.load(str(config.instance_path)))

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        algorithms = [
            ("NSGA2", NSGA2),
            ("SPEA2", SPEA2),
        ]
        population_sizes = [50, 100, 150, 200, 300, 400, 500, 700, 1000]

        for (
            (algorithm_name, algorithm),
            population,
        ) in product(algorithms, population_sizes):
            name = f"pymoo {algorithm_name} pop_{population}"
            yield (
                name,
                self.make_func(
                    name,
                    algorithm(
                        sampling=PermutationRandomSampling(),
                        eliminate_duplicates=True,
                        pop_size=population,
                    ),
                ),
            )

    def make_func(self, name: str, algorithm: NSGA2 | SPEA2):
        def run(config: RunConfig):
            res = minimize(
                problem=self.problem,
                algorithm=algorithm,
                termination=TimeBasedTermination(config.run_time_seconds),
                verbose=True,
            )

            results = res.F
            if results is not None:
                nd_sorting = NonDominatedSorting()
                non_dominated_indices = nd_sorting.do(
                    results, only_non_dominated_front=True
                )

                final_solutions_F = results[non_dominated_indices]
            else:
                final_solutions_F = np.array([])

            solutions_data = [
                SavedSolution(cast(np.ndarray, sol).tolist())
                for sol in final_solutions_F
            ]

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=name,
                    version=5,
                    problem_name="moacbw",
                    instance_name=Path(config.instance_path).stem,
                ),
                solutions=solutions_data,
            )

        return run


if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    CLI("moacbw", base, [VNSInstanceRunner, PymooInstanceRunner], MOACBWProblem).run()
