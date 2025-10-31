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
from src.problems.moacbw.vns import shake_swap, swap_op
from src.vns.acceptance import AcceptBatch, AcceptBeam
from src.vns.local_search import (
    best_improvement,
    composite,
    composite_parallel,
    noop,
)
from src.vns.optimizer import VNSOptimizer
from src.vns_extensions.skewed_vns import (
    AcceptBatchSkewedV1,
    AcceptBatchSkewedV2,
    AcceptBatchSkewedV3,
    AcceptBatchSkewedV4,
    AcceptBatchSkewedV5,
    AcceptBatchSkewedV6,
    AcceptBatchSkewedV7,
    AcceptBatchSkewedV8,
)
from src.vns_extensions.variable_formulation_vns import AcceptBatchVFS


class VNSInstanceRunner(InstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.problem = MOACBWProblem.load(str(config.instance_path))

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        return []

    def make_func(self, optimizer: OptimizerAbstract):
        def run(config: RunConfig):
            solutions = run_vns_optimizer(
                optimizer, terminate_time_based(config.run_time_seconds)
            )

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=optimizer.name,
                    version=18,
                    problem_name="moacbw",
                    instance_name=config.instance_path.stem,
                ),
                solutions=[
                    SavedSolution(sol.objectives, sol.to_json_serializable())
                    for sol in solutions
                ],
            )

        return run


class StandardVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        acceptance_criteria = [
            ("beam", AcceptBeam()),
            ("batch", AcceptBatch()),
        ]

        search_functions = [
            ("noop shake_swap", noop(), shake_swap),
            ("BI op_swap shake_swap", best_improvement(swap_op), shake_swap),
            (
                "BI_1 op_swap shake_swap",
                best_improvement(swap_op, max_transitions=1),
                shake_swap,
            ),
            (
                "BI_2 op_swap shake_swap",
                best_improvement(swap_op, max_transitions=2),
                shake_swap,
            ),
        ]

        for (
            (acc_name, acceptance_criterion),
            (search_name, search_func, shake_func),
            k,
        ) in itertools.product(acceptance_criteria, search_functions, range(1, 8)):
            common_name = "BVNS"
            if "composite_" in search_name:
                common_name = "GVNS"
            elif "noop" in search_name:
                common_name = "RVNS"

            config_name = f"vns {common_name} {acc_name} {search_name} k{k}"

            optimizer = VNSOptimizer(
                problem=self.problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
            )

            yield config_name, self.make_func(optimizer)


class VFSVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        acceptance_criteria = [
            (
                "vfs vfs_max_sum",
                AcceptBatchVFS(
                    self.problem.calculate_objectives_max,
                    self.problem.calculate_objectives_sum,
                ),
            ),
            (
                "vfs vfs_sum_max",
                AcceptBatchVFS(
                    self.problem.calculate_objectives_max,
                    self.problem.calculate_objectives_sum,
                ),
            ),
        ]

        search_functions = [
            ("noop shake_swap", noop(), shake_swap),
            ("BI op_swap shake_swap", best_improvement(swap_op), shake_swap),
            (
                "BI_1 op_swap shake_swap",
                best_improvement(swap_op, max_transitions=1),
                shake_swap,
            ),
            (
                "BI_2 op_swap shake_swap",
                best_improvement(swap_op, max_transitions=2),
                shake_swap,
            ),
        ]

        for (
            (acc_name, acceptance_criterion),
            (search_name, search_func, shake_func),
            k,
        ) in itertools.product(acceptance_criteria, search_functions, range(1, 8)):
            common_name = "BVNS"
            if "composite_" in search_name:
                common_name = "GVNS"
            elif "noop" in search_name:
                common_name = "RVNS"

            config_name = f"vns {common_name} {acc_name} {search_name} k{k}"

            optimizer = VNSOptimizer(
                problem=self.problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
            )

            yield config_name, self.make_func(optimizer)


class SVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        alpha_weights = [5, 5]

        alpha_range = [0.25, 0.5, 1, 2, 3]
        dist_func = MOACBWProblem.calculate_solution_distance
        acceptance_criteria = [
            (
                f"skewed_v{acc_idx} {name} a{mult}",
                acc(alpha_weights * mult, dist_func),
            )
            for mult in alpha_range
            for acc_idx, (name, acc) in enumerate(
                [
                    ("wrapped skewed_direct_compare", AcceptBatchSkewedV1),
                    ("wrapped skewed_direct_compare keep_skewed", AcceptBatchSkewedV2),
                    ("wrapped skewed_avg_compare", AcceptBatchSkewedV3),
                    ("wrapped skewed_min_compare", AcceptBatchSkewedV4),
                    ("shallow skewed_min_compare", AcceptBatchSkewedV5),
                    ("shallow skewed_min_compare keep_skewed", AcceptBatchSkewedV6),
                    ("shallow skewed_direct_compare keep_skewed", AcceptBatchSkewedV7),
                    ("shallow skewed_direct_compare", AcceptBatchSkewedV8),
                ],
                start=1,
            )
        ]

        search_functions = [
            ("noop shake_swap", noop(), shake_swap),
            ("BI op_swap shake_swap", best_improvement(swap_op), shake_swap),
            (
                "BI_1 op_swap shake_swap",
                best_improvement(swap_op, max_transitions=1),
                shake_swap,
            ),
            (
                "BI_2 op_swap shake_swap",
                best_improvement(swap_op, max_transitions=2),
                shake_swap,
            ),
        ]

        for (
            (acc_name, acceptance_criterion),
            (search_name, search_func, shake_func),
            k,
        ) in itertools.product(acceptance_criteria, search_functions, range(1, 8)):
            config_name = f"vns SVNS {acc_name} {search_name} k{k}"

            optimizer = VNSOptimizer(
                problem=self.problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
            )

            yield config_name, self.make_func(optimizer)


class MOVND_VNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        acceptance_criteria = [
            ("batch", AcceptBatch()),
        ]

        search_functions = [
            (
                "MOVND_BI op_swap shake_swap",
                composite(
                    [
                        best_improvement(swap_op, objective_index=i)
                        for i in range(self.problem.num_objectives)
                    ]
                ),
                shake_swap,
            ),
            (
                "parallel_BI op_flip shake_flip",
                composite_parallel(
                    [
                        best_improvement(swap_op, objective_index=i)
                        for i in range(self.problem.num_objectives)
                    ]
                ),
                shake_swap,
            ),
        ]

        for (
            (acc_name, acceptance_criterion),
            (search_name, search_func, shake_func),
            k,
        ) in itertools.product(acceptance_criteria, search_functions, range(1, 8)):
            config_name = f"vns MOVNS {acc_name} {search_name} k{k}"

            optimizer = VNSOptimizer(
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
                problem=self.problem,
            )

            yield config_name, self.make_func(optimizer)


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
                    problem_name=self.problem.problem_instance.problem_name,
                    instance_name=Path(config.instance_path).stem,
                ),
                solutions=solutions_data,
            )

        return run


if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    CLI("moacbw", base, [VNSInstanceRunner, PymooInstanceRunner], MOACBWProblem).run()
