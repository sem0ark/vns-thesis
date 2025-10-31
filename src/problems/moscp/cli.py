from itertools import chain, product
from pathlib import Path
from typing import Callable, Iterable, cast

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.max_time import TimeBasedTermination

from src.cli.problem_cli import CLI, InstanceRunner, RunConfig
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.cli.vns_runner import run_vns_optimizer
from src.core.abstract import AcceptanceCriterion, OptimizerAbstract
from src.core.termination import terminate_time_based
from src.problems.default_configurations import (
    get_bvns_variants,
    get_movnd_vns_variants,
    get_rvns_variants,
)
from src.problems.moscp.problem import MOSCPProblem, MOSCPProblemPymoo
from src.problems.moscp.vns import flip_op_v2 as flip_op
from src.problems.moscp.vns import shake_flip
from src.vns.acceptance import AcceptBatch, AcceptBeam
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


class VNSInstanceRunner(InstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.problem = MOSCPProblem.load(str(config.instance_path))

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
                    problem_name="MOSCP",
                    instance_name=config.instance_path.stem,
                ),
                solutions=[
                    SavedSolution(sol.objectives, sol.to_json_serializable())
                    for sol in solutions
                ],
            )

        return run


class SharedVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        for config_name, optimizer in get_rvns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("shake_flip", shake_flip)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_bvns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_flip", flip_op)],
            [("shake_flip", shake_flip)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_movnd_vns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_flip", flip_op)],
            [("shake_flip", shake_flip)],
        ):
            yield config_name, self.make_func(optimizer)


class SVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        alpha_weights = np.mean(self.problem.costs, axis=1)

        alpha_range = [0.25, 0.5, 1, 2, 3]
        dist_func = MOSCPProblem.calculate_solution_distance_2
        acceptance_criteria: list[tuple[str, AcceptanceCriterion]] = [
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

        for config_name, optimizer in chain(
            get_rvns_variants(
                self.problem, acceptance_criteria, [("shake_flip", shake_flip)]
            ),
            get_bvns_variants(
                self.problem,
                acceptance_criteria,
                [("op_flip", flip_op)],
                [("shake_flip", shake_flip)],
            ),
        ):
            config_name = config_name.replace("RVNS", "SVNS").replace("RVNS", "SVNS")
            yield config_name, self.make_func(optimizer)


class PymooInstanceRunner(InstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.problem = MOSCPProblemPymoo(MOSCPProblem.load(str(config.instance_path)))

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
                        sampling=BinaryRandomSampling(),
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
            if results is None:
                raise ValueError("Expected res.F to be non-null")

            solution_data = cast(None | np.ndarray, res.X)
            if solution_data is None:
                raise ValueError("Expected res.X to be non-null")

            # for some reason even with binary sampling, result is still float
            solution_data = np.round(solution_data).astype(bool)

            solutions_data = [
                SavedSolution(
                    cast(np.ndarray, objectives).tolist(),
                    cast(np.ndarray, data).tolist(),
                )
                for objectives, data in zip(results, solution_data)
            ]

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=name,
                    version=5,
                    problem_name="MOSCP",
                    instance_name=Path(config.instance_path).stem,
                ),
                solutions=solutions_data,
            )

        return run


if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    CLI(
        "MOSCP",
        base,
        [
            SharedVNSInstanceRunner,
            SVNSInstanceRunner,
            PymooInstanceRunner,
        ],
        MOSCPProblem,
    ).run()
