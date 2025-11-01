from itertools import chain
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from pymoo.operators.sampling.rnd import BinaryRandomSampling

from src.cli.problem_cli import CLI, RunConfig
from src.cli.shared import SavedRun
from src.core.abstract import AcceptanceCriterion
from src.problems.default_pymoo_configurations import (
    BasePymooInstanceRunner,
    get_nsga_variants,
    get_spea_variants,
)
from src.problems.default_vns_configurations import (
    BaseVNSInstanceRunner,
    get_bvns_variants,
    get_movnd_vns_variants,
    get_rvns_hybrid_variants,
    get_rvns_variants,
)
from src.problems.mokp.problem import MOKPProblem, MOKPPymoo
from src.problems.mokp.vns import (
    flip_op,
    shake_flip,
    shake_swap,
    swap_op,
)
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


class VNSInstanceRunner(BaseVNSInstanceRunner[MOKPProblem]):
    def __init__(self, config: RunConfig):
        super().__init__(config, MOKPProblem.load(str(config.instance_path)), 18)


class SharedVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        for config_name, optimizer in get_rvns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("shake_flip", shake_flip), ("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_bvns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_flip", flip_op)],
            [("shake_flip", shake_flip), ("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_movnd_vns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_flip", flip_op)],
            [("shake_flip", shake_flip), ("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_rvns_hybrid_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_flip", flip_op), ("op_swap", swap_op)],
            [("shake_flip", shake_flip)],
        ):
            yield config_name, self.make_func(optimizer)


class SVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        alpha_weights = np.mean(self.problem.profits, axis=1)

        alpha_range = [0.25, 0.5, 1, 2, 3]
        dist_func = MOKPProblem.calculate_solution_distance_2
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
                self.problem,
                acceptance_criteria,
                [("shake_flip", shake_flip), ("shake_swap", shake_swap)],
            ),
            get_bvns_variants(
                self.problem,
                acceptance_criteria,
                [("op_flip", flip_op)],
                [("shake_flip", shake_flip), ("shake_swap", shake_swap)],
            ),
        ):
            config_name = config_name.replace("RVNS", "SVNS").replace("RVNS", "SVNS")
            yield config_name, self.make_func(optimizer)


class PymooInstanceRunner(BasePymooInstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(
            config, MOKPPymoo(MOKPProblem.load(str(config.instance_path))).to_config()
        )

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        for config_name, algorithm in get_nsga_variants(
            self.problem_config, BinaryRandomSampling()
        ):
            yield (config_name, self.make_func(config_name, algorithm))

        for config_name, algorithm in get_spea_variants(
            self.problem_config, BinaryRandomSampling()
        ):
            yield (config_name, self.make_func(config_name, algorithm))


if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    CLI(
        "MOKP",
        base,
        [
            SharedVNSInstanceRunner,
            SVNSInstanceRunner,
            PymooInstanceRunner,
        ],
        MOKPProblem,
    ).run()
