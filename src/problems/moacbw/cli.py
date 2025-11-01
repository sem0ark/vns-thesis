from itertools import chain
from pathlib import Path
from typing import Callable, Iterable

from pymoo.operators.sampling.rnd import PermutationRandomSampling

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
    get_rvns_variants,
)
from src.problems.moacbw.problem import MOACBWProblem, MOACBWProblemPymoo
from src.problems.moacbw.vns import shake_swap, swap_op
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
from src.vns_extensions.variable_formulation_vns import AcceptBatchVFS


class VNSInstanceRunner(BaseVNSInstanceRunner[MOACBWProblem]):
    def __init__(self, config: RunConfig):
        super().__init__(config, MOACBWProblem.load(str(config.instance_path)), 18)


class SharedVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        for config_name, optimizer in get_rvns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_bvns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_swap", swap_op)],
            [("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_movnd_vns_variants(
            self.problem,
            [("beam", AcceptBeam()), ("batch", AcceptBatch())],
            [("op_swap", swap_op)],
            [("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)


class VFS_VNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        acceptance_criteria: list[tuple[str, AcceptanceCriterion]] = [
            (
                "VFS vfs_max_sum",
                AcceptBatchVFS(
                    self.problem.calculate_objectives_max,
                    self.problem.calculate_objectives_sum,
                ),
            ),
            (
                "VFS vfs_sum_max",
                AcceptBatchVFS(
                    self.problem.calculate_objectives_max,
                    self.problem.calculate_objectives_sum,
                ),
            ),
        ]

        for config_name, optimizer in get_rvns_variants(
            self.problem,
            acceptance_criteria,
            [("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_bvns_variants(
            self.problem,
            acceptance_criteria,
            [("op_swap", swap_op)],
            [("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)

        for config_name, optimizer in get_movnd_vns_variants(
            self.problem,
            acceptance_criteria,
            [("op_swap", swap_op)],
            [("shake_swap", shake_swap)],
        ):
            yield config_name, self.make_func(optimizer)


class SVNSInstanceRunner(VNSInstanceRunner):
    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        alpha_range = [0.25, 0.5, 1, 2, 3]
        dist_func = MOACBWProblem.calculate_solution_distance
        acceptance_criteria: list[tuple[str, AcceptanceCriterion]] = [
            (
                f"skewed_v{acc_idx} {name} a{mult}",
                acc([mult] * self.problem.num_objectives, dist_func),
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
                [("shake_swap", shake_swap)],
            ),
            get_bvns_variants(
                self.problem,
                acceptance_criteria,
                [("op_swap", swap_op)],
                [("shake_swap", shake_swap)],
            ),
        ):
            config_name = config_name.replace("RVNS", "SVNS").replace("RVNS", "SVNS")
            yield config_name, self.make_func(optimizer)


class PymooInstanceRunner(BasePymooInstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(
            config,
            MOACBWProblemPymoo(
                MOACBWProblem.load(str(config.instance_path))
            ).to_config(),
        )

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        for config_name, algorithm in get_nsga_variants(
            self.problem_config, PermutationRandomSampling()
        ):
            yield (config_name, self.make_func(config_name, algorithm))

        for config_name, algorithm in get_spea_variants(
            self.problem_config, PermutationRandomSampling()
        ):
            yield (config_name, self.make_func(config_name, algorithm))


if __name__ == "__main__":
    base = Path(__file__).parent / "runs"
    CLI(
        "moacbw",
        base,
        [
            SharedVNSInstanceRunner,
            VFS_VNSInstanceRunner,
            SVNSInstanceRunner,
            PymooInstanceRunner,
        ],
        MOACBWProblem,
    ).run()
