from pathlib import Path
from typing import Callable, Iterable

from pymoo.operators.sampling.rnd import BinaryRandomSampling

from src.cli.problem_cli import CLI, RunConfig
from src.cli.shared import SavedRun
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
from src.problems.uflp.problem import MOUFLPProblem, MOUFLPProblemPymoo
from src.problems.uflp.vns import flip_op_v2 as flip_op
from src.problems.uflp.vns import shake_flip
from src.vns.acceptance import AcceptBatch, AcceptBeam


class VNSInstanceRunner(BaseVNSInstanceRunner[MOUFLPProblem]):
    def __init__(self, config: RunConfig):
        super().__init__(config, MOUFLPProblem.load(str(config.instance_path)), 18)


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


class PymooInstanceRunner(BasePymooInstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(
            config,
            MOUFLPProblemPymoo(
                MOUFLPProblem.load(str(config.instance_path))
            ).to_config(),
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
        "MOUFLP",
        base,
        [
            SharedVNSInstanceRunner,
            PymooInstanceRunner,
        ],
        MOUFLPProblem,
    ).run()
