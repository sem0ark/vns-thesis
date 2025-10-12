import itertools
from itertools import product
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
from src.core.abstract import OptimizerAbstract
from src.problems.moscp.problem import MOSCPProblem, MOSCPProblemPymoo
from src.problems.moscp.vns import flip_op_v2, shake_flip
from src.problems.vns_runner_utils import run_vns_optimizer
from src.vns.acceptance import AcceptBatch, AcceptBatchSkewed
from src.vns.local_search import (
    best_improvement,
    noop,
)
from src.vns.optimizer import VNSOptimizer


class VNSInstanceRunner(InstanceRunner):
    def __init__(self, config: RunConfig):
        super().__init__(config)
        self.problem = MOSCPProblem.load(str(config.instance_path))

    def get_variants(self) -> Iterable[tuple[str, Callable[[RunConfig], SavedRun]]]:
        alpha_weights = np.mean(self.problem.costs, axis=1)
        
        acceptance_criteria = [
            ("batch", AcceptBatch()),
            *[
                (
                    f"skewed a{mult}",
                    AcceptBatchSkewed(alpha_weights * mult, MOSCPProblem.calculate_solution_distance),
                )
                for mult in range(1, 10 + 1)
            ],
        ]

        local_search_functions = [
            ("noop", noop()),
            *[
                (f"{search_name} {op_name}", search_func_factory(op_func))
                for (
                    (search_name, search_func_factory),
                    (op_name, op_func),
                ) in itertools.product(
                    [("BI", best_improvement)],
                    [("op_flip", flip_op_v2)],
                )
            ],
        ]
        shake_functions = [
            ("shake_flip", shake_flip),
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

            config_name = (
                f"vns {common_name} {acc_name} {search_name} k{k} {shake_name}"
            )

            optimizer = VNSOptimizer(
                problem=self.problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acc_criteria,
                shake_function=shake_func,
                name=config_name,
                version=16,
            )

            yield config_name, self.make_func(optimizer)

    def make_func(self, optimizer: OptimizerAbstract):
        def run(config: RunConfig):
            solutions = run_vns_optimizer(
                config.run_time_seconds,
                optimizer,
            )

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=optimizer.name,
                    version=optimizer.version,
                    problem_name="MOSCP",
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
    CLI("MOSCP", base, [VNSInstanceRunner, PymooInstanceRunner], MOSCPProblem).run()
