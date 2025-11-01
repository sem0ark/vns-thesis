import itertools
from typing import Iterable, TypeVar

from src.cli.problem_cli import InstanceRunner, RunConfig
from src.cli.shared import Metadata, SavedRun, SavedSolution
from src.cli.vns_runner import run_vns_optimizer
from src.core.abstract import AcceptanceCriterion, OptimizerAbstract, Problem
from src.core.termination import terminate_time_based
from src.vns.local_search import (
    NeighborhoodOperator,
    best_improvement,
    composite,
    composite_parallel,
    noop,
)
from src.vns.optimizer import ShakeFunction, VNSOptimizer

P = TypeVar("P", bound=Problem)


class BaseVNSInstanceRunner[P](InstanceRunner):
    def __init__(self, config: RunConfig, problem: P, version: int):
        super().__init__(config)
        self.problem = problem
        self.version = version

    def make_func(self, optimizer: OptimizerAbstract):
        def run(config: RunConfig):
            solutions = run_vns_optimizer(
                optimizer, terminate_time_based(config.run_time_seconds)
            )

            return SavedRun(
                metadata=Metadata(
                    run_time_seconds=int(config.run_time_seconds),
                    name=optimizer.name,
                    version=self.version,
                    problem_name=optimizer.problem.problem_name,
                    instance_name=config.instance_path.stem,
                ),
                solutions=[
                    SavedSolution(sol.objectives, sol.to_json_serializable())
                    for sol in solutions
                ],
            )

        return run


def get_rvns_variants(
    problem: Problem,
    acceptance_operators: list[tuple[str, AcceptanceCriterion]],
    shake_functions: list[tuple[str, ShakeFunction]],
) -> Iterable[tuple[str, VNSOptimizer]]:
    for (
        (acc_name, acceptance_criterion),
        (search_name, search_func, shake_func),
        k,
    ) in itertools.product(
        acceptance_operators,
        [(f"noop {name}", noop(), shake) for name, shake in shake_functions],
        range(1, 10),
    ):
        config_name = f"vns RVNS {acc_name} {search_name} k{k}"

        yield (
            config_name,
            VNSOptimizer(
                problem=problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
            ),
        )


def get_bvns_variants(
    problem: Problem,
    acceptance_operators: list[tuple[str, AcceptanceCriterion]],
    neightborhoods: list[tuple[str, NeighborhoodOperator]],
    shake_functions: list[tuple[str, ShakeFunction]],
) -> Iterable[tuple[str, VNSOptimizer]]:
    for (
        (acc_name, acceptance_criterion),
        (search_name, search_func, shake_func),
        k,
    ) in itertools.product(
        acceptance_operators,
        [
            (f"BI {neib_name} {shake_name}", best_improvement(neightborhood), shake)
            for shake_name, shake in shake_functions
            for neib_name, neightborhood in neightborhoods
        ],
        range(1, 10),
    ):
        config_name = f"vns BVNS {acc_name} {search_name} k{k}"

        yield (
            config_name,
            VNSOptimizer(
                problem=problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
            ),
        )


def get_movnd_vns_variants(
    problem: Problem,
    acceptance_operators: list[tuple[str, AcceptanceCriterion]],
    neightborhoods: list[tuple[str, NeighborhoodOperator]],
    shake_functions: list[tuple[str, ShakeFunction]],
) -> Iterable[tuple[str, VNSOptimizer]]:
    movnd_search = [
        (
            f"MOVND_BI {neib_name} {shake_name}",
            composite(
                [
                    best_improvement(neightborhood, objective_index=i)
                    for i in range(problem.num_objectives)
                ]
            ),
            shake,
        )
        for shake_name, shake in shake_functions
        for neib_name, neightborhood in neightborhoods
    ]
    parallel_movnd_search = [
        (
            f"parallel_MO_BI {neib_name} {shake_name}",
            composite_parallel(
                [
                    best_improvement(neightborhood, objective_index=i)
                    for i in range(problem.num_objectives)
                ]
            ),
            shake,
        )
        for shake_name, shake in shake_functions
        for neib_name, neightborhood in neightborhoods
    ]

    for (
        (acc_name, acceptance_criterion),
        (search_name, search_func, shake_func),
        k,
    ) in itertools.product(
        acceptance_operators,
        movnd_search + parallel_movnd_search,
        range(1, 10),
    ):
        config_name = f"vns GVNS {acc_name} {search_name} k{k}"

        yield (
            config_name,
            VNSOptimizer(
                problem=problem,
                search_functions=[search_func] * k,
                acceptance_criterion=acceptance_criterion,
                shake_function=shake_func,
                name=config_name,
            ),
        )
