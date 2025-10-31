import itertools
from typing import Iterable

from src.core.abstract import AcceptanceCriterion, Problem
from src.problems.moscp.vns import flip_op_v2 as flip_op
from src.vns.local_search import (
    NeighborhoodOperator,
    best_improvement,
    composite,
    composite_parallel,
    noop,
)
from src.vns.optimizer import ShakeFunction, VNSOptimizer


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
    for (
        (acc_name, acceptance_criterion),
        (search_name, search_func, shake_func),
        k,
    ) in itertools.product(
        acceptance_operators,
        [
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
        + [
            (
                f"parallel_MO_BI {neib_name} {shake_name}",
                composite_parallel(
                    [
                        best_improvement(flip_op, objective_index=i)
                        for i in range(problem.num_objectives)
                    ]
                ),
                shake,
            )
            for shake_name, shake in shake_functions
            for neib_name, neightborhood in neightborhoods
        ],
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
