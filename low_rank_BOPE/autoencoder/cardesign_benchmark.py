#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from logging import Logger
from typing import Any, Dict, List, Tuple

import fblearner.flow.api as flow
import torch
from ax.utils.common.logger import get_logger
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.sampling import draw_sobol_samples
from low_rank_BOPE.autoencoder.bope_flow import (
    run_single_bo_stage,
    run_single_pe_stage,
)
from low_rank_BOPE.autoencoder.car_problems import (
    problem_setup_augmented,
)
from low_rank_BOPE.autoencoder.pairwise_autoencoder_gp import (
    get_fitted_outcome_model,
)
from low_rank_BOPE.autoencoder.utils import (
    generate_random_pref_data,
)

logger: Logger = get_logger(__name__)


def construct_problem(
    problem_name: str,
) -> Tuple[MultiObjectiveTestProblem, torch.nn.Module]:
    sim_setup = problem_setup_augmented(
        problem_name,
        augmented_dims_noise=0.0001,
        **{"dtype": torch.double},
    )
    problem = sim_setup[2]
    util_func = sim_setup[4]
    return problem, util_func


@flow.flow_async(
    # resource_requirements=flow.ResourceRequirements(cpu=1, memory="8"),
    # max_run_count=1,
)
@flow.registered(owners=["oncall+ae"])
@flow.typed()
def run_cardesign_benchmark(
    problem_name: str,
    strategy_name: str,
    strategies_args: Dict[str, Any],
    num_sobol_designs: int = 32,
    num_sobol_prefs: int = 25,
    num_bo_iters: int = 50,
    num_pref_iters: int = 50,
    num_stages: int = 1,
) -> Dict:
    problem, util_func = construct_problem(problem_name)
    pe_config_info = strategies_args

    # TODO:
    # to keep it aligned with our setup
    # we need to generate the initial X, Y, comps together
    # then fit the utility model first (together w autoencoder)
    # then apply the autoencoder to fit an outcome model

    # init - design points
    train_X = (
        draw_sobol_samples(bounds=problem.bounds, n=1, q=num_sobol_designs)
        .squeeze(0)
        .to(torch.double)
        .detach()
    )
    train_outcomes = problem(train_X).detach()

    outcome_model = get_fitted_outcome_model(train_X=train_X, train_Y=train_outcomes)

    # init - pref exploration
    train_comps, train_pref_outcomes, train_util_val = generate_random_pref_data(
        problem=problem,
        n=num_sobol_prefs,
        outcome_model=outcome_model,
        util_func=util_func,
    )

    pe_stage_result_list = []
    bo_stage_result_list = []

    for istage in range(num_stages):
        logger.info(f"run {istage}th PE stage ...")
        # run PE stage
        # train_pref_outcomes --- sampled from posterior (not the true outcome)
        (
            max_val_list,
            train_pref_outcomes,  # output updated pref datapoints from PE stage
            train_comps,  # output updated comparisons from PE stage
            util_model,
            pref_obj,  # learned objective
        ) = run_single_pe_stage(
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            outcome_model=outcome_model,
            pe_config_info=pe_config_info,
            num_pref_iters=num_pref_iters,
            problem=problem,
            util_func=util_func,
        )
        pe_stage_result_list.extend(max_val_list)

        logger.info(f"run {istage}th BO stage ...")
        # run BO stage
        util_list, train_X, train_outcomes, outcome_model = run_single_bo_stage(
            train_X=train_X,
            train_outcomes=train_outcomes,
            train_comps=train_comps,  # pass PE data to BO stage for util retraining
            train_pref_outcomes=train_pref_outcomes,  # pass PE data to BO stage for util retraining
            outcome_model=outcome_model,
            pref_obj=pref_obj,
            pe_config_info=pe_config_info,
            num_bo_iters=num_bo_iters,
            problem=problem,
            util_func=util_func,
        )
        bo_stage_result_list = (
            util_list  # util_list is updated within run_single_bo_stage
        )

    return {
        "PE": pe_stage_result_list,
        "BO": bo_stage_result_list,
    }


@flow.registered(owners=["oncall+ae"])
@flow.typed()
def run_cardesign_benchmark_reps(
    problem_name: str,
    strategies: Dict[str, Dict],
    num_sobol_designs: int = 32,
    num_sobol_prefs: int = 25,
    num_bo_iters: int = 25,
    num_pref_iters: int = 25,
    num_stages: int = 3,
    reps: int = 5,
) -> Dict[str, List[Dict]]:
    res_all = {s: [] for s in strategies}
    # carcabdesign_7d9d_piecewiselinear_72
    for _ in range(reps):
        for strategy, strategies_args in strategies.items():
            res_all[strategy].append(
                run_cardesign_benchmark(
                    problem_name=problem_name,
                    strategy_name=strategy,
                    num_sobol_designs=num_sobol_designs,
                    num_sobol_prefs=num_sobol_prefs,
                    num_bo_iters=num_bo_iters,
                    num_pref_iters=num_pref_iters,
                    num_stages=num_stages,
                    strategies_args=strategies_args,
                    return_on_failure=[{"PE": "Failed"}],
                )
            )
    return res_all
