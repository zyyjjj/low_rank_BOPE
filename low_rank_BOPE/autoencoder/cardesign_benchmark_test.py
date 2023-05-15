# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
from logging import Logger
from typing import Any, Dict, List, Tuple
import sys
sys.path.append("../..")

import torch
# from ax.utils.common.logger import get_logger 
import logging
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import PairwiseLaplaceMarginalLogLikelihood
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.sampling import draw_sobol_samples
from low_rank_BOPE.autoencoder.bope_flow import (
    run_single_bo_stage,
    run_single_pe_stage,
)
from low_rank_BOPE.autoencoder.pairwise_autoencoder_gp import (
    get_fitted_standard_outcome_model, 
    get_fitted_pca_outcome_model,
    get_fitted_pca_util_model,
    get_fitted_standard_util_model,
    get_autoencoder,
    jointly_optimize_models,
    initialize_util_model,
    initialize_outcome_model,
)
from low_rank_BOPE.autoencoder.utils import gen_comps
from low_rank_BOPE.fblearner.make_problem import make_problem_and_util_func
from low_rank_BOPE.fblearner.problem_setups import PROBLEM_SETUPS

# logger: Logger = get_logger(__name__)
logger = logging.getLogger("botorch")


def construct_problem(
    problem_name: str
) -> Tuple[MultiObjectiveTestProblem, torch.nn.Module]:
    
    problem, util_func = make_problem_and_util_func(
        problem_name, options = PROBLEM_SETUPS[problem_name])
    
    return problem, util_func


def run_cardesign_benchmark(
    problem_name: str,
    strategy_name: str,
    strategies_args: Dict[str, Any],
    num_sobol_designs: int = 32,
    num_bo_iters: int = 2,
    bo_batch_size: int = 1,
    num_pref_iters: int = 2,
    every_n_comps: int = 2,
    num_stages: int = 1,
) -> Dict:
    problem, util_func = construct_problem(problem_name)
    pe_config_info = strategies_args

    train_X = (
        draw_sobol_samples(bounds=problem.bounds, n=1, q=num_sobol_designs)
        .squeeze(0)
        .to(torch.double)
        .detach()
    )
    train_Y = problem(train_X).detach()
    train_util_val = util_func(train_Y).detach()
    train_comps = gen_comps(train_util_val)
    train_pref_outcomes = copy.deepcopy(train_Y)

    if strategies_args["util_model_name"] in ("autoencoder", "joint_autoencoder"):

        autoencoder = get_autoencoder(
            train_Y=train_Y,
            latent_dims=strategies_args["util_model_kwargs"]["autoencoder_latent_dims"], 
            # pre_train_epoch=strategies_args["util_model_kwargs"]["autoencoder_num_individual_pretrain_epochs"],
            pre_train_epoch=0,
        )
        train_Y_latent = autoencoder.encoder(train_Y).detach()
        outcome_model, mll_outcome = initialize_outcome_model(
            train_X=train_X, train_Y=train_Y_latent, 
            latent_dims=strategies_args["util_model_kwargs"]["autoencoder_latent_dims"]
        )
        train_pref_outcomes_latent = autoencoder.encoder(train_pref_outcomes).detach()
        util_model, mll_util = initialize_util_model(
            outcomes=train_pref_outcomes_latent, comps=train_comps, 
            latent_dims=strategies_args["util_model_kwargs"]["autoencoder_latent_dims"]
        )
    
    else:
        autoencoder = None


    pe_stage_result_list = []
    bo_stage_result_list = [train_util_val.squeeze(-1).tolist()]

    for istage in range(num_stages):

        logger.info(f"========== Start {istage}th stage ... ==========")

        if strategies_args["util_model_name"] == "joint_autoencoder":
            logger.info(f"======= Jointly train VAE, outcome, util models in {istage}th stage ... =======")
            logger.debug(f"train_X: {train_X.shape}, train_Y: {train_Y.shape}, train_comps: {train_comps.shape}")
            logger.debug(f"train X: {train_X}")
            # training VAE jointly w outcome and util models
            outcome_model, util_model, autoencoder = jointly_optimize_models(
                train_X=train_X,
                train_Y=train_Y,
                train_pref_outcomes=train_pref_outcomes,
                train_comps=train_comps,
                outcome_model=outcome_model,
                mll_outcome=mll_outcome,
                util_model=util_model,
                mll_util=mll_util,
                autoencoder=autoencoder,
                num_epochs=strategies_args["util_model_kwargs"]["autoencoder_num_joint_train_epochs"],
                train_ae=True,
                train_outcome_model=True,
                train_util_model=True,
            )
            logger.debug(f"After joint training w VAE, outcome_model.train_targets.shape: {outcome_model.train_targets.shape}")
            logger.debug(f"After joint training w VAE, util model input and output shape: {util_model.unconsolidated_datapoints.shape}, {util_model.unconsolidated_comparisons.shape}")
        elif strategies_args["util_model_name"] == "autoencoder":
            logger.info(f"======= Train VAE, outcome, util models separately in {istage}th stage ... =======")
            logger.debug(f"train_X: {train_X.shape}, train_Y: {train_Y.shape}, train_comps: {train_comps.shape}")
            autoencoder = get_autoencoder(
                train_Y=train_Y,
                latent_dims=strategies_args["util_model_kwargs"]["autoencoder_latent_dims"], 
                pre_train_epoch=strategies_args["util_model_kwargs"]["autoencoder_num_individual_pretrain_epochs"],
            )
            train_Y_latent = autoencoder.encoder(train_Y).detach()
            outcome_model = get_fitted_standard_outcome_model(
                train_X=train_X, train_Y=train_Y_latent,
            )
            logger.debug(f"outcome_model.train_targets.shape: {outcome_model.train_targets.shape}")
            train_pref_outcomes_latent = autoencoder.encoder(train_pref_outcomes).detach()
            util_model = get_fitted_standard_util_model(
                train_pref_outcomes = train_pref_outcomes_latent,
                train_comps = train_comps,
            )
        elif strategies_args["util_model_name"] == "pca":
            logger.info(f"======= Train PCA outcome and util models in {istage}th stage ... =======")
            outcome_model = get_fitted_pca_outcome_model(
                train_X=train_X,
                train_Y=train_Y,
                pca_var_threshold = strategies_args["util_model_kwargs"]["pca_var_threshold"],
                standardize=strategies_args["util_model_kwargs"].get("standardize", False)
            )
            util_model = get_fitted_pca_util_model(
                train_Y=train_Y,
                train_pref_outcomes=train_pref_outcomes,
                train_comps=train_comps,
                pca_var_threshold = strategies_args["util_model_kwargs"]["pca_var_threshold"],
                num_unlabeled_outcomes=strategies_args["util_model_kwargs"].get("num_unlabeled_outcomes", 0),
                standardize=strategies_args["util_model_kwargs"].get("standardize", False)
            )
        elif strategies_args["util_model_name"] == "standard":
            logger.info(f"======= Train standard outcome and util models in {istage}th stage ... =======")
            outcome_model = get_fitted_standard_outcome_model(
                train_X=train_X,
                train_Y=train_Y,
            )
            util_model = get_fitted_standard_util_model(
                train_pref_outcomes=train_pref_outcomes,
                train_comps=train_comps,
            )
        else:
            raise NotImplementedError(f"{strategies_args['util_model_name']} is not supported")

        logger.info(f"======= run {istage}th PE stage ... =======")
        # run PE stage
        # train_pref_outcomes --- sampled from posterior (not the true outcome)
        (
            max_val_list,
            train_pref_outcomes,  # output updated pref datapoints from PE stage
            train_comps,  # output updated comparisons from PE stage
            util_model,
            pref_obj,  # learned objective
            autoencoder
        ) = run_single_pe_stage(
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            outcome_model=outcome_model,
            pe_config_info=pe_config_info,
            num_pref_iters=num_pref_iters,
            every_n_comps=every_n_comps,
            problem=problem,
            util_func=util_func,
            autoencoder=autoencoder,
        )
        pe_stage_result_list.extend(max_val_list)
        mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)

        logger.debug(f"train_pref_outcomes shape: {train_pref_outcomes.shape}")
        logger.debug(f"train_comps shape: {train_comps.shape}")
        logger.debug(f"outcome model num outputs: {outcome_model.num_outputs}")
        logger.debug(f"autoencoder after {istage}th PE stage: {autoencoder}")
        logger.debug(f"After {istage}th PE stage, util model input and output shape: {util_model.unconsolidated_datapoints.shape}, {util_model.unconsolidated_comparisons.shape}")

        logger.info(f"======= run {istage}th BO stage ... =======")
        # run BO stage
        bo_stage_result_list, train_X, train_Y, outcome_model, autoencoder = run_single_bo_stage(
            train_X=train_X,
            train_outcomes=train_Y,
            train_comps=train_comps,  # pass PE data to BO stage for util retraining
            train_pref_outcomes=train_pref_outcomes,  # pass PE data to BO stage for util retraining
            outcome_model=outcome_model,
            pref_obj=pref_obj,
            pe_config_info=pe_config_info,
            num_bo_iters=num_bo_iters,
            bo_batch_size=bo_batch_size,
            problem=problem,
            util_func=util_func,
            util_list=bo_stage_result_list,
            autoencoder=autoencoder,
        )
        logger.debug(f"After {istage}th BO stage, train_X and train_Y shape: {train_X.shape}, {train_Y.shape}")
        mll_outcome = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)

    return {
        "problem_name": problem_name,
        "strategy_name": strategy_name,
        "PE": pe_stage_result_list,
        "BO": bo_stage_result_list,
    }


def run_cardesign_benchmark_reps(
    problem_name: str,
    strategies: Dict[str, Dict],
    num_sobol_designs: int = 32,
    num_bo_iters: int = 25,
    bo_batch_size: int = 1,
    num_pref_iters: int = 25,
    every_n_comps: int = 2,
    num_stages: int = 3,
    reps: int = 5,
) -> Dict[str, List[Dict]]:
    res_all = {s: [] for s in strategies}

    for _ in range(reps):
        for strategy, strategies_args in strategies.items():
            print(f"~~~~~~~~~~~~ Run {strategy} ~~~~~~~~~~~~")
            res_all[strategy].append(
                run_cardesign_benchmark(
                    problem_name=problem_name,
                    strategy_name=strategy,
                    num_sobol_designs=num_sobol_designs,
                    num_bo_iters=num_bo_iters,
                    bo_batch_size=bo_batch_size,
                    num_pref_iters=num_pref_iters,
                    every_n_comps=every_n_comps,
                    num_stages=num_stages,
                    strategies_args=strategies_args,
                    # return_on_failure=[{"PE": "Failed"}],
                )
            )
    return res_all
