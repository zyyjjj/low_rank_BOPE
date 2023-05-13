#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
# from ax.utils.common.logger import get_logger
from botorch.acquisition import LearnedObjective
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement, qSimpleRegret
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.models import PairwiseGP, SingleTaskGP
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.sampling import draw_sobol_samples
from low_rank_BOPE.autoencoder.pairwise_autoencoder_gp import (
    get_fitted_standard_util_model,
    get_fitted_autoencoded_util_model,
    get_fitted_pca_util_model,
    get_fitted_standard_outcome_model,
    get_fitted_autoencoded_outcome_model,
    get_fitted_pca_outcome_model,
    Autoencoder
)
from low_rank_BOPE.autoencoder.utils import gen_comps
from torch import Tensor

# logger: Logger = get_logger(__name__)
logger = logging.getLogger("botorch")
# logger.setLevel(logging.INFO)
# logger.handlers.pop()

MC_SAMPLES = 256
NUM_RESTARTS = 10
RAW_SAMPLES = 256


BOPE_STRATEGY_FACTORY = {
    "autoencoder-eubo": {
        "util_model_name": "autoencoder",
        "util_model_kwargs": {
            "autoencoder_latent_dims": 2,
            "num_unlabeled_outcomes": 0,  # do not add unlabeled outcomes
            "autoencoder_num_joint_train_epochs": 500,
            "autoencoder_num_pretrain_epochs": 200,
        },
        "pe_gen_strategy": "eubo",
        "bo_gen_kwargs": {
            "retrain_util_model": False,
        },
    },
    "pca-eubo": {
        "util_model_name": "pca",
        "util_model_kwargs": {
            "num_unlabeled_outcomes": 0,  # do not add unlabeled outcomes
            "pca_var_threshold": 0.95,
        },
        "pe_gen_strategy": "eubo",
        "bo_gen_kwargs": {
            "retrain_util_model": False,
        },
    },
    "eubo": {
        "util_model_name": "standard",
        "pe_gen_strategy": "eubo",
        "bo_gen_kwargs": {
            "retrain_util_model": False,
        },
    },
    "random-f": {
        "util_model_name": "standard",
        "pe_gen_strategy": "random-f",
        "bo_gen_kwargs": {
            "retrain_util_model": False,
        },
    },
}


def gen_eubo_candidates(
    pref_model: PairwiseGP, outcome_model: GPyTorchModel, bounds: Tensor
) -> Union[Tensor, Tensor, Tensor]:
    one_sample_outcome_model = FixedSingleSampleModel(model=outcome_model)
    acqf = AnalyticExpectedUtilityOfBestOption(
        pref_model=pref_model, outcome_model=one_sample_outcome_model
    )
    # opt acqf
    cand_X, acqf_val = optimize_acqf(
        acq_function=acqf,
        q=2,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "sequential": True},  # change batch limit to 5
    )
    cand_Y = one_sample_outcome_model(cand_X)
    return cand_X, cand_Y, acqf_val


def gen_random_f_candidates(
    outcome_model: GPyTorchModel, bounds: Tensor
) -> Tuple[Tensor, Tensor]:
    cand_X = (
        draw_sobol_samples(
            bounds=bounds,
            n=1,
            q=2,
        )
        .squeeze(0)
        .to(torch.double)
    )
    cand_Y = outcome_model.posterior(cand_X).rsample().squeeze(0).detach()
    return cand_X, cand_Y


def get_candidate_maximize_util(
    pref_model: PairwiseGP, outcome_model: GPyTorchModel, bounds: Tensor
) -> Tensor:
    sampler = SobolQMCNormalSampler(MC_SAMPLES)  # check default val
    # use default sampler in the LearnedObjective
    pref_obj = LearnedObjective(pref_model=pref_model)

    acq_func = qSimpleRegret(
        model=outcome_model,
        sampler=sampler,
        objective=pref_obj,  # learned objective
    )

    # identified candidate via maximize posterior mean (this depends on outcome model accuracy)
    candidates, acqf_val = optimize_acqf(
        acq_function=acq_func,
        q=1,
        bounds=bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=True,
    )
    return candidates


def get_and_fit_outcome_model(
    train_X: Tensor,
    train_Y: Tensor,
    util_model_name: str, # not the best name but lets keep it for now
    **kwargs,
) -> SingleTaskGP:
    util_model_kwargs = kwargs.get("util_model_kwargs", {})
    logger.info(f"Running get_and_fit_outcome_model, util_model_kwargs: {util_model_kwargs}")

    if util_model_name == "joint_autoencoder": 
        outcome_model, _ = get_fitted_autoencoded_outcome_model(
            train_X=train_X,
            train_Y=train_Y,
            latent_dims=util_model_kwargs.get("autoencoder_latent_dims", 2),
            num_joint_train_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_train_epochs", 500
            ),
            num_autoencoder_pretrain_epochs=util_model_kwargs.get(
                "autoencoder_num_pretrain_epochs", 200
            ),
            fix_vae=True
        )
    elif util_model_name == "pca":
        outcome_model = get_fitted_pca_outcome_model(
            train_X=train_X,
            train_Y=train_Y,
            var_threshold=util_model_kwargs.get("pca_var_threshold", 0.95),
        )
    else: # TODO: doublecheck
        outcome_model = get_fitted_standard_outcome_model(
            train_X=train_X,
            train_Y=train_Y
        )

    return outcome_model

    
def get_and_fit_util_model(
    train_Y: Tensor,
    train_comps: Tensor,
    util_model_name: str,
    outcome_model: Optional[GPyTorchModel] = None,
    bounds: Optional[Tensor] = None,
    **kwargs,
) -> PairwiseGP:
    util_model_kwargs = kwargs.get("util_model_kwargs", {})
    logger.info(f"Running get_and_fit_util_model, util_model_kwargs: {util_model_kwargs}")

    if util_model_name == "autoencoder":
        util_model, _ = get_fitted_autoencoded_util_model(
            train_Y=train_Y,
            train_comps=train_comps,
            latent_dims=util_model_kwargs.get("autoencoder_latent_dims", 2), 
            # TODO: should keep this consistent with pca 
            # if it's hard to do in one run, maybe set it in config with reasonable number 
            # from other runs with PCA
            num_joint_train_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_train_epochs", 500
            ),
            num_autoencoder_pretrain_epochs=util_model_kwargs.get(
                "autoencoder_num_pretrain_epochs", 200
            ),
            num_unlabeled_outcomes=util_model_kwargs.get("num_unlabeled_outcomes", 0),
            outcome_model=outcome_model,
            bounds=bounds,
        )
    elif util_model_name == "joint_autoencoder":
        util_model, _ = get_fitted_autoencoded_util_model(
            train_Y=train_Y,
            train_comps=train_comps,
            latent_dims=util_model_kwargs.get("autoencoder_latent_dims", 2), 
            num_joint_train_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_train_epochs", 500
            ),
            num_autoencoder_pretrain_epochs=util_model_kwargs.get(
                "autoencoder_num_pretrain_epochs", 200
            ),
            num_unlabeled_outcomes=util_model_kwargs.get("num_unlabeled_outcomes", 0),
            outcome_model=outcome_model,
            bounds=bounds,
            fix_vae=True
        )
    elif util_model_name == "pca":
        util_model = get_fitted_pca_util_model(
            train_Y=train_Y,
            train_comps=train_comps,
            pca_var_threshold=util_model_kwargs.get("pca_var_threshold", 0.95),
            num_unlabeled_outcomes=util_model_kwargs.get("num_unlabeled_outcomes", 0),
            outcome_model=outcome_model,
            bounds=bounds,
        )
    elif util_model_name == "standard":
        util_model = get_fitted_standard_util_model(
            train_Y=train_Y,
            train_comps=train_comps,
        )
    return util_model


def gen_pe_candidates(
    pe_gen_strategy: str,
    pref_model: PairwiseGP,
    outcome_model: GPyTorchModel,
    bounds: Tensor,
) -> Tuple[Tensor, Tensor]:
    # EUBO-zeta:
    if pe_gen_strategy == "eubo":
        cand_X, cand_Y, _ = gen_eubo_candidates(
            pref_model=pref_model,
            outcome_model=outcome_model,
            bounds=bounds,
        )
    elif pe_gen_strategy == "random-f":
        # random-f
        cand_X, cand_Y = gen_random_f_candidates(
            outcome_model=outcome_model,
            bounds=bounds,
        )
    return cand_X, cand_Y


def run_single_pe_stage(
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
    outcome_model: GPyTorchModel,
    pe_config_info: Dict[str, Any],
    num_pref_iters: int,
    problem: MultiObjectiveTestProblem,
    util_func: torch.nn.Module,
) -> Union[List, Tensor, Tensor, PairwiseGP, LearnedObjective]:
    max_val_list = []
    for i in range(num_pref_iters):  # number of batch iterations
        # fit pref model
        util_model_name = pe_config_info["util_model_name"]
        util_model_kwargs = pe_config_info.get("util_model_kwargs", {})
        util_model = get_and_fit_util_model(
            train_Y=train_pref_outcomes,
            train_comps=train_comps,
            util_model_name=util_model_name,
            outcome_model=outcome_model,
            bounds=problem.bounds,
            # **util_model_kwargs,
            util_model_kwargs=util_model_kwargs, # TODO: call this out when committing
        )

        # identified candidate that max util
        best_candidates = get_candidate_maximize_util(
            pref_model=util_model,
            outcome_model=outcome_model,
            bounds=problem.bounds,
        )

        max_val_identified = util_func(
            problem.evaluate_true(best_candidates.to(torch.double))
        ).item()  # best obtained util
        max_val_list.append(max_val_identified)
        logger.info(f"{i}th iter: max val identified = {max_val_identified}")

        # gen PE candidates
        cand_X, cand_Y = gen_pe_candidates(
            pe_gen_strategy=pe_config_info["pe_gen_strategy"],
            pref_model=util_model,
            outcome_model=outcome_model,
            bounds=problem.bounds,
        )

        # obtain evaluated data - preferences from DM
        cand_Y = cand_Y.detach().clone()
        cand_comps = gen_comps(util_func(cand_Y))

        # update training data
        train_comps = torch.cat(
            (train_comps, cand_comps + train_pref_outcomes.shape[0])
        )
        train_pref_outcomes = torch.cat((train_pref_outcomes, cand_Y))

    util_model = get_and_fit_util_model(
        train_Y=train_pref_outcomes,
        train_comps=train_comps,
        util_model_name=util_model_name,
        outcome_model=outcome_model,
        bounds=problem.bounds,
        # **util_model_kwargs,
        util_model_kwargs=util_model_kwargs
    )
    # Use DEFAULT sampler to speed things up
    # TODO: this is different from what we do, to revisit
    pref_obj = LearnedObjective(pref_model=util_model)
    return max_val_list, train_pref_outcomes, train_comps, util_model, pref_obj


def run_single_bo_stage(
    train_X: Tensor,
    train_outcomes: Tensor,
    train_comps: Tensor,
    train_pref_outcomes: Tensor,
    outcome_model: GPyTorchModel,
    pref_obj: LearnedObjective,
    pe_config_info: Dict[str, Any],
    num_bo_iters: int,
    problem: MultiObjectiveTestProblem,
    util_func: torch.nn.Module,
) -> Union[List, Tensor, Tensor, GPyTorchModel]:
    bo_gen_kwargs = pe_config_info.get("bo_gen_kwargs", {})
    util_list = list(util_func(problem.evaluate_true(train_X)).detach().numpy())

    logger.info(f"Running BO, current length of util val = {len(util_list)}")

    for i in range(num_bo_iters):

        acq_func = qNoisyExpectedImprovement(
            model=outcome_model,
            objective=pref_obj,  # should be learned utility objective
            X_baseline=train_X,
            sampler=SobolQMCNormalSampler(MC_SAMPLES),
            prune_baseline=True,
            cache_root=False,
        )

        # optimize the acquisition function
        candidates, acqf_val = optimize_acqf(
            acq_function=acq_func,
            q=1,
            bounds=problem.bounds,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={
                "batch_limit": 5,
            },
            sequential=True,
        )
        # append eval
        train_X = torch.cat([train_X, candidates], dim=0)
        logger.info(f"train x shape = {train_X.shape}")
        train_outcomes = torch.cat(
            [train_outcomes, problem(candidates.to(torch.double)).detach()], dim=0
        )
        logger.info(f"train outcome shape = {train_outcomes.shape}")

        true_util = util_func(
            problem.evaluate_true(candidates.to(torch.double))
        ).item()  # best obtained util
        util_list.append(true_util)
        logger.info(
            f"Finished {i}th BO iteration with best obtained util val = {max(util_list)}"
        )

        # outcome_model = get_fitted_standard_outcome_model(
        #     train_X=train_X, train_Y=train_outcomes
        # )
        outcome_model = get_and_fit_outcome_model(
            train_X=train_X,
            train_Y=train_outcomes,
            util_model_name=pe_config_info["util_model_name"],
            bounds=problem.bounds,
            util_model_kwargs=pe_config_info["util_model_kwargs"],
        )

        # update util model with updated outcome model if retrain_util_model=True
        if bo_gen_kwargs.get("retrain_util_model", False):
            util_model = get_and_fit_util_model(
                train_Y=train_pref_outcomes,  # labeled train_pref_outcomes data is not updated in BO stage
                train_comps=train_comps,
                util_model_name=pe_config_info["util_model_name"],
                outcome_model=outcome_model,  # updated outcome model (only useful if adding sampled data from outcome model)
                bounds=problem.bounds,
                util_model_kwargs=pe_config_info["util_model_kwargs"],
            )
            pref_obj = LearnedObjective(pref_model=util_model)

    return util_list, train_X, train_outcomes, outcome_model
