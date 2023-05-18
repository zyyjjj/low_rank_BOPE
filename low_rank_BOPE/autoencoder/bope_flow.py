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
from low_rank_BOPE.autoencoder.utils import gen_comps, ModifiedFixedSingleSampleModel
from torch import Tensor

# logger: Logger = get_logger(__name__)
logger = logging.getLogger("botorch")

OUTCOME_MC_SAMPLES = 256
PREF_MC_SAMPLES = 4
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
    pref_model: PairwiseGP, outcome_model: GPyTorchModel, bounds: Tensor,
    outcome_dim: int,
    autoencoder: Optional[Autoencoder] = None,
    use_modified_fixedsinglesamplemodel: bool = False,
) -> Union[Tensor, Tensor, Tensor]:
    r"""
    Generate candidate outcomes to compare using the EUBO acquisition function.
    Args:
        pref_model: The preference model.
        outcome_model: The outcome model.
        bounds: The bounds of the design space.
        outcome_dim: The dimension of the outcome space.
        autoencoder: The autoencoder for decoding the outcome model outputs, if
            the outcome model is fit on dim-reduced space.
        use_modified_fixedsinglesamplemodel: Whether to use the modified
            FixedSingleSampleModel. This must be True for the pca method, since
            the dim reduction is implemented through an outcome transform. This 
            does not have to be True for autoencoder methods, since the outcome
            model is fitted on the latent outcome representations explicitly.
    Returns:
        cand_X: The candidate design points.
        cand_Y: The outcome values associated with cand_X.
        acqf_val: The acquisition function value at the candidate design points.
    """

    if not use_modified_fixedsinglesamplemodel:
        one_sample_outcome_model = FixedSingleSampleModel(model=outcome_model)
    else:
        one_sample_outcome_model = ModifiedFixedSingleSampleModel(
            model=outcome_model, outcome_dim=outcome_dim)

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
    if autoencoder is not None:
        cand_Y = autoencoder.decoder(cand_Y).detach()
    
    return cand_X, cand_Y, acqf_val


def gen_random_f_candidates(
    outcome_model: GPyTorchModel, bounds: Tensor,
    autoencoder: Optional[Autoencoder] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Generate candidate outcomes to compare using Random-f strategy, i.e., take
    samples from the outcome model posterior.
    Args:
        outcome_model: The outcome model.
        bounds: The bounds of the design space.
        autoencoder: The autoencoder for decoding the outcome model outputs, if
            the outcome model is fit on dim-reduced space.
    Returns:
        cand_X: The candidate design points.
        cand_Y: The outcome values associated with cand_X.
    """

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
    if autoencoder is not None:
        cand_Y = autoencoder.decoder(cand_Y).detach()

    return cand_X, cand_Y


def get_candidate_maximize_util(
    pref_model: PairwiseGP, outcome_model: GPyTorchModel, bounds: Tensor
) -> Tensor:
    r"""
    Generate candidate designs that maximize the utility model posterior mean.
    Args:
        pref_model: The preference model.
        outcome_model: The outcome model.
        bounds: The bounds of the design space.
    Returns:
        candidates: The candidate design points.
    """

    sampler = SobolQMCNormalSampler(OUTCOME_MC_SAMPLES) 
    # use the same number of samples as in RetrainingBopeExperiment
    pref_obj = LearnedObjective(
        pref_model=pref_model,
        sampler=SobolQMCNormalSampler(1)
    )

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
    r"""
    Instantiate and fit an outcome model under the specified **kwargs.
    Args:
        train_X: The training design points.
        train_Y: The training outcome values.
        util_model_name: The name of the utility model to use.
            if "autoencoder" or "joint_autoencoder": train the outcome model 
                under a fixed autoencoder's latent space
            if "pca": train the outcome model on the PCA subspace learned on
                train_Y
            otherwise: fit a standard, non-dim-reduced outcome model
    Returns:
        outcome_model: The fitted outcome model.
    """

    util_model_kwargs = kwargs.get("util_model_kwargs", {})
    logger.debug(f"Running get_and_fit_outcome_model, util_model_kwargs: {util_model_kwargs}")

    if util_model_name in ("autoencoder", "joint_autoencoder"): 
        outcome_model, _ = get_fitted_autoencoded_outcome_model(
            train_X=train_X,
            train_Y=train_Y,
            autoencoder=kwargs.get("autoencoder", None),
            latent_dims=util_model_kwargs.get("autoencoder_latent_dims", 2),
            num_joint_train_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_train_epochs", 500
            ),
            num_autoencoder_pretrain_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_pretrain_epochs", 200
            ),
            fix_vae=True
        )
    elif util_model_name == "pca":
        outcome_model = get_fitted_pca_outcome_model(
            train_X=train_X,
            train_Y=train_Y,
            pca_var_threshold=util_model_kwargs.get("pca_var_threshold", 0.95),
            standardize=util_model_kwargs.get("standardize", False),
        )
    else: 
        outcome_model = get_fitted_standard_outcome_model(
            train_X=train_X,
            train_Y=train_Y
        )

    return outcome_model

    
def get_and_fit_util_model(
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
    util_model_name: str,
    outcome_model: Optional[GPyTorchModel] = None,
    bounds: Optional[Tensor] = None,
    train_Y: Optional[Tensor] = None,
    **kwargs,
) -> PairwiseGP:
    r"""
    Instantiate and fit a utility model under the specified **kwargs.
    Args:
        train_pref_outcomes: `n2 x outcome_dim` tensor of outcomes used for 
            preference exploration only; a combination of initial experimental
            outcomes and hypothetical outcomes generated in PE
        train_comps: `n2/2 x 2` tensor of pairwise comparisons of 
            outcomes in train_pref_outcomes
        util_model_name: The name of the utility model to use.
            if "autoencoder": train the utility model and autoencoder jointly
                (through specifying 'fix_vae=False`)
            if "joint_autoencoder": train the utility model under a fixed 
                autoencoder (through specifying 'fix_vae=True`)
            if "pca": train the utility model the PCA subspace learned on
                train_Y as the input space
            otherwise: fit a standard, non-dim-reduced utility model
        outcome_model: an outcome model
        bounds: The bounds of the design space.
        train_Y: The observed experimental outcomes.
    Returns:
        util_model: The fitted utility model.
    """
    
    util_model_kwargs = kwargs.get("util_model_kwargs", {})
    logger.debug(f"Running get_and_fit_util_model, util_model_kwargs: {util_model_kwargs}")

    if util_model_name == "autoencoder":
        util_model, _ = get_fitted_autoencoded_util_model(
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            autoencoder=util_model_kwargs.get("autoencoder", None),
            latent_dims=util_model_kwargs.get("autoencoder_latent_dims", 2), 
            # TODO: should keep this consistent with pca 
            # if it's hard to do in one run, maybe set it in config with reasonable number 
            # from other runs with PCA
            num_joint_train_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_train_epochs", 500
            ),
            num_autoencoder_pretrain_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_pretrain_epochs", 200
            ),
            num_unlabeled_outcomes=util_model_kwargs.get("num_unlabeled_outcomes", 0),
            outcome_model=outcome_model,
            bounds=bounds,
            fix_vae=False
        )
    elif util_model_name == "joint_autoencoder":
        util_model, _ = get_fitted_autoencoded_util_model(
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            autoencoder=util_model_kwargs.get("autoencoder", None),
            latent_dims=util_model_kwargs.get("autoencoder_latent_dims", 2), 
            num_joint_train_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_train_epochs", 500
            ),
            num_autoencoder_pretrain_epochs=util_model_kwargs.get(
                "autoencoder_num_joint_pretrain_epochs", 200
            ),
            num_unlabeled_outcomes=util_model_kwargs.get("num_unlabeled_outcomes", 0),
            outcome_model=outcome_model,
            bounds=bounds,
            fix_vae=True
        )
    elif util_model_name == "pca":
        util_model = get_fitted_pca_util_model(
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            pca_var_threshold=util_model_kwargs.get("pca_var_threshold", 0.95),
            num_unlabeled_outcomes=util_model_kwargs.get("num_unlabeled_outcomes", 0),
            outcome_model=outcome_model,
            bounds=bounds,
            standardize=util_model_kwargs.get("standardize", False),
        )
    elif util_model_name == "standard":
        util_model = get_fitted_standard_util_model(
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
        )
    return util_model


def gen_pe_candidates(
    util_model_name: str,
    pe_gen_strategy: str,
    pref_model: PairwiseGP,
    outcome_model: GPyTorchModel,
    bounds: Tensor,
    outcome_dim: int,
    autoencoder: Optional[Autoencoder] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Generate candidates for the preference exploration stage.
    Args:
        util_model_name: The name of the utility model to use.
        pe_gen_strategy: Strategy, one of "eubo" and "random-f".
        pref_model: The preference model.
        outcome_model: The outcome model.
        bounds: The bounds of the design space.
        outcome_dim: The dimension of the outcome space.
        autoencoder: The autoencoder.
    Returns:
        cand_X: The generated candidate designs.
        cand_Y: The outcomes associated with the candidate designs.
    """

    # EUBO-zeta:
    if pe_gen_strategy == "eubo":
        cand_X, cand_Y, _ = gen_eubo_candidates(
            pref_model=pref_model,
            outcome_model=outcome_model,
            bounds=bounds,
            outcome_dim=outcome_dim,
            use_modified_fixedsinglesamplemodel=True if util_model_name == "pca" else False,
            autoencoder=autoencoder
        )
    elif pe_gen_strategy == "random-f":
        # random-f
        cand_X, cand_Y = gen_random_f_candidates(
            outcome_model=outcome_model,
            bounds=bounds,
            autoencoder=autoencoder
        )
    return cand_X, cand_Y


def run_single_pe_stage(
    train_Y: Tensor,
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
    outcome_model: GPyTorchModel,
    pe_config_info: Dict[str, Any],
    num_pref_iters: int,
    every_n_comps: int,
    problem: MultiObjectiveTestProblem,
    util_func: torch.nn.Module,
    autoencoder: Optional[Autoencoder] = None,
) -> Union[List, Tensor, Tensor, PairwiseGP, LearnedObjective, Any]:
    r"""
    Run one stage of preference exploration.
    Args:
        train_Y: `n1 x outcome_dim` tensor of observed experimental outcomes
        train_pref_outcomes: `n2 x outcome_dim` tensor of outcomes used for 
            preference exploration only; a combination of initial experimental
            outcomes and hypothetical outcomes generated in PE
        train_comps: `n2/2 x 2` tensor of pairwise comparisons of 
            outcomes in train_pref_outcomes
        outcome_model: The outcome model.
        pe_config_info: The config info for the preference exploration stage, 
            which determines the specifics of utility model training.
        num_pref_iters: The number of comparisons to make.
        every_n_comps: The frequency with which to compute the true utility of
            the candidate design that maximizes the current posterior mean utility.
        problem: The true outcome funcion.
        util_func: The true utility function.
        autoencoder: The autoencoder.
    Returns:
        max_val_list: The list of maximum utility values identified throughout
            the PE stage. Length should be `num_pref_iters // every_n_comps`.
        train_pref_outcomes: `(n2+2*num_pref_iters) x outcome_dim` of accumulated
            outcomes used for preference exploration
        train_comps: `(n2//2 + num_pref_iters) x 2` tensor of accumulated pairwise comparisons
        util_model: updated utility model
        pref_obj: optimization objective object based on updated utility model
        autoencoder: autoencoder at the end of the PE stage; may be updated if
            the autoencoder is jointly trained with the utility model
    """
    
    max_val_list = []
    for i in range(num_pref_iters):  # number of batch iterations
        # fit pref model
        util_model_name = pe_config_info["util_model_name"]
        util_model_kwargs = pe_config_info.get("util_model_kwargs", {})
        util_model = get_and_fit_util_model(
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            util_model_name=util_model_name,
            outcome_model=outcome_model,
            bounds=problem.bounds,
            autoencoder=autoencoder,
            util_model_kwargs=util_model_kwargs, 
        )

        if i % every_n_comps == 0:
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
            util_model_name=util_model_name,
            pe_gen_strategy=pe_config_info["pe_gen_strategy"],
            pref_model=util_model,
            outcome_model=outcome_model,
            bounds=problem.bounds,
            outcome_dim=problem.outcome_dim,
        )

        # obtain evaluated data - preferences from DM
        cand_Y = cand_Y.detach().clone()
        if autoencoder is not None:
            cand_Y = autoencoder.decoder(cand_Y).detach()
        cand_comps = gen_comps(util_func(cand_Y))

        # update training data
        train_comps = torch.cat(
            (train_comps, cand_comps + train_pref_outcomes.shape[0])
        )
        train_pref_outcomes = torch.cat((train_pref_outcomes, cand_Y))

    util_model = get_and_fit_util_model(
        train_Y=train_Y,
        train_pref_outcomes=train_pref_outcomes,
        train_comps=train_comps,
        util_model_name=util_model_name,
        outcome_model=outcome_model,
        bounds=problem.bounds,
        autoencoder=autoencoder,
        util_model_kwargs=util_model_kwargs
    )
    # use the same sampler size as in RetrainingBopeExperiment
    sampler = SobolQMCNormalSampler(PREF_MC_SAMPLES)
    pref_obj = LearnedObjective(pref_model=util_model, sampler=sampler)
    return max_val_list, train_pref_outcomes, train_comps, util_model, pref_obj, autoencoder


def run_single_bo_stage(
    train_X: Tensor,
    train_outcomes: Tensor,
    train_comps: Tensor,
    train_pref_outcomes: Tensor,
    outcome_model: GPyTorchModel,
    pref_obj: LearnedObjective,
    pe_config_info: Dict[str, Any],
    num_bo_iters: int,
    bo_batch_size: int,
    problem: MultiObjectiveTestProblem,
    util_func: torch.nn.Module,
    util_list: List[List[float]],
    autoencoder: Optional[Autoencoder] = None,
) -> Union[List, Tensor, Tensor, GPyTorchModel, Any]:
    r"""
    Run one stage of experimentation on candidates from Bayesian optimization.
    Args:
        train_X: `n1 x d` tensor of training data
        train_outcomes: `n1 x outcome_dim` tensor of outcomes
        train_comps: `n2/2 x 2` tensor of pairwise comparisons of outcomes in train_pref_outcomes
        train_pref_outcomes: `n2 x outcome_dim` tensor of outcomes
        outcome_model: outcome model
        pref_obj: optimization objective object based on utility model
        pe_config_info: dictionary of PE config info
        num_bo_iters: number of BO iterations
        bo_batch_size: number of candidates to evaluate at each BO iteration
        problem: The true outcome funcion.
        util_func: The true utility function.
        util_list: A nested list of utility values for the candidates evaluted 
            so far; each sublist corresponds to candidates evaluated in one stage,
            either the initial experimentation stage or a subsequent BO stage.
        autoencoder: autoencoder at the end of the BO stage (doesn't change)
    Returns:
        util_list: updated list storing the utility values of candidate designs
            evaluated so far
        train_X: `(n1+num_bo_iters*q) x d` tensor of accumulated evaluated designs
        train_outcomes: `(n1+num_bo_iters*q) x outcome_dim` tensor of accumulated
            experimental outcomes
        outcome_model: updated outcome model on expanded data
        autoencoder: autoencoder
    """
    
    bo_gen_kwargs = pe_config_info.get("bo_gen_kwargs", {})

    logger.info(f"Running BO, length of current nested list of batch util vals = {len(util_list)}")

    for i in range(num_bo_iters):

        logger.info(f"Running BO iteration {i}")
        logger.debug(f"train X shape: {train_X.shape}")

        acq_func = qNoisyExpectedImprovement(
            model=outcome_model,
            objective=pref_obj,  # should be learned utility objective
            X_baseline=train_X,
            sampler=SobolQMCNormalSampler(OUTCOME_MC_SAMPLES),
            prune_baseline=True,
            cache_root=False,
        )

        # optimize the acquisition function
        candidates, acqf_val = optimize_acqf(
            acq_function=acq_func,
            q=bo_batch_size,
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
        ).squeeze(-1).tolist()  # best obtained util
        util_list.append(true_util)
        logger.debug(f"util_list = {util_list}")
        max_util_so_far = max([max(sublist) for sublist in util_list])
        logger.info(
            f"Finished {i}th BO iteration with best obtained util val = {max_util_so_far}"
        )

        outcome_model = get_and_fit_outcome_model(
            train_X=train_X,
            train_Y=train_outcomes,
            util_model_name=pe_config_info["util_model_name"],
            bounds=problem.bounds,
            autoencoder=autoencoder,
            util_model_kwargs=pe_config_info.get("util_model_kwargs", {})
        )

        # update util model with updated outcome model if retrain_util_model=True
        if bo_gen_kwargs.get("retrain_util_model", False):
            util_model = get_and_fit_util_model(
                train_pref_outcomes=train_pref_outcomes,  # labeled train_pref_outcomes data is not updated in BO stage
                train_comps=train_comps,
                train_Y=train_outcomes,
                util_model_name=pe_config_info["util_model_name"],
                outcome_model=outcome_model,  # updated outcome model (only useful if adding sampled data from outcome model)
                bounds=problem.bounds,
                autoencoder=autoencoder,
                util_model_kwargs=pe_config_info["util_model_kwargs"],
            )
            pref_obj = LearnedObjective(pref_model=util_model)

    return util_list, train_X, train_outcomes, outcome_model, autoencoder
