from typing import Dict, Optional

import gpytorch

import torch
from pref_learning_helpers import (
    check_pref_model_fit,
    fit_pref_model,
    gen_comps,
    gen_exp_cand,
    generate_random_inputs,
    ModifiedFixedSingleSampleModel,
)
from botorch.acquisition import LearnedObjective
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, LCMKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
# TODO: remove unused imports
# TODO: these functions here don't incorporate the changes in 11/17 diff

def run_pref_learn(
    outcome_model,
    train_Y,
    train_comps,
    n_comps,
    problem,
    util_func,
    pe_strategy,
    input_transform=None,
    covar_module=None,
    likelihood=None,
    verbose=False,
):
    """Perform preference exploration with a given PE strategy for n_comps rounds"""
    acqf_vals = []
    for i in range(n_comps):
        if verbose:
            print(f"Running {i+1}/{n_comps} preference learning using {pe_strategy}")

        fit_model_succeed = False
        for _ in range(3):
            try:
                pref_model = fit_pref_model(
                    train_Y,
                    train_comps,
                    input_transform=input_transform,
                    covar_module=covar_module,
                    likelihood=likelihood,
                )
                pref_model_acc = check_pref_model_fit(
                    pref_model, problem=problem, util_func=util_func, n_test=1000
                )
                print("Pref model fitting successful")
                fit_model_succeed = True
                break
            except (ValueError, RuntimeError):
                continue
        if not fit_model_succeed:
            print(
                "fit_pref_model() failed 3 times, stop current call of run_pref_learn()"
            )
            return train_Y, train_comps, None, acqf_vals

        if pe_strategy == "EUBO-zeta":
            # EUBO-zeta
            one_sample_outcome_model = ModifiedFixedSingleSampleModel(
                model=outcome_model, outcome_dim=train_Y.shape[-1]
            )
            acqf = AnalyticExpectedUtilityOfBestOption(
                pref_model=pref_model, outcome_model=one_sample_outcome_model
            )
            found_valid_candidate = False
            for _ in range(3):
                try:
                    cand_X, acqf_val = optimize_acqf(
                        acq_function=acqf,
                        q=2,
                        bounds=problem.bounds,
                        num_restarts=8,
                        raw_samples=64,  # used for intialization heuristic
                        options={"batch_limit": 4},
                    )
                    cand_Y = one_sample_outcome_model(cand_X)
                    acqf_vals.append(acqf_val.item())

                    found_valid_candidate = True
                    break
                except (ValueError, RuntimeError):
                    continue

            if not found_valid_candidate:
                print(
                    "optimize_acqf() failed 3 times for EUBO, stop current call of run_pref_learn()"
                )
                return train_Y, train_comps, None, acqf_vals

        elif pe_strategy == "Random-f":
            # Random-f
            cand_X = generate_random_inputs(problem, n=2)
            cand_Y = outcome_model.posterior(cand_X).sample().squeeze(0)
        else:
            raise RuntimeError("Unknown preference exploration strategy!")

        cand_Y = cand_Y.detach().clone()
        cand_comps = gen_comps(util_func(cand_Y))

        train_comps = torch.cat((train_comps, cand_comps + train_Y.shape[0]))
        train_Y = torch.cat((train_Y, cand_Y))

    return train_Y, train_comps, pref_model_acc, acqf_vals


def find_max_posterior_mean(
    outcome_model: Model,
    train_Y: Tensor,
    train_comps: Tensor,
    problem: torch.nn.Module,
    util_func: torch.nn.Module,
    input_transform: Optional[InputTransform] = None,
    covar_module: Optional[torch.nn.Module] = None,
    num_pref_samples: int = 1,
    verbose=False,
) -> Dict:
    """Helper function that (1) finds experimental design(s)
    maximizing the current posterior mean of the utility, and
    (2) computes the true utility values of these designs.
    Args:
        outcome_model: GP model mapping input to outcome
        train_Y: existing data for outcomes
        train_comps: existing data for comparisons
        problem: TestProblem
        util_func: ground truth utility function (outcome -> utility)
        input_transform: InputTransform to apply on the outcomes
            when fitting utility model using PairwiseGP
        covar_module: covariance module
        verbose: whether to print more details
    Returns:
        within_result: a dictionary logging
            "n_comps": the number of comparisons used for training preference model,
            "util": true utility of selected utility-maximizing candidates
    """

    fit_model_succeed = False
    for _ in range(3):
        try:
            pref_model = fit_pref_model(
                Y=train_Y,
                comps=train_comps,
                input_transform=input_transform,
                covar_module=covar_module,
            )
            print("Pref model fitting successful")
            fit_model_succeed = True
            break
        except (ValueError, RuntimeError):
            continue
    if not fit_model_succeed:
        print(
            "fit_pref_model() failed 3 times, stop current call of find_max_posterior_mean()"
        )
        return {
            "n_comps": train_comps.shape[0],
            "util": None,
        }

    sampler = SobolQMCNormalSampler(num_samples=num_pref_samples)
    pref_obj = LearnedObjective(pref_model=pref_model, sampler=sampler)

    # find experimental candidate(s) that maximize the posterior mean utility
    post_mean_cand_X = gen_exp_cand(
        outcome_model=outcome_model,
        objective=pref_obj,
        problem=problem,
        q=1,
        acqf_name="posterior_mean",
    )
    # evaluate the quality of these candidates by computing their true utilities
    post_mean_util = util_func(problem.evaluate_true(post_mean_cand_X)).item()
    if verbose:
        print(f"True utility of posterior mean utility maximizer: {post_mean_util:.3f}")
    within_result = {
        "n_comps": train_comps.shape[0],
        "util": post_mean_util,
    }
    return within_result
