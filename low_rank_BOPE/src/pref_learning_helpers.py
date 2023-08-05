# This file contains helper functions for doing preference learning
# with custom input / outcome transforms (e.g., PCA) when fitting the
# outcome and utility models.
# Code is mostly inspired by Jerry Lin's implementation here:
# https://github.com/facebookresearch/preference-exploration/blob/main/sim_helpers.py

import sys

sys.path.append('/home/yz685/low_rank_BOPE')

from typing import Dict, Optional, Tuple

import numpy as np
import scipy
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import LearnedObjective
from botorch.acquisition.monte_carlo import (qNoisyExpectedImprovement,
                                             qSimpleRegret)
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model
from botorch.models.pairwise_gp import (PairwiseGP,
                                        PairwiseLaplaceMarginalLogLikelihood)
from botorch.models.transforms.input import (ChainedInputTransform,
                                             InputTransform)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from low_rank_BOPE.src.models import make_modified_kernel
from low_rank_BOPE.src.transforms import InputCenter, PCAInputTransform

# ======= Initial data generation =======


def generate_random_inputs(problem: torch.nn.Module, n: int) -> Tensor:
    r"""Generate n quasi-random Sobol points in the design space.
    Args:
        problem: a TestProblem in Botorch
        n: number of random inputs to generate
    Returns:
        `n x problem input dim` tensor of randomly generated points in problem's input domain
    """
    return (
        draw_sobol_samples(bounds=problem.bounds, n=1, q=n).squeeze(0).to(torch.double)
    )


def gen_comps(
    util_vals: Tensor, comp_noise_type: str = None, comp_noise: float = None
) -> Tensor:
    r"""Create pairwise comparisons.
    Args:
        util_vals: `num_outcomes x 1` tensor of utility values
        comp_noise_type: type of comparison noise to inject, one of {'constant', 'probit'}
        comp_noise: parameter related to probability of making a comparison mistake
    Returns:
        comp_pairs: `(num_outcomes // 2) x 2` tensor showing the preference,
            with the more preferable outcome followed by the other one in each row
    """
    cpu_util = util_vals.cpu()

    comp_pairs = []
    for i in range(cpu_util.shape[0] // 2):
        i1 = i * 2
        i2 = i * 2 + 1
        if cpu_util[i1] > cpu_util[i2]:
            new_comp = [i1, i2]
            util_diff = cpu_util[i1] - cpu_util[i2]
        else:
            new_comp = [i2, i1]
            util_diff = cpu_util[i2] - cpu_util[i1]

        new_comp = torch.tensor(new_comp, device=util_vals.device, dtype=torch.long)
        if comp_noise_type is not None:
            new_comp = inject_comp_error(
                new_comp, util_diff, comp_noise_type, comp_noise
            )
        comp_pairs.append(new_comp)

    comp_pairs = torch.stack(comp_pairs)

    return comp_pairs


def gen_initial_real_data(
    n: int, problem: torch.nn.Module, util_func: torch.nn.Module, comp_noise: float = 0, batch_eval: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # generate (noisy) ground truth data
    r"""Generate noisy ground truth inputs, outcomes, utility values, and comparisons.
    Args:
        n: number of samples to generate
        problem: a TestProblem
        util_func: ground truth utility function (outcome -> utility)
        comp_noise: noise to inject into the comparisons
    Returns:
        X: generated inputs
        Y: generated (noisy) outcomes from evaluating the problem on X
        util_vals: utility values of generated Y
        comps: comparison results for adjacent pairs of util_vals
    """

    X = generate_random_inputs(problem, n).detach()

    if not batch_eval:
        Y_list = []
        for idx in range(len(X)):
            y = problem(X[idx]).detach()
            Y_list.append(y)
        Y = torch.stack(Y_list).squeeze(1)
    else:
        Y = problem(X).detach()

    util_vals = util_func(Y).detach()
    # comps = gen_comps(util_vals, comp_noise_type="constant", comp_noise=comp_noise)
    comps = gen_comps(util_vals)

    return X, Y, util_vals, comps


# ======= Fitting outcome and utility models =======


def fit_outcome_model(X: torch.Tensor, Y: torch.Tensor, **model_kwargs) -> Model:
    r"""Fit outcome model.
    Args:
        X: `num_samples x input_dim` input data
        Y: `num_samples x outcome_dim` outcome data
        model_kwargs: arguments for fitting outcome GP,
            such as outcome_transform, covar_module, likelihood, etc.
    Returns:
        outcome_model: Fitted outcome model mapping input to outcome
    """

    outcome_model = SingleTaskGP(train_X=X, train_Y=Y, **model_kwargs)

    mll_outcome = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
    # fit_gpytorch_model(mll_outcome)
    fit_gpytorch_mll(mll_outcome)

    return outcome_model


def fit_pref_model(Y: Tensor, comps: Tensor, **model_kwargs) -> Model:
    r"""
    Fit a preference / utility GP model for the mapping from outcome to scalar utility value
    Args:
        Y: `num_outcome_samples x outcome_dim` tensor of outcomes
        comps: `num_comparisons x 2` tensor of comparisons;
                comps[i] is a noisy indicator suggesting the utility value
                of comps[i, 0]-th is greater than comps[i, 1]-th
        model_kwargs: arguments for fitting utility GP,
            such as outcome_transform, covar_module, likelihood, jitter, etc.
    Returns:
        util_model: a GP model mapping outcome to utility
    """

    util_model = PairwiseGP(datapoints=Y, comparisons=comps, **model_kwargs)

    mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)
    # fit_gpytorch_model(mll_util)
    fit_gpytorch_mll(mll_util)

    return util_model


def fit_util_models(train_Y, comps, util_vals, input_transform, covar_module):
    """ 
    Fit utility model given (1) comparisons (2) ground truth utility values.
    Return the two fitted GPs.
    (In practice, fitting model (2) is usually not feasible.)
    (This function is developed for easier ablation testing)
    """
    util_model_rel = fit_pref_model(
        train_Y, 
        comps, 
        input_transform = input_transform, 
        covar_module = covar_module
    )
    util_model_abs = SingleTaskGP(
        train_Y, 
        util_vals, 
        input_transform = input_transform, 
        covar_module = covar_module
    )
    mll = ExactMarginalLogLikelihood(util_model_abs.likelihood, util_model_abs)
    fit_gpytorch_mll(mll)

    return util_model_rel, util_model_abs

def fit_util_models_wrapper(
    train_Y, comps, util_vals, method, axes_dict = None, 
    modify_kernel = False, a=0.2, b=5
):
    """ 
    Fit utility models for different methods (st, pca, pcr) and potentially a 
    set of different axes in `axes_dict`. If specified, also modify the hyperpriors 
    of the input covar_module based on the supplied parameter value `a` and `b`.
    Return the fitted models in a dictionary. The suffix '_rel' means the model
    is fit on pairwise comparisons; the suffix '_abs' means the model is fit on 
    ground truth utility values.
    (This function is developed for easier ablation testing)
    """
    input_transform = None
    models_dict = {}
    if method in ("pca", "pcr"):
        for axes_label, axes in axes_dict.items():
            latent_dim = axes.shape[0]
            input_transform = ChainedInputTransform(
                        **{
                            "center": InputCenter(train_Y.shape[-1]),
                            "pca": PCAInputTransform(axes.to(torch.double)),
                        }
                    )
            covar_module = make_modified_kernel(
                ard_num_dims=latent_dim, a=a, b=b) if modify_kernel else None

            util_model_rel, util_model_abs = fit_util_models(
                train_Y, comps, util_vals, input_transform, covar_module)
            models_dict[method+'_'+axes_label+'_rel'] = util_model_rel
            models_dict[method+'_'+axes_label+'_abs'] = util_model_abs
    
    elif method == "st":
        covar_module = make_modified_kernel(ard_num_dims=train_Y.shape[-1]) if modify_kernel else None
        input_transform = None
        util_model_rel, util_model_abs = fit_util_models(
                train_Y, comps, util_vals, input_transform, covar_module)
        models_dict[method+'_rel'] = util_model_rel
        models_dict[method+'_abs'] = util_model_abs
    
    return models_dict

# ======= Data generation within preference learning =======


def generate_random_exp_data(problem: torch.nn.Module, n: int, batch_eval: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Generate n observations of experimental designs and outcomes.
    Args:
        problem: a TestProblem
        n: number of samples
    Returns:
        X: `n x problem input dim` tensor of sampled inputs
        Y: `n x problem outcome dim` tensor of noisy evaluated outcomes at X
    """
    X = generate_random_inputs(problem, n).detach()
    if not batch_eval:
        Y_list = []
        for idx in range(len(X)):
            # print('X[idx]', idx, X[idx])
            # print('problem(X[idx])', problem(X[idx]))
            y = problem(X[idx]).detach()
            Y_list.append(y)
        Y = torch.stack(Y_list).squeeze(1)
    else:
        Y = problem(X).detach()

    print('generated outcomes: ', Y)

    return X, Y


def generate_random_pref_data(
    problem: torch.nn.Module, outcome_model: Model, n: int, util_func: torch.nn.Module
) -> Tuple[Tensor, Tensor]:
    """Generate pairwise comparisons between 2n points,
    where `2n` inputs are generated randomly and `2n` outcomes are sampled
    from the posterior of the outcome model. Then, the `n` adjacent pairs
    of the outcomes are compared according to the given
    ground-truth utility function.
    Args:
        problem: TestProblem
        outcome_model: GP mapping input to outcome
        n: number of comparison pairs to generate
        util_func: ground truth utility function (outcome -> utility)
    Returns:
        Y: outcomes generated from the posterior of outcome_model
        comps: pairwise comparisons of adjacent pairs in Y
    """
    X = generate_random_inputs(problem, 2 * n)
    print('X shape in gen_random_pref_data: ', X.shape)
    # Y = outcome_model.posterior(X).sample().squeeze(0)
    Y = outcome_model.posterior(X).rsample().squeeze(0).detach()
    print('Y shape in gen_random_pref_data: ', Y.shape)
    util = util_func(Y)
    print('util in gen_random_pref_data: ', util)
    comps = gen_comps(util)
    return Y, comps


# ======= Candidate and outcome generation =======


def gen_exp_cand(
    model: Model,
    problem: torch.nn.Module,
    q: int,
    acqf_name: str,
    seed: int,
    objective: Optional[MCAcquisitionObjective] = None,
    X: Optional[Tensor] = None,
    sampler_num_outcome_samples: int = 128,
    num_restarts: int = 8,
    raw_samples: int = 64,
    batch_limit: int = 4,
) -> Tensor:
    """Given an outcome model and an objective, generate q experimental candidates
    using a specified acquisition function.
    Args:
        outcome_model: GP model mapping input to outcome
        objective: MC objective mapping outcome to utility
            if the objective is not specified, `model` maps from input to utility 
        problem: a TestProblem
        q: number of candidates to generate
        acqf_name: name of acquisition function, one of {'qNEI', 'posterior_mean'}
        X: `num_outcome_samples x input_dim` current training data
        sampler_num_outcome_samples: number of base samples in acq function's sampler
        num_restarts: number of starting points for multi-start acqf optimization
        raw_samples: number of samples for initializing acqf optimization
        batch_limit: the limit on batch size in gen_candidates_scipy() within optimize_acqf()
    Returns:
        candidates: `q x problem input dim` generated candidates
    """
    sampler = SobolQMCNormalSampler(sampler_num_outcome_samples)
    if acqf_name == "qNEI":
        # generate experimental candidates with qNEI/qNEIUU
        acq_func = qNoisyExpectedImprovement(
            model=model,
            objective=objective,
            X_baseline=X,
            sampler=sampler,
            prune_baseline=True,
            cache_root=False,
        )
    elif acqf_name == "posterior_mean":
        # generate experimental candidates with maximum posterior mean
        acq_func = qSimpleRegret(
            model=model,
            sampler=sampler,
            objective=objective,
        )
    else:
        raise RuntimeError("Unknown acquisition function name!")

    # optimize the acquisition function
    candidates, acqf_val = optimize_acqf(
        acq_function=acq_func,
        q=q,
        bounds=problem.bounds, # this is the input bounds
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": batch_limit, "seed": seed},
        sequential=True,
    )
    return candidates, acqf_val


class ModifiedFixedSingleSampleModel(DeterministicModel):
    r"""
    A deterministic model defined by a single sample `w`.

    Given a base model `f` and a fixed sample `w`, the model always outputs

        y = f_mean(x) + f_stddev(x) * w

    We assume the outcomes are uncorrelated here.

    This is modified from FixedSingleSampleModel to handle dimensionality reduction.
    For models with dim reduction, model.num_outputs is the reduced outcome dimension,
    whereas we want w to be in the original outcome dimension.
    In this modification, we define self.w within forward() rather than __init__(),
    where we fix the dimensionality of w to be posterior(X).event_shape[-1].
    """

    def __init__(
        self, model: Model, outcome_dim: int, w: Optional[torch.Tensor] = None
    ) -> None:
        r"""
        Args:
            model: The base model.
            outcome_dim: dimensionality of the outcome space
            w: A 1-d tensor with length = outcome_dim.
                If None, draw it from a standard normal distribution.
        """
        super().__init__()
        self.model = model
        self.w = torch.randn(outcome_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        post = self.model.posterior(X)

        # return post.mean + post.variance.sqrt() * self.w.to(X)
        # adding jitter to avoid numerical issues
        return post.mean + torch.sqrt(post.variance + 1e-8) * self.w.to(X)


def run_pref_learn(
    outcome_model,
    train_Y,
    train_comps,
    n_comps,
    problem,
    util_func,
    pe_strategy,
    seed,
    input_transform=None,
    covar_module=None,
    likelihood=None,
    verbose=False,
    batch_eval=True
):
    """Perform preference exploration with a given PE strategy for n_comps rounds"""
    acqf_vals = []
    for i in range(n_comps):
        if verbose:
            print(f"Running {i+1}/{n_comps} preference learning using {pe_strategy}")

        fit_model_succeed = False
        pref_model_acc = None
        for _ in range(3):
            try:
                pref_model = fit_pref_model(
                    train_Y,
                    train_comps,
                    input_transform=input_transform,
                    covar_module=covar_module,
                    likelihood=likelihood,
                )
                # TODO: commented out to accelerate things
                # pref_model_acc = check_pref_model_fit(
                #     pref_model, problem=problem, util_func=util_func, n_test=1000, batch_eval=batch_eval
                # )
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
                        options={"batch_limit": 4, "seed": seed},
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
            cand_Y = outcome_model.posterior(cand_X).rsample().squeeze(0).detach()
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
    pref_model = fit_pref_model(
        Y=train_Y,
        comps=train_comps,
        input_transform=input_transform,
        covar_module=covar_module,
    )
    sampler = SobolQMCNormalSampler(num_pref_samples)
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


# ==========
# TODO: use this later
def inject_comp_error(comp, util_diff, comp_noise_type, comp_noise):

    std_norm = torch.distributions.normal.Normal(
        torch.zeros(1, dtype=util_diff.dtype, device=util_diff.device),
        torch.ones(1, dtype=util_diff.dtype, device=util_diff.device),
    )

    if comp_noise_type == "constant":
        comp_error_p = comp_noise
    elif comp_noise_type == "probit":
        comp_error_p = 1 - std_norm.cdf(util_diff / comp_noise)
    else:
        raise UnsupportedError(f"Unsupported comp_noise_type: {comp_noise_type}")

    # with comp_error_p probability to make a comparison mistake
    flip_rand = torch.rand(util_diff.shape).to(util_diff)
    to_flip = flip_rand < comp_error_p
    flipped_comp = comp.clone()
    if len(flipped_comp.shape) > 1:
        assert (util_diff >= 0).all()
        # flip tensor
        flipped_comp[to_flip, 0], flipped_comp[to_flip, 1] = (
            comp[to_flip, 1],
            comp[to_flip, 0],
        )
    else:
        assert util_diff > 0
        # flip a single pair
        if to_flip:
            flipped_comp[[0, 1]] = flipped_comp[[1, 0]]
    return flipped_comp


def find_true_optimal_utility(
    problem: torch.nn.Module, 
    util_func: torch.nn.Module, 
    n: int,
    maximize: bool = True
):
    r"""
    Find the optimal utility value, i.e., max_x (util_func(problem(x))) across
    the domain through taking a large number of samples.
    Args:
        problem: a TestProblem that maps designs to outcomes
        util_func: maps outcomes to scalar utility
        n: number of evalutions
        maximize: boolean for whether to maximize (if False, minimize)
    """

    meta_batch_size = 20000 // problem.dim
    num_meta_batches = n // meta_batch_size + 1
    best_util_vals = []

    for _ in range(num_meta_batches):
        _, _, util_vals, _ = gen_initial_real_data(
            n=meta_batch_size,
            problem=problem,
            util_func=util_func,
            comp_noise=0,
            batch_eval=True
        )
        if maximize:
            best_util_vals.append(torch.max(util_vals).item())
        else:
            best_util_vals.append(torch.min(util_vals).item())
    
    if maximize:
        return np.max(best_util_vals)
    else:
        return np.min(best_util_vals)


# TODO: fix this later, using scipy but not optimizing at all
# def find_true_optimal_utility_scipy(
#     problem: torch.nn.Module, 
#     util_func: torch.nn.Module, 
#     maximize: bool = True
# ):
#     r"""
#     Find the true optimal utility value, i.e., max_x (util_func(problem(x)))
#     Args:
#         problem: a TestProblem that maps designs to outcomes
#         util_func: maps outcomes to scalar utility
#         maximize: boolean for whether to maximize (if False, minimize)
#     """

#     print("problem._bounds: ", problem._bounds)
#     if problem._bounds.shape[0] == 2:
#         bounds = list(map(tuple, torch.transpose(problem._bounds, -2, -1).numpy()))
#         x0 = np.array(problem._bounds.to(torch.double).mean(dim = 0))
#     else:
#         bounds = list(map(tuple, problem._bounds.numpy()))
#         x0 = np.array(problem._bounds.to(torch.double).mean(dim = 1))
    
#     print('x0: ', x0)
#     print("bounds: ", bounds)

#     # define function to be minimized using scipy.optimize.minimize
#     def util_of_design(x):
#         # x is a 1d array with shape (d, )
#         # convert it to tensor and compute its utility
#         x_tensor = torch.Tensor(x).unsqueeze(0)
#         outcomes = problem.evaluate_true(x_tensor)
#         util = util_func(outcomes)
#         if maximize:
#             util *= -1
    
#         return util.item()
    
#     res = scipy.optimize.minimize(util_of_design, x0, bounds = bounds, options = {'eps': 1e-3})
#     print(res)

#     if res.success:
#         print('best design: ', res.x, 'best utility: ', res.fun)
#         return [res.x, res.fun]
#     else:
#         print('Failed to find the true optimal utility value')
#         return None