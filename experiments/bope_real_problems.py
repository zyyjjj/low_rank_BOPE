#!/usr/bin/env python3
import copy
import random
import time
from typing import Any, Dict, List, NamedTuple

import fblearner.flow.api as flow
import gpytorch
import numpy as np
import torch
from low-rank-BOPE.src.pref_learning_helpers import (
    check_outcome_model_fit,
    check_pref_model_fit,
    find_max_posterior_mean, # TODO: later see if we want the error-handled version
    fit_outcome_model,
    fit_pref_model, # TODO: later see if we want the error-handled version
    gen_exp_cand,
    generate_random_exp_data,
    generate_random_pref_data,
)
from low-rank-BOPE.src.real_problems import (
    AdaptedOSY,
    CarCabDesign,
    DTLZ2,
    LinearUtil,
    NegativeVehicleSafety,
    NegDist,
    OSYSigmoidConstraintsUtil,
    PiecewiseLinear,
    probit_noise_dict,
    AugmentedProblem,
    problem_setup_augmented
)
from low-rank-BOPE.src.transforms import (
    generate_random_projection,
    InputCenter,
    LinearProjectionInputTransform,
    LinearProjectionOutcomeTransform,
    PCAInputTransform,
    PCAOutcomeTransform,
    SubsetOutcomeTransform,
)
from botorch.acquisition.objective import GenericMCObjective, LearnedObjective
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    Normalize,
)

from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem

from low-rank-BOPE.src.diagnostics import (
    empirical_max_outcome_error,
    empirical_max_util_error,
    mc_max_outcome_error,
    mc_max_util_error,
)
from low-rank-BOPE.src.models import make_modified_kernel, MultitaskGPModel
from gpytorch.kernels import LCMKernel, MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

N_BOPE_REPS = 30
PCA_VAR_THRESHOLD = 0.95
AUGMENTED_DIMS_NOISE = 0.01
MIN_STD = 100000
N_TEST = 1000

tkwargs = {"dtype": torch.double}


# class for saving experiment data
class OneRun(NamedTuple):
    exp_candidate_results: List[Dict[str, Any]]
    within_session_results: List[Dict[str, Any]]


# problem_setup_names = [
#     "vehiclesafety_5d3d_piecewiselinear_20d",
#     "carcabdesign_7d9d_piecewiselinear_20d",
#     "carcabdesign_7d9d_linear_20d",
# ]

problem_setup_names = [
    "vehiclesafety_5d3d_piecewiselinear_3c",
    "carcabdesign_7d9d_piecewiselinear_3c",
    "carcabdesign_7d9d_linear_3c",
    # "osy_6d8d_piecewiselinear_3c",
]

BASE_CONFIG = {
    "initial_experimentation_batch": 16,
    "n_check_post_mean": 13,
    "every_n_comps": 3,
}


# workflow
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    # test_problems: Dict[str, Tuple[Any]] = test_problems,
    problem_setup_names: List[str] = problem_setup_names,
    n_trials: int = N_BOPE_REPS,
) -> Dict[str, List[OneRun]]:

    all_results = {}

    for problem_setup_name in problem_setup_names:

        input_dim, outcome_dim, problem, _, util_func, _, _ = problem_setup_augmented(
            problem_setup_name, augmented_dims_noise=AUGMENTED_DIMS_NOISE, **tkwargs
        )
        config = copy.deepcopy(BASE_CONFIG)
        config["input_dim"] = input_dim
        config["outcome_dim"] = outcome_dim

        all_results[problem_setup_name] = [
            run_one_trial(
                problem=problem,
                util_func=util_func,
                trial_idx=i,
                config=config,
                **tkwargs,
            )
            for i in range(N_BOPE_REPS)
        ]

    return all_results


# each trial is one BOPE run for a particular problem, utility function
@flow.flow_async()
@flow.typed()
def run_one_trial(
    problem: torch.nn.Module,
    util_func: torch.nn.Module,
    trial_idx: int,
    config: Dict[str, Any],
    verbose=True,
    **tkwargs,
) -> OneRun:

    print(f"Running trial number {trial_idx} for the problem config:")
    print(problem, util_func, config)

    torch.manual_seed(trial_idx)
    np.random.seed(trial_idx)

    within_session_results = []
    exp_candidate_results = []

    # ======= Experimentation stage =======
    # initial exploration batch

    X, Y = generate_random_exp_data(problem, config["initial_experimentation_batch"])
    print("X,Y dtypes", X.dtype, Y.dtype)

    # create dictionary storing the outcome-transforms, the input-transforms,
    # and the covar_module (for PairwiseGP) for each method
    transforms_covar_dict = {
        # Baseline 0: Single-task GP with no dimensionality reduction
        "st": {
            "outcome_tf": Standardize(config["outcome_dim"]),
            "input_tf": Normalize(config["outcome_dim"]),
            "covar_module": make_modified_kernel(ard_num_dims=config["outcome_dim"]),
        },
        "pca": {
            "outcome_tf": ChainedOutcomeTransform(
                **{
                    "standardize": Standardize(
                        config["outcome_dim"],
                        min_stdv=MIN_STD,  # TODO: try standardize again
                    ),
                    # "pca": PCAOutcomeTransform(num_axes=config["lin_proj_latent_dim"]),
                    "pca": PCAOutcomeTransform(
                        variance_explained_threshold=PCA_VAR_THRESHOLD
                    ),
                }
            ),
        },
        "mtgp": {
            "outcome_tf": Standardize(config["outcome_dim"]),
            "input_tf": Normalize(config["outcome_dim"]),
            "covar_module": make_modified_kernel(ard_num_dims=config["outcome_dim"]),
        },
        "lmc": {
            "outcome_tf": Standardize(config["outcome_dim"]),
            "input_tf": Normalize(config["outcome_dim"]),
            "covar_module": make_modified_kernel(ard_num_dims=config["outcome_dim"]),
        },
    }

    for method in [
        "st",
        "pca",
        "random_linear_proj",
        "random_subset",
        "lmc"
        # "mtgp",
    ]:

        print(f"=====Running method {method}=====")

        lin_proj_latent_dim = 1  # define variable, placeholder

        if method == "mtgp":
            outcome_model = KroneckerMultiTaskGP(
                X,
                Y,
                outcome_transform=copy.deepcopy(
                    transforms_covar_dict[method]["outcome_tf"]
                ),
                rank=3,
            )
            icm_mll = ExactMarginalLogLikelihood(
                outcome_model.likelihood, outcome_model
            )
            fit_gpytorch_model(icm_mll)

        elif method == "lmc":
            lcm_kernel = LCMKernel(
                base_kernels=[MaternKernel()] * lin_proj_latent_dim,
                num_tasks=config["outcome_dim"],
                rank=1,
            )
            lcm_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=config["outcome_dim"]
            )
            outcome_model = MultitaskGPModel(
                X,
                Y,
                lcm_likelihood,
                num_tasks=config["outcome_dim"],
                multitask_kernel=lcm_kernel,
                outcome_transform=copy.deepcopy(
                    transforms_covar_dict[method]["outcome_tf"]
                ),
            ).to(**tkwargs)
            lcm_mll = ExactMarginalLogLikelihood(
                outcome_model.likelihood, outcome_model
            )
            fit_gpytorch_mll(lcm_mll)

        else:
            outcome_model = fit_outcome_model(
                X,
                Y,
                outcome_transform=transforms_covar_dict[method]["outcome_tf"],
            )

        if method == "pca":

            axes_learned = outcome_model.outcome_transform["pca"].axes_learned

            transforms_covar_dict["pca"]["input_tf"] = ChainedInputTransform(
                **{
                    # "standardize": InputStandardize(config["outcome_dim"]),
                    # TODO: was trying standardize again
                    "center": InputCenter(config["outcome_dim"]),
                    "pca": PCAInputTransform(axes=axes_learned),
                }
            )

            lin_proj_latent_dim = axes_learned.shape[0]

            print(
                f"amount of variance explained by {lin_proj_latent_dim} axes: {outcome_model.outcome_transform['pca'].PCA_explained_variance}"
            )

            # here we first see how many latent dimensions PCA learn
            # then we create a random linear projection mapping to the same dimensionality

            transforms_covar_dict["pca"]["covar_module"] = make_modified_kernel(
                ard_num_dims=lin_proj_latent_dim
            )

            random_proj = generate_random_projection(
                config["outcome_dim"], lin_proj_latent_dim, **tkwargs
            )
            transforms_covar_dict["random_linear_proj"] = {
                "outcome_tf": LinearProjectionOutcomeTransform(random_proj),
                "input_tf": LinearProjectionInputTransform(random_proj),
                "covar_module": make_modified_kernel(ard_num_dims=lin_proj_latent_dim),
            }
            random_subset = random.sample(
                range(config["outcome_dim"]), lin_proj_latent_dim
            )
            transforms_covar_dict["random_subset"] = {
                "outcome_tf": SubsetOutcomeTransform(
                    outcome_dim=config["outcome_dim"], subset=random_subset
                ),
                "input_tf": FilterFeatures(
                    feature_indices=torch.Tensor(random_subset).to(int)
                ),
                "covar_module": make_modified_kernel(ard_num_dims=lin_proj_latent_dim),
            }

        # ======= Preference exploration stage =======
        # initialize the preference model with comparsions
        # between pairs of outcomes estimated using random design points

        init_train_Y, init_train_comps = generate_random_pref_data(
            problem, outcome_model, n=1, util_func=util_func
        )  # TODO: should we increase n?

        # Perform preference exploration using either Random-f or EUBO-zeta
        for pe_strategy in ["EUBO-zeta", "Random-f"]:

            time_start = time.time()

            print("Running PE strategy " + pe_strategy)
            train_Y, train_comps = init_train_Y.clone(), init_train_comps.clone()
            # get the true utility of the candidate that maximizes the posterior mean utility
            # this tells us the quality of the candidate we select
            within_result = find_max_posterior_mean(
                outcome_model=outcome_model,
                train_Y=train_Y,
                train_comps=train_comps,
                problem=problem,
                util_func=util_func,
                input_transform=copy.deepcopy(
                    transforms_covar_dict[method]["input_tf"]
                ),
                covar_module=copy.deepcopy(
                    transforms_covar_dict[method]["covar_module"]
                ),
            )
            within_result.update(
                {"run_id": trial_idx, "pe_strategy": pe_strategy, "method": method}
            )
            within_session_results.append(within_result)

            for j in range(config["n_check_post_mean"]):
                train_Y, train_comps, pref_model_acc, acqf_val = run_pref_learn(
                    outcome_model=outcome_model,
                    train_Y=train_Y,
                    train_comps=train_comps,
                    n_comps=config["every_n_comps"],
                    problem=problem,
                    util_func=util_func,
                    pe_strategy=pe_strategy,
                    input_transform=copy.deepcopy(
                        transforms_covar_dict[method]["input_tf"]
                    ),
                    covar_module=copy.deepcopy(
                        transforms_covar_dict[method]["covar_module"]
                    ),
                    verbose=verbose,
                )
                if verbose:
                    print(
                        f"Checking posterior mean after {(j+1) * config['every_n_comps']} comps using PE strategy {pe_strategy}"
                    )
                    print(
                        f"Pref model accuracy {pref_model_acc}, EUBO acqf value {acqf_val}"
                    )
                within_result = find_max_posterior_mean(
                    outcome_model,
                    train_Y,
                    train_comps,
                    problem=problem,
                    util_func=util_func,
                    input_transform=copy.deepcopy(
                        transforms_covar_dict[method]["input_tf"]
                    ),
                    covar_module=copy.deepcopy(
                        transforms_covar_dict[method]["covar_module"]
                    ),
                    verbose=verbose,
                )
                within_result.update(
                    {
                        "run_id": trial_idx,
                        "pe_strategy": pe_strategy,
                        "method": method,
                        "pref_model_acc": pref_model_acc,
                        "acqf_val": acqf_val,
                    }
                )
                within_session_results.append(within_result)

            # ======= Second experimentation stage =======
            # generate an additional batch of experimental evaluations
            # with the learned preference model and qNEIUU

            fit_pref_model_succeed = False
            for _ in range(3):
                try:
                    pref_model = fit_pref_model(
                        Y=train_Y,
                        comps=train_comps,
                        input_transform=copy.deepcopy(
                            transforms_covar_dict[method]["input_tf"]
                        ),
                        covar_module=copy.deepcopy(
                            transforms_covar_dict[method]["covar_module"]
                        ),
                    )
                    fit_pref_model_succeed = True
                    break
                except (ValueError, RuntimeError):
                    continue
            # diagnostic: model fit
            if fit_pref_model_succeed:
                outcome_model_mse = check_outcome_model_fit(
                    outcome_model=outcome_model, problem=problem, n_test=N_TEST
                )
                print(f"outcome model mse: {outcome_model_mse}")
                pref_model_acc = check_pref_model_fit(
                    pref_model, problem=problem, util_func=util_func, n_test=N_TEST
                )
                print(f"final pref model acc: {pref_model_acc}")
                sampler = SobolQMCNormalSampler(num_samples=1)
                pref_obj = LearnedObjective(pref_model=pref_model, sampler=sampler)
                exp_cand_X = gen_exp_cand(
                    outcome_model, pref_obj, problem=problem, q=1, acqf_name="qNEI", X=X
                )  # noqa
                qneiuu_util = util_func(problem.evaluate_true(exp_cand_X)).item()
                print(
                    f"{method}-{pe_strategy} qNEIUU candidate utility: {qneiuu_util:.5f}"
                )
            else:
                qneiuu_util = None
                pref_model_acc = None

            time_consumed = time.time() - time_start

            # log the true utility of the selected candidate experimental design
            exp_result = {
                "candidate": exp_cand_X,
                "candidate_util": qneiuu_util,
                "method": method,
                "strategy": pe_strategy,
                "run_id": trial_idx,
                "time_consumed": time_consumed,
                "outcome_model_mse": outcome_model_mse,
                "pref_model_acc": pref_model_acc,
            }
            if method == "pca":
                print("Updating extra recovery diagnostics for PCA")
                exp_result["empirical_max_outcome_error"] = empirical_max_outcome_error(
                    Y, axes_learned  # Y here is the initial experiment data
                )
                exp_result["mc_max_outcome_error"] = mc_max_outcome_error(
                    problem, axes_learned, N_TEST
                )
                exp_result["empirical_max_util_error"] = empirical_max_util_error(
                    train_Y, axes_learned, util_func
                )
                exp_result["mc_max_util_error"] = mc_max_util_error(
                    problem, axes_learned, util_func, N_TEST
                )
                exp_result["num_axes"] = lin_proj_latent_dim
            exp_candidate_results.append(exp_result)

        # Baseline 3: Oracle / Assume knowing true utility
        # -- this depends on the outcome model, which in turn depends on the method
        true_obj = GenericMCObjective(util_func)
        true_obj_cand_X = gen_exp_cand(
            outcome_model, true_obj, problem=problem, q=1, acqf_name="qNEI", X=X
        )
        true_obj_util = util_func(problem.evaluate_true(true_obj_cand_X)).item()
        print(f"True objective utility: {true_obj_util:.5f}")
        exp_result = {
            "candidate": true_obj_cand_X,
            "candidate_util": true_obj_util,
            "method": method,
            "strategy": "True Utility",
            "run_id": trial_idx,
        }
        exp_candidate_results.append(exp_result)

    # Baseline 4: Random experiment
    # this does not depend on the method
    _, random_Y = generate_random_exp_data(problem, 1)
    random_util = util_func(random_Y).item()
    print(f"Random experiment utility: {random_util:.5f}")
    exp_result = {
        "candidate_util": random_util,
        "strategy": "Random Experiment",
        "run_id": trial_idx,
    }
    exp_candidate_results.append(exp_result)

    return OneRun(
        exp_candidate_results=exp_candidate_results,
        within_session_results=within_session_results,
    )
