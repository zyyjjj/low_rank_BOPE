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
    # find_max_posterior_mean,
    fit_outcome_model,
    # fit_pref_model, # modified, used fit_gpytorch_mll()
    gen_comps,
    gen_exp_cand,
    generate_random_exp_data,
    generate_random_inputs,
    generate_random_pref_data,
    ModifiedFixedSingleSampleModel,
)
from low-rank-BOPE.src.real_problems import LinearUtil
from low-rank-BOPE.src.synthetic_problem import generate_principal_axes, PCATestProblem
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
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model
from botorch.models.model import Model
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    Normalize,
)
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from low-rank-BOPE.src.diagnostics import (
    empirical_max_outcome_error,
    empirical_max_util_error,
    mc_max_outcome_error,
    mc_max_util_error,
)
from low-rank-BOPE.src.models import make_modified_kernel, MultitaskGPModel
from low-rank-BOPE.experiments.synthetic_test_problem_configs import (
    test_configs_low_latent_dim,
    test_configs_new_moderate_scaling,
    test_configs_new_scaling,
    test_configs_outcome_model_fit,
)
from gpytorch.kernels import LCMKernel, MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

N_BOPE_REPS = 30
PCA_VAR_THRESHOLD = 0.99
MIN_STD = 100000
# ALPHAS = [0.2, 0.8]
# ALPHAS = [0.4, 0.6]
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
# ALPHAS = [1.0]
tkwargs = {"dtype": torch.double}
N_TEST = 1000
N_FINAL_CAND = 1

# specify test simulation config
# test_configs = {"config_1": test_configs_low_latent_dim["config_1"]}
# test_configs = test_configs_moderate_latent_dim
# test_configs = test_configs_low_latent_dim
# test_configs = {"config_1": test_configs_outcome_model_fit["config_1"]}
# test_configs = test_configs_outcome_model_fit
# test_configs = {"config_1": test_configs_new_scaling["config_1"]}
# test_configs = {"config_1": test_configs_new_moderate_scaling["config_1"]}
test_configs = test_configs_new_moderate_scaling

# class for saving experiment data
class OneRun(NamedTuple):
    exp_candidate_results: List[Dict[str, Any]]
    within_session_results: List[Dict[str, Any]]


# workflow
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    configs: Dict[str, Dict[str, Any]] = test_configs,
    n_trials: int = N_BOPE_REPS,
) -> Dict[str, List[OneRun]]:

    all_results = {}

    for key, config in configs.items():
        problem, full_axes, ground_truth_axes = make_problem(
            config=config, np_seed=1234, torch_seed=1234, **tkwargs
        )
        config["ground_truth_axes"] = ground_truth_axes

        for alpha in ALPHAS:

            coeffs = make_controlled_coeffs(
                full_axes=full_axes,
                latent_dim=config["latent_dim"],
                alpha=alpha,
                # n_reps=n_trials,
                n_reps=1,
                **tkwargs,
            )

            all_results[key + "_" + str(alpha)] = [
                run_one_trial(
                    problem=problem,
                    util_func=LinearUtil(coeffs[0]),  # changed from coeffs[i]
                    trial_idx=i,
                    config=config,
                )
                for i in range(n_trials)
            ]

    return all_results


def make_controlled_coeffs(full_axes, latent_dim, alpha, n_reps, **tkwargs):

    """
    Create norm-1 vectors with a specified norm in the subspace
    spanned by a specified set of axes.
    This is used here to generate coefficients for the linear
    utility function, with a controlled level of (dis)alignment
    with the subspace for outcomes.
    Args:
        full_axes: `outcome_dim x outcome_dim` orthonormal matrix,
            each row representing an axis
        latent_dim: latent dimension
        alpha: a number in [0,1] specifying the desired norm of the
            projection of each coefficient onto the space
            spanned by the first `latent_dim` rows of full_axes
        n_reps: number of coefficients to generate
    Returns:
        `n_reps x outcome_dim` tensor, with each row being a linear
            utility function coefficient
    """

    k = full_axes.shape[0]

    # first generate vectors lying in the latent space with norm alpha
    # z1 is `latent_dim x n_reps`, V1 is `outcome_dim x latent_dim`
    z1 = torch.randn((latent_dim, n_reps)).to(**tkwargs)
    V1 = torch.transpose(full_axes[:latent_dim], -2, -1).to(**tkwargs)
    Vz1 = torch.matmul(V1, z1)
    c_proj = torch.nn.functional.normalize(Vz1, dim=0) * alpha

    if alpha == 1:
        return torch.transpose(c_proj, -2, -1)

    else:
        # then generate vectors orthogonal to the latent space
        # with norm sqrt(1-alpha^2)
        # z2 is `(outcome_dim - latent_dim) x n_reps`
        # V2 is `outcome_dim x (outcome_dim - latent_dim)`
        z2 = torch.randn((k - latent_dim, n_reps)).to(**tkwargs)
        V2 = torch.transpose(full_axes[: (k - latent_dim)], -2, -1).to(**tkwargs)
        Vz2 = torch.matmul(V2, z2)
        c_orth = torch.nn.functional.normalize(Vz2, dim=0) * np.sqrt(1 - alpha**2)

        return torch.transpose(c_proj + c_orth, -2, -1)


# each trial is one BOPE run for a particular problem, utility function
@flow.flow_async()
@flow.typed()
def run_one_trial(
    problem: torch.nn.Module,
    util_func: torch.nn.Module,
    trial_idx: int,
    config: Dict[str, Any],
    verbose=True,
) -> OneRun:

    print(f"Running trial number {trial_idx} for the problem config:")
    print(config)

    torch.manual_seed(trial_idx)
    np.random.seed(trial_idx)

    within_session_results = []
    exp_candidate_results = []

    # ======= Experimentation stage =======
    # initial exploration batch

    X, Y = generate_random_exp_data(problem, config["initial_experimentation_batch"])

    initial_batch_max_util = util_func(problem.evaluate_true(X)).max().item()

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
                        m=config["outcome_dim"],
                        min_stdv=MIN_STD,
                        # TODO: here set the min-sd-to-not-scale bound to be very large
                        # so that we only center, not standardize
                    ),
                    "pca": PCAOutcomeTransform(
                        variance_explained_threshold=PCA_VAR_THRESHOLD
                        # num_axes=config["latent_dim"]
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
        # "mtgp",  # TODO: revisit
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
                rank=config["latent_dim"],
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
                    # "standardize": InputStandardize(
                    #     config["outcome_dim"]
                    # ), # TODO
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
            # TODO: we are not centering / standardizing when doing random linear proj
            # this probably shouldn't matter that much?
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
                # TODO: generate >1 candidates
                exp_cand_X_EI = gen_exp_cand(
                    outcome_model,
                    pref_obj,
                    problem=problem,
                    q=N_FINAL_CAND,
                    acqf_name="qNEI",
                    X=X,
                )  # noqa

                exp_cand_X_PM = gen_exp_cand(
                    outcome_model,
                    pref_obj,
                    problem=problem,
                    q=N_FINAL_CAND,
                    acqf_name="posterior_mean",
                    X=X,
                )  # noqa

                # TODO: also apply util model / acqf to the initial exp batch
                # from {generated candidates, initial designs}
                # choose >1 that maximizes qNEIUU / util posterior mean
                # shouldn't choose just one, otherwise it's not improving from
                # what we have right now

                # qNEI, qSimpleRegret

                # qneiuu_util_EI = util_func(problem.evaluate_true(exp_cand_X_EI)).item()
                # qneiuu_util_PM = util_func(problem.evaluate_true(exp_cand_X_PM)).item()

                qneiuu_util_EI = (
                    util_func(problem.evaluate_true(exp_cand_X_EI)).max().item()
                )
                qneiuu_util_PM = (
                    util_func(problem.evaluate_true(exp_cand_X_PM)).max().item()
                )

                print(
                    f"{method}-{pe_strategy} qNEIUU candidate utility: {qneiuu_util_EI:.5f}",
                    f"{method}-{pe_strategy} post mean candidate utility: {qneiuu_util_PM:.5f}",
                )
            else:
                print("final fitting of utility model failed 3 times")
                qneiuu_util_EI, qneiuu_util_PM = None, None
                pref_model_acc = None
                outcome_model_mse = None
                exp_cand_X_EI, exp_cand_X_PM = None, None

            time_consumed = time.time() - time_start

            # log the true utility of the selected candidate experimental design
            exp_result = {
                "candidate_EI": exp_cand_X_EI,
                "candidate_util_EI": qneiuu_util_EI,
                "candidate_PM": exp_cand_X_PM,
                "candidate_util_PM": qneiuu_util_PM,
                "method": method,
                "strategy": pe_strategy,
                "run_id": trial_idx,
                "time_consumed": time_consumed,
                "outcome_model_mse": outcome_model_mse,
                "pref_model_acc": pref_model_acc,
                "initial_batch_util": initial_batch_max_util,
            }
            if method == "pca":
                print("Updating extra recovery diagnostics for PCA")
                exp_result["subspace_recovery_error"] = subspace_recovery_error(
                    axes_learned, config["ground_truth_axes"]
                )
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
        true_obj_cand_X_EI = gen_exp_cand(
            outcome_model, true_obj, problem=problem, q=1, acqf_name="qNEI", X=X
        )
        true_obj_cand_X_PM = gen_exp_cand(
            outcome_model,
            true_obj,
            problem=problem,
            q=1,
            acqf_name="posterior_mean",
            X=X,
        )
        true_obj_util_EI = util_func(problem.evaluate_true(true_obj_cand_X_EI)).item()
        true_obj_util_PM = util_func(problem.evaluate_true(true_obj_cand_X_PM)).item()

        print(
            f"True objective utility using qNEIUU: {true_obj_util_EI:.5f}",
            f"True objective utility using posterior mean: {true_obj_util_PM:.5f}",
        )
        exp_result = {
            "candidate_EI": true_obj_cand_X_EI,
            "candidate_util_EI": true_obj_util_EI,
            "candidate_PM": true_obj_cand_X_PM,
            "candidate_util_PM": true_obj_util_PM,
            "method": method,
            "strategy": "True Utility",
            "run_id": trial_idx,
            "initial_batch_util": initial_batch_max_util,
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


def make_problem(config, np_seed, torch_seed, **tkwargs):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.autograd.set_detect_anomaly(True)

    # generate a full set of basis vectors in the `output_dim`-dim outcome space
    full_axes = generate_principal_axes(
        output_dim=config["outcome_dim"], num_axes=config["outcome_dim"], **tkwargs
    )
    # take the first `num_axes to be the ground truth principal axes, from which we simulate the metrics
    ground_truth_principal_axes = full_axes[: config["latent_dim"]]

    # sample inputs to initialize the synthetic GP for generating PCs
    initial_X = torch.randn(
        (config["num_initial_samples"], config["input_dim"]), **tkwargs
    )

    obj_indices = list(range(config["outcome_dim"]))
    cons_indices = []

    problem = PCATestProblem(
        opt_config=[obj_indices, cons_indices],
        initial_X=initial_X,
        bounds=torch.Tensor([[0, 1]] * config["input_dim"]),
        ground_truth_principal_axes=ground_truth_principal_axes,
        noise_std=config["noise_std"],
        PC_lengthscales=Tensor(config["PC_lengthscales"]),
        PC_scaling_factors=Tensor(config["PC_scaling_factors"]),
        dtype=torch.double,
    )

    return problem, full_axes, ground_truth_principal_axes


# TODO: revisit
# here use fit_gpytorch_mll()
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
    fit_gpytorch_mll(mll_util)

    return util_model
