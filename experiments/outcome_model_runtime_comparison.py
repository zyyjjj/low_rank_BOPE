#!/usr/bin/env python3
import copy
from typing import Any, Dict, List, NamedTuple

import fblearner.flow.api as flow
import gpytorch
import numpy as np
import torch

from low-rank-BOPE.src.pref_learning_helpers import generate_random_inputs
from low-rank-BOPE.src.synthetic_problem import generate_principal_axes, PCATestProblem
from low-rank-BOPE.src.transforms import PCAOutcomeTransform
from low-rank-BOPE.src.models import MultitaskGPModel

from botorch.models import SingleTaskGP
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.utils.sampling import draw_sobol_samples

from low-rank-BOPE.experiments.synthetic_test_problem_configs import (
    test_configs_outcome_model_fit,
)
from gpytorch.kernels import LCMKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

N_REPS = 30
PCA_VAR_THRESHOLD = 0.9
N_TEST_FIT = 800

tkwargs = {"dtype": torch.double}

# specify test simulation config
# test_configs = test_configs_low_latent_dim
# test_configs = {"config_3": test_configs_low_latent_dim["config_3"]}
# test_configs = test_configs_outcome_model_fit
test_configs = {
    "config_1": test_configs_outcome_model_fit["config_1"],
    "config_2": test_configs_outcome_model_fit["config_2"],
    "config_3": test_configs_outcome_model_fit["config_3"],
}


# class for saving experiment data
class OneRun(NamedTuple):
    # outcome_model_fit_results: List[Dict[str, Any]]
    outcome_model_fit_results: Dict[str, Any]


options = {"maxiter": 1000}

# workflow
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    configs: Dict[str, Dict[str, Any]] = test_configs,
    n_trials: int = N_REPS,
) -> Dict[str, List[OneRun]]:

    all_results = {}

    for config_name, config in configs.items():
        problem, full_axes = make_problem(
            config=config, np_seed=1234, torch_seed=1234, **tkwargs
        )

        all_results[config_name] = [
            run_one_trial(
                problem=problem,
                trial_idx=i,
                config=config,
            )
            for i in range(n_trials)
        ]

    return all_results


@flow.flow_async()
@flow.typed()
def run_one_trial(
    problem: torch.nn.Module,
    trial_idx: int,
    config: Dict[str, Any],
    verbose=True,
) -> OneRun:

    results_dict = {}
    print("running on config")
    print(config)

    # generate training data
    # problem, _, _ = make_problem(config, np_seed = trial_idx, torch_seed = trial_idx, **tkwargs)
    train_X = (
        draw_sobol_samples(bounds=problem.bounds, n=1, q=2 * config["outcome_dim"])
        .squeeze(0)
        .to(torch.double)
    )
    train_Y = problem.eval_metrics_noisy(train_X).detach()
    print(f"generated train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")

    # ======= initialize pca model =======
    print("initialize and train pca model")
    pca_model = SingleTaskGP(
        train_X,
        train_Y,
        outcome_transform=ChainedOutcomeTransform(
            **{
                "standardize": Standardize(config["outcome_dim"], min_stdv=100000),
                "pca": PCAOutcomeTransform(num_axes=config["latent_dim"]),
            }
        ),
        likelihood=GaussianLikelihood(noise_prior=GammaPrior(0.9, 10)),
    )
    pca_mll = ExactMarginalLogLikelihood(pca_model.likelihood, pca_model)

    # train PCA model and log training stats
    pca_fit_result = fit_gpytorch_scipy(pca_mll, options=options)
    results_dict["pca"] = {
        "wall_time": pca_fit_result[1]["wall_time"],
        "n_iterations": pca_fit_result[1]["OptimizeResult"].nit,
        "n_params": pca_fit_result[1]["OptimizeResult"].x.shape[0],
    }

    # ======= initialize lcm model =======
    print("intialize and train LCM model")

    lcm_kernel = LCMKernel(
        base_kernels=[MaternKernel()] * config["latent_dim"],
        num_tasks=config["outcome_dim"],
        rank=1,
    )
    lcm_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=config["outcome_dim"]
    )
    lcm_model = MultitaskGPModel(
        train_X,
        train_Y,
        lcm_likelihood,
        num_tasks=config["outcome_dim"],
        multitask_kernel=lcm_kernel,
        outcome_transform=Standardize(config["outcome_dim"], min_stdv=100000),
    )
    lcm_model.to(**tkwargs)
    # make a copy for testing warm-starting later
    lcm_model_warm_start = copy.deepcopy(lcm_model)
    lcm_mll = ExactMarginalLogLikelihood(lcm_model.likelihood, lcm_model)

    # train LCM model and log training stats
    lcm_fit_result = fit_gpytorch_scipy(lcm_mll, options=options)
    results_dict["lcm"] = {
        "wall_time": lcm_fit_result[1]["wall_time"],
        "n_iterations": lcm_fit_result[1]["OptimizeResult"].nit,
        "n_params": lcm_fit_result[1]["OptimizeResult"].x.shape[0],
    }

    # also try warm starting LCM with PCA estimates
    print("warm-start LCM model with PCA estimates")

    new_state_dict = {}
    for i in range(len(pca_model.outcome_transform["pca"].axes_learned)):
        key = (
            "covar_module.covar_module_list."
            + str(i)
            + ".task_covar_module.covar_factor"
        )
        new_state_dict[key] = (
            pca_model.outcome_transform["pca"].axes_learned[i].unsqueeze(1)
        )
    lcm_model_warm_start.load_state_dict(new_state_dict, strict=False)
    lcm_mll_warm_start = ExactMarginalLogLikelihood(
        lcm_model_warm_start.likelihood, lcm_model_warm_start
    )

    lcm_fit_result_warm_start = fit_gpytorch_scipy(lcm_mll_warm_start, options=options)

    results_dict["lcm_ws"] = {
        "wall_time": lcm_fit_result_warm_start[1]["wall_time"],
        "n_iterations": lcm_fit_result_warm_start[1]["OptimizeResult"].nit,
        "n_params": lcm_fit_result_warm_start[1]["OptimizeResult"].x.shape[0],
    }

    # ======= initialize ST model =======
    print("initialize and train single-task model")
    st_model = SingleTaskGP(
        train_X,
        train_Y,
        outcome_transform=Standardize(config["outcome_dim"], min_stdv=100000),
        likelihood=GaussianLikelihood(noise_prior=GammaPrior(0.9, 10)),
    )
    st_mll = ExactMarginalLogLikelihood(st_model.likelihood, st_model)
    st_model.train()

    # train ST model and log training stats
    st_fit_result = fit_gpytorch_scipy(st_mll, options=options)
    results_dict["st"] = {
        "wall_time": st_fit_result[1]["wall_time"],
        "n_iterations": st_fit_result[1]["OptimizeResult"].nit,
        "n_params": st_fit_result[1]["OptimizeResult"].x.shape[0],
    }

    # ======= initialize mtgp model =======
    print("initialize and train mtgp model")
    mtgp_model = KroneckerMultiTaskGP(
        train_X,
        train_Y,
        outcome_transform=Standardize(config["outcome_dim"], min_stdv=100000),
        rank=config["latent_dim"],
    )
    mtgp_mll = ExactMarginalLogLikelihood(mtgp_model.likelihood, mtgp_model)

    # train MTGP model and log training stats
    mtgp_fit_result = fit_gpytorch_scipy(mtgp_mll, options=options)

    results_dict["mtgp"] = {
        "wall_time": mtgp_fit_result[1]["wall_time"],
        "n_iterations": mtgp_fit_result[1]["OptimizeResult"].nit,
        "n_params": mtgp_fit_result[1]["OptimizeResult"].x.shape[0],
    }

    return OneRun(outcome_model_fit_results=results_dict)


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
    initial_X = torch.randn((config["num_initial_samples"], config["input_dim"]))

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

    return problem, full_axes
