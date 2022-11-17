import time
from typing import Callable, Dict, List, Tuple, Type

import torch

from ae.pca.models import (
    get_posterior_predictions_PCA,
    get_posterior_predictions_ST,
    initialize_model_PCA,
    initialize_model_ST,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.exceptions.errors import BotorchError
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood


def make_objective_metric(
    obj_indices: List, cons_indices: List
) -> ConstrainedMCObjective:
    r"""Creates optimization objective in metric space according to the indices for objective and constraints
    among all the metrics modeled.

    Args:
        obj_indices: list of indices of metrics that are the optimization objective(s)
        cons_indices: list of indices of metrics that are the optimization constraints

    Returns:
        ConstrainedMCObjective() objective specified in terms of metrics
    """

    def obj_callable_metric(Z):
        # if single-objective
        # ref https://www.internalfb.com/code/fbsource/[2c0e0fa3a4570be8b9d46eb6bc0034f3b1dfae66]/fbcode/pytorch/botorch/botorch/acquisition/objective.py?lines=297
        if len(obj_indices) == 1:
            return Z[..., obj_indices[0]]
        else:
            return Z[..., obj_indices]

    cons_callables_list_metric = []
    for cons_idx in cons_indices:

        def cons_callable(Z):
            return Z[..., cons_idx]

        cons_callables_list_metric.append(cons_callable)

    constrained_objective_ST = ConstrainedMCObjective(
        obj_callable_metric, cons_callables_list_metric
    )

    return constrained_objective_ST


def make_objective_PC(
    obj_indices: List, cons_indices: List, axes_learned: torch.Tensor, **tkwargs
) -> ConstrainedMCObjective:
    r"""Creates optimization objective in terms of principal components

    Args:
        obj_indices: list of indices of metrics that are the optimization objective(s)
        cons_indices: list of indices of metrics that are the optimization constraints
        axes_learned: `num_axes x output_dim` tensor where rows are learned principal axes

    Returns:
        constrained_objective_PC: ConstrainedMCObjective() objective specified in terms of PCs (rather than metrics)
    """

    axes_learned.to(**tkwargs)

    def obj_callable_PC(P):
        return torch.matmul(P, axes_learned[:, obj_indices]).squeeze(-1)

    cons_callables_list_PC = []
    for cons_idx in cons_indices:

        def cons_callable_PC(P):
            return torch.matmul(P, axes_learned[:, cons_idx]).squeeze(-1)

        cons_callables_list_PC.append(cons_callable_PC)

    constrained_objective_PC = ConstrainedMCObjective(
        obj_callable_PC, cons_callables_list_PC
    )

    return constrained_objective_PC


def get_weighted_objective(problem, **tkwargs):
    r"""Create a mapping from input to the feasibility-weighed objective value of the given problem."""

    def weighted_objective(X, **tkwargs):

        # row vector specifying whether all constraints are satisfied for each data point
        feasibility = torch.all(problem.evaluate_slack(X) <= 0, dim=1)

        return problem.evaluate_true(X) * feasibility

    return weighted_objective


def generate_initial_data(
    num_sample_points: int,
    problem: Type[ConstrainedBaseTestProblem],
    weighted_objective: Callable,
    **tkwargs,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    r"""Generate initial data for Bayesian optimization;
    also compute the best feasibility-weighted objective values among the generated points

    Args:
        num_sample_points: number of points to generate
        problem: a test problem
        weighted_objective: mapping from input to the feasibility-weighed objective value of the given problem

    Returns:
        train_X: `num_sample_points x input_dim` tensor of simulated inputs
        train_Y: `num_sample_points x output_dim` tensor of simulated outputs (metric observations)
        best_weighted_obj_val: (scalar) the best feasibility-weighted objective value among the generated points
    """

    train_X = (
        draw_sobol_samples(
            bounds=problem.bounds,
            n=1,
            q=num_sample_points,
        )
        .squeeze(0)
        .to(**tkwargs)
    )

    train_Y = problem.eval_metrics_noisy(train_X).detach()

    weighted_obj_vals = weighted_objective(train_X, **tkwargs).detach()
    best_weighted_obj_val = weighted_obj_vals.max().item()

    print("weighted objective values in initial data: ", weighted_obj_vals)
    print("best weighted objective value", best_weighted_obj_val)

    return train_X, train_Y, best_weighted_obj_val


def optimize_acqf_and_get_observation(
    model: Type[GPyTorchModel],
    problem: Type[ConstrainedBaseTestProblem],
    acqf_cls: Type[AcquisitionFunction],
    train_X: torch.Tensor,
    sampler: Type[MCSampler],
    constrained_objective: Type[ConstrainedMCObjective],
    BO_PARAMS: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Set up an acquisition function, optimize it, then return the candidate(s), observation(s), and acquisition function value(s).

    Args:
        model: a GPyTorchModel
        problem: a test problem
        acqf_cls: an AcquisitionFunction class; currently only supports qNoisyExpectedImprovement
        train_X: inputs evaluated so far
        sampler: a MCSampler
        constrained_objective: a ConstrainedMCObjective
        BO_PARAMS: dictionary of configuration parameters for running the BO loop

    Returns:
        new_x: `num_candidates x input_dim` tensor of selected candidates
        new_observations: `num_candidates x output_dim` tensor of observations at the candidate inputs
        new_X_acqf_vals: `num_candidates x 1` tensor of acquisition function values at the candidate inputs #TODO: check shape is right
    """

    # TODO: later need to add support for acqf_cls = qNEHVI; what additional arguments are needed?

    acqf = acqf_cls(
        model=model,
        X_baseline=train_X,
        sampler=sampler,
        objective=constrained_objective,
        cache_root=False,  # setting this according to https://github.com/pytorch/botorch/issues/1030
    )

    candidates, new_X_acqf_vals = optimize_acqf(
        acq_function=acqf,
        bounds=problem.bounds,
        q=BO_PARAMS["batch_size"],
        num_restarts=BO_PARAMS["num_restarts"],
        raw_samples=BO_PARAMS["raw_samples"],  # used for initialization heuristic
        options={
            "batch_limit": BO_PARAMS["batch_limit"],
            "maxiter": BO_PARAMS["maxiter"],
        },
    )

    new_X = candidates.detach()

    new_observations = problem.eval_metrics_noisy(new_X).detach()

    return new_X, new_observations, new_X_acqf_vals


def one_BO_trial(
    problem: Type[ConstrainedBaseTestProblem],
    num_initial_samples: int,
    obj_indices: List,
    cons_indices: List,
    method: str,
    acqf_cls: Type[AcquisitionFunction],
    infer_noise: bool,
    seed: int,
    num_BO_iterations: int,
    device: torch.device,
    variance_explained_threshold: float = None,
    verbose: bool = True,
    acqf_batch_size: int = 1,
    acqf_num_restarts: int = 10,
    acqf_raw_samples: int = 512,
    acqf_batch_limit: int = 5,
    acqf_maxiter: int = 200,
    num_MC_samples: int = 256,
    dtype: torch.dtype = torch.double,
):
    r"""Run one BO trial

    Args:
        problem: a test problem
        num_initial_samples: number of initial samples to take before starting BO
        obj_indices: list of indices of metrics that are the optimization objective(s)
        cons_indices: list of indices of metrics that are the optimization constraints
        method: one of {"PCA", "ST", "RS"}, where "RS" stands for random sampling
        acqf_cls: an AcquisitionFunction class; for now should be one of {qNoisyExpectedImprovement, qNoisyExpectedHypervolumeImprovement}
        infer_noise: whether to infer noise in model fitting. If True, fit a SingleTaskGP(); if False, fit a FixedNoiseGP()
        seed: random seed
        num_BO_iterations: number of iterations of BO
        device: device of all experiment tensors
        variance_explained_threshold: the fraction of variance in the data that we want the learned principal axes to explain
        verbose: if True, print more information, e.g., details about PCA model fitting
        acqf_batch_size: the number of candidates to generate in each iteration
        acqf_num_restarts: the number of starting points for multi-start acqf optimization
        acqf_raw_samples: the number of samples for initializing acqf optimization
        acqf_batch_limit: the limit on batch size in candidate generation
        acqf_maxiter: maximum number of iterations in acqf optimization
        num_MC_samples: the number of Monte Carlo samples in SobolQMCNormalSampler() -- how does the sampler work in qNEI()?????
        dtype: dtype of all experiment tensors

    Returns:
        Dictionary of logged values:
            "candidates": candidates selected in each BO iteration
            "best_vals": best objective value achieved after each BO iteration
            "candidate_values": objective value of each candidate
            "candidate_posterior_means": metric posterior mean of each candidate
            "candidate_posterior_vars": metric posterior variance of each candidate
            "candidate_acqf_vals": value of acquisition function for each candidate
            "model_MSEs": MSE of fitted GP model in predicting metric values
            "iteration_times": time consumed (seconds) of each iteration
    """

    tkwargs = {"dtype": dtype, "device": device}
    BO_kwargs = {
        "batch_size": acqf_batch_size,
        "num_restarts": acqf_num_restarts,
        "raw_samples": acqf_raw_samples,
        "batch_limit": acqf_batch_limit,
        "maxiter": acqf_maxiter,
    }

    weighted_objective = get_weighted_objective(problem, **tkwargs)

    torch.manual_seed(seed)

    # generate initial data
    train_X, train_Y, best_val = generate_initial_data(
        num_sample_points=num_initial_samples,
        problem=problem,
        weighted_objective=weighted_objective,
        **tkwargs,
    )

    print("train_X, train_Y shapes: ", train_X.shape, train_Y.shape)

    best_vals = []
    best_vals.append(best_val)

    candidates = []
    candidate_values = []
    candidate_posterior_means = []
    candidate_posterior_vars = []
    candidate_acqf_vals = []
    model_MSEs = []
    iteration_times = []

    for iteration in range(num_BO_iterations):
        print("BO iteration {}".format(iteration))
        iteration_start_time = time.time()

        # initialize GP model
        if method == "PCA":
            # manually center the Y data if using PCA,
            # since need to compute PCA decomposition before fitting GP

            # TODO: should also set the variance of train_Y to 1
            # is there a good way to handle this through some transform, rather than manually doing this before PCA? How does Wesley do it??

            train_Y_mean = torch.mean(train_Y, dim=0).detach()
            train_Y_centered = train_Y - train_Y_mean
            model, axes_learned = initialize_model_PCA(
                train_X=train_X,
                train_Y=train_Y_centered,
                variance_explained_threshold=variance_explained_threshold,
                verbose=verbose,
                **tkwargs,
            )
            print(
                f"number of outputs of PCA model at iteration {iteration}: {model.num_outputs}"
            )

        elif method in {"ST", "RS"}:
            model = initialize_model_ST(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=problem.noise_std**2,
                infer_noise=infer_noise,
                **tkwargs,
            )
        else:
            raise RuntimeError("Method not recognized")

        # fit GP model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # generate candidates
        if method == "RS":
            # sample new_X randomly, then get observations at new_X
            new_X = torch.rand((problem.input_dim, acqf_batch_size), **tkwargs)
            print("new_X shape: ", new_X.shape)

            new_Y = problem.eval_metrics_noisy(new_X).detach()

        else:
            qmc_sampler = SobolQMCNormalSampler(num_samples=num_MC_samples)

            if method == "PCA":
                # create a constrained objective in the PC space based on the latest axes_learned
                constrained_objective = make_objective_PC(
                    obj_indices=obj_indices,
                    cons_indices=cons_indices,
                    axes_learned=axes_learned,
                    **tkwargs,
                )
            elif method == "ST":
                # create a constrained objective in metric space
                constrained_objective = make_objective_metric(
                    obj_indices=obj_indices, cons_indices=cons_indices
                )

            # construct and optimize acqf and get new observations; new_Y here is not centered
            new_X, new_Y, new_X_acqf_vals = optimize_acqf_and_get_observation(
                model=model,
                problem=problem,
                acqf_cls=acqf_cls,
                train_X=train_X,
                sampler=qmc_sampler,
                constrained_objective=constrained_objective,
                BO_PARAMS=BO_kwargs,
            )

            candidate_acqf_vals.append(new_X_acqf_vals)

            print("new_X, new_Y shape: ", new_X.shape, new_Y.shape)

        # save candidate(s) and auxiliary information (acqf value, objective value, posterior mean and var, model fit MSE)
        candidates.append(new_X.item())
        new_X_value = weighted_objective(new_X, **tkwargs)
        candidate_values.append(new_X_value)
        if method == "PCA":
            (
                _,
                _,
                new_X_posterior_mean,
                new_X_posterior_var,
            ) = get_posterior_predictions_PCA(
                test_X=new_X, model_PC=model, axes_learned=axes_learned, **tkwargs
            )
            _, _, train_X_posterior_mean, _ = get_posterior_predictions_PCA(
                test_X=train_X, model_PC=model, axes_learned=axes_learned, **tkwargs
            )
            # these metric posterior means are centered; add back train_Y_mean
            new_X_posterior_mean = new_X_posterior_mean + train_Y_mean
            train_X_posterior_mean = train_X_posterior_mean + train_Y_mean

        elif method in {"ST", "RS"}:
            new_X_posterior_mean, new_X_posterior_var = get_posterior_predictions_ST(
                test_X=new_X, model=model, **tkwargs
            )
            train_X_posterior_mean, _ = get_posterior_predictions_ST(
                test_X=train_X, model=model, **tkwargs
            )
        candidate_posterior_means.append(new_X_posterior_mean)
        candidate_posterior_vars.append(new_X_posterior_var)
        model_MSEs.append(((train_Y - train_X_posterior_mean) ** 2).mean(axis=0))

        print(
            f"{method} sampled at candidate(s) {new_X.detach()} with objective value(s) {new_X_value.detach()}, \
            posterior mean(s) {new_X_posterior_mean.detach()}, posterior var(s) {new_X_posterior_var.detach()}"
        )

        # update training data
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y]).detach()

        # update best value so far
        best_val_tmp = weighted_objective(train_X, **tkwargs).max().item()
        best_vals.append(best_val_tmp)
        print(f"best value so far: {best_val_tmp}")

        iteration_time_spent = time.time() - iteration_start_time
        iteration_times.append(iteration_time_spent)
        print(
            f"Iteration {iteration} of {method} took {iteration_time_spent:.2f} seconds"
        )

        # re-initialize the models
        train_Y_mean = torch.mean(train_Y, dim=0).detach()
        train_Y_centered = train_Y - train_Y_mean

        if method == "PCA":
            model, axes_learned = initialize_model_PCA(
                train_X=train_X,
                train_Y=train_Y_centered,
                variance_explained_threshold=variance_explained_threshold,
                verbose=verbose,
                **tkwargs,
            )
            print(
                f"number of outputs of fitted PCA model after iteration {iteration}: {model.num_outputs}"
            )

        elif method in {"ST", "RS"}:
            model = initialize_model_ST(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=problem.noise_std**2,
                infer_noise=infer_noise,
                **tkwargs,
            )

    return {
        "candidates": candidates,
        "best_vals": best_vals,
        "candidate_values": candidate_values,
        "candidate_posterior_means": candidate_posterior_means,
        "candidate_posterior_vars": candidate_posterior_vars,
        "candidate_acqf_vals": candidate_acqf_vals,
        "model_MSEs": model_MSEs,
        "iteration_times": iteration_times,
    }
