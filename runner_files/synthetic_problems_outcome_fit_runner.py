import os
import sys

# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append('/home/yz685/low_rank_BOPE/low_rank_BOPE')
import argparse
import math
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch
from botorch.models.model import Model

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.src.pref_learning_helpers import generate_random_inputs
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, generate_principal_axes, make_controlled_coeffs, make_problem)


def check_outcome_model_fit(
    outcome_model: Model, 
    problem: torch.nn.Module, 
    n_test: int, 
    batch_eval: bool = True
) -> float:
    r"""
    Evaluate the goodness of fit of the outcome model.
    Args:
        outcome_model: GP model mapping input to outcome
        problem: TestProblem
        n_test: size of test set
    Returns:
        mse: mean squared error between posterior mean and true value
            of the test set observations
    """

    torch.manual_seed(n_test)

    # generate test set
    test_X = generate_random_inputs(problem, n_test).detach()
    if not batch_eval:
        Y_list = []
        for idx in range(len(test_X)):
            y = problem(test_X[idx]).detach()
            Y_list.append(y)
        test_Y = torch.stack(Y_list).squeeze(1)
    else:
        test_Y = problem.evaluate_true(test_X).detach()

    test_Y_mean = test_Y.mean(axis=0)

    # run outcome model posterior prediction on test data
    test_posterior_mean = outcome_model.posterior(test_X).mean

    # compute relative mean squared error
    # mse = ((test_posterior_mean - test_Y)**2 / test_Y**2).mean(axis=0).detach().sum().item()
    relative_mse = ((test_posterior_mean - test_Y)**2 / (test_Y-test_Y_mean)**2).mean(axis=0).detach().sum().item()

    # se_rel = torch.sum((test_posterior_mean - test_Y) ** 2, dim=1) / torch.sum(test_Y**2, dim=1)
    # mse_rel = torch.sqrt(se_rel).mean(axis=0).item()

    return relative_mse


def run_fitting_helper(
    input_dim,
    outcome_dim,
    rank,
    noise_std,
    methods=["st", "pca", "lmc", "mtgp"], 
    datasizes=[20,50,100], 
    output_path="/home/yz685/low_rank_BOPE/experiments/synthetic/test_model_fit_",
    trial_idx = 101
):

    print(f"==========Running outcome dim {outcome_dim} trial {trial_idx}==========")

    full_axes = generate_principal_axes(
        output_dim=outcome_dim,
        num_axes=outcome_dim,
        seed = trial_idx,
        dtype=torch.double
    )

    beta = make_controlled_coeffs(
        full_axes=full_axes,
        latent_dim=rank,
        alpha=1,
        n_reps = 1,
        dtype=torch.double
    ).transpose(-2, -1)

    util_func = LinearUtil(beta=beta)

    true_axes = full_axes[: rank]

    problem = make_problem(
        input_dim = input_dim, 
        outcome_dim = outcome_dim,
        noise_std = noise_std,
        num_initial_samples = input_dim*outcome_dim,
        true_axes = true_axes,
        PC_lengthscales = [0.5]*rank,
        PC_scaling_factors = [2] # TODO: do not hardcode
    )


    fitting_time_dict = defaultdict(dict)
    mse_dict = defaultdict(dict)

    exp = BopeExperiment(
        problem, 
        util_func, 
        methods = methods,
        pe_strategies = ["EUBO-zeta"],
        trial_idx = trial_idx,
        output_path = output_path,
    )

    for datasize in datasizes:
        print(f"========Running datasize = {datasize}========")
        exp.generate_random_experiment_data(n=datasize, compute_util = False)
        for method in methods:
            print(f"======Running method {method}=======")
            start_time = time.time()
            exp.fit_outcome_model(method)
            model_fitting_time = time.time() - start_time
            mse = check_outcome_model_fit(exp.outcome_models_dict[method], exp.problem, n_test=1000)

            fitting_time_dict[datasize][method] = model_fitting_time
            mse_dict[datasize][method] = mse

            print(f"Fitting time {model_fitting_time} sec; mse {mse}")

            save_path = output_path + f'outcome_dim={outcome_dim}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(fitting_time_dict, save_path + f'fitting_time_dict_trial={trial_idx}.th')
            torch.save(mse_dict, save_path + f'mse_dict_trial={trial_idx}.th')
    
    # return fitting_time_dict, mse_dict


def parse():
    # experiment-running params -- read from command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_idx", type = int, default = 0)
    parser.add_argument("--outcome_dims", type = int, nargs = "+", default = [10,20,30,40])
    parser.add_argument("--methods", type = str, nargs = "+", 
        default = ["st", "pca", "lmc", "mtgp"])
    parser.add_argument("--pca_var_threshold", type = float, default = 0.9)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    for outcome_dim in args.outcome_dims:

        run_fitting_helper(
            input_dim=1,
            outcome_dim=outcome_dim,
            rank=1,
            noise_std=0.05,
            methods=args.methods,
            trial_idx=args.trial_idx
        )