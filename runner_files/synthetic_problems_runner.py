import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import argparse

import torch
from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, generate_principal_axes, make_controlled_coeffs, make_problem)

experiment_configs = {
    "rank_1_linear": [2],
    "rank_2_linear": [2,1],
    "rank_4_linear": [4,2,2,1],
    # "rank_6_linear": [8,4,4,2,2,1],
    "rank_8_linear": [8,8,4,4,4,2,2,1],
    # TODO later: add nonlinear utility functions
}



def run_pipeline(
    config_name, trial_idx, outcome_dim, input_dim, noise_std, 
    methods = ["st", "pca", "pcr", "true_proj"],
    pe_strategies = ["EUBO-zeta", "Random-f"],
    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    **kwargs):

    _, rank, util_type = config_name.split('_')
    rank = int(rank)
    print('rank', rank)

    torch.manual_seed(trial_idx)

    full_axes = generate_principal_axes(
        output_dim=outcome_dim,
        num_axes=outcome_dim,
        seed = trial_idx,
        dtype=torch.double
    )

    for alpha in alphas: 

        print(f"=============== Running alpha={alpha} ===============")

        beta = make_controlled_coeffs(
            full_axes=full_axes,
            latent_dim=rank,
            alpha=alpha,
            n_reps = 1,
            dtype=torch.double
        ).transpose(-2, -1)
        print('beta shape', beta.shape)

        util_func = LinearUtil(beta=beta)

        true_axes = full_axes[: rank]
        print('ground truth principal axes', true_axes)

        problem = make_problem(
            input_dim = input_dim, 
            outcome_dim = outcome_dim,
            noise_std = noise_std,
            num_initial_samples = input_dim*outcome_dim,
            true_axes = true_axes,
            PC_lengthscales = [0.5]*rank,
            PC_scaling_factors = experiment_configs[config_name]
        )

        output_path = "/home/yz685/low_rank_BOPE/experiments/synthetic/" + \
            f"{config_name}_{input_dim}_{outcome_dim}_{alpha}_{noise_std}/"

        print("methods to plug into BopeExperiment: ", methods)

        experiment = BopeExperiment(
            problem, 
            util_func, 
            methods = methods,
            pe_strategies = pe_strategies,
            trial_idx = trial_idx,
            output_path = output_path,
            **kwargs
        )
        experiment.run_BOPE_loop()


def parse():
    # experiment-running params -- read from command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_idx", type = int, default = 0)
    parser.add_argument("--input_dim", type = int, default = 1)
    parser.add_argument("--outcome_dim", type = int, default = 50)
    parser.add_argument("--noise_std", type = float, default = 0.1)
    parser.add_argument("--n_check_post_mean", type = int, default = 13)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    for config_name in experiment_configs:
        print(f"================ Running {config_name} ================")
        run_pipeline(
            config_name = config_name,
            trial_idx = args.trial_idx,
            outcome_dim = args.outcome_dim,
            input_dim = args.input_dim,
            noise_std = args.noise_std,
            n_check_post_mean = args.n_check_post_mean, 
            methods=["st", "pca", "true_proj", "pcr"], 
            pe_strategies=["EUBO-zeta"],
            alphas=[0, 0.5, 1]
        )

    # TODO: can I replace absolute path with script directory, like
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # what's the difference between this and 
    # file_dir = os.path.dirname(__file__) ??
    # if output_path is None:
        # output_path = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp_output"
        # )
