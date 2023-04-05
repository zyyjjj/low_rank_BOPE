import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import torch
import yaml

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.bope_class_retraining import RetrainingBopeExperiment
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, generate_principal_axes, make_controlled_coeffs, make_problem)


def run_pipeline(
    experiment_configs, config_name, trial_idx, outcome_dim, input_dim, noise_std, 
    retrain,
    methods = ["st", "pca", "pcr", "true_proj"],
    pe_strategies = ["EUBO-zeta", "Random-f"],
    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    problem_seed = None,
    **kwargs):

    _, rank, util_type = config_name.split('_')
    rank = int(rank)
    print('rank', rank)

    # torch.manual_seed(trial_idx)

    full_axes = generate_principal_axes(
        output_dim=outcome_dim,
        num_axes=outcome_dim,
        dtype=torch.double,
        seed=problem_seed
    )

    if retrain: 
        experiment_class = RetrainingBopeExperiment
        suffix = "_rt"
    else:
        experiment_class = BopeExperiment
        suffix = ""

    for alpha in alphas: 

        print(f"=============== Running alpha={alpha} ===============")

        beta = make_controlled_coeffs(
            full_axes=full_axes,
            latent_dim=rank,
            alpha=alpha,
            n_reps = 1,
            dtype=torch.double,
            seed = problem_seed
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
            PC_scaling_factors = experiment_configs[config_name],
            problem_seed = problem_seed
        )

        output_path = f"/home/yz685/low_rank_BOPE/experiments/synthetic{suffix}/" + \
            f"{config_name}_{input_dim}_{outcome_dim}_{alpha}_{noise_std}/"

        print("methods to plug into BopeExperiment: ", methods)

        experiment = experiment_class(
            problem, 
            util_func, 
            methods = methods,
            pe_strategies = pe_strategies,
            trial_idx = trial_idx,
            output_path = output_path,
            **kwargs
        )
        experiment.run_BOPE_loop()


if __name__ == "__main__":

    # read trial_idx from command line input
    trial_idx = int(sys.argv[1])
    # read experiment config from yaml file
    args = yaml.load(open(sys.argv[2]), Loader = yaml.FullLoader)

    print("Experiment args: ", args)

    for config_name in args["experiment_configs"]:
        print(f"================ Running {config_name} ================")
        run_pipeline(
            experiment_configs = args["experiment_configs"],
            config_name = config_name,
            trial_idx = trial_idx,
            outcome_dim = args["outcome_dim"],
            input_dim = args["input_dim"],
            noise_std = args["noise_std"],
            retrain = args["retrain"],
            n_check_post_mean = args["n_check_post_mean"], 
            methods = args["methods"],
            pe_strategies = args["pe_strategies"],
            alphas=args["alphas"],
            pca_var_threshold = args["pca_var_threshold"],
            initial_experimentation_batch = args["init_exp_batch"],
            problem_seed = args["problem_seed"] 
            # prblem_seed should be set the same for all trials in one problem instance
        )
