import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import torch
import yaml
import numpy as np

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.bope_class_retraining import RetrainingBopeExperiment
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, PiecewiseLinear,\
    generate_principal_axes, make_controlled_coeffs, make_problem)

METHODS = {
    0: ["st"],
    1: ["pca", "random_linear_proj"], # since latent dim of rand_lin_proj depends on pca
    2: ["pca_rt"],
    3: ["random_search"]
}

"""
Steps:
- load csv of metric correlation matrix
- do SVD on the correlation matrix
- take the first p columns of V, set as true axes 
- take the first p entries of S, set their sqrt as scaling factor
- ? design utility function. Can try linear / piecewise linear. Would need to know which ones are important? 
"""

if __name__ == "__main__":

    # read trial_idx from command line input
    trial_idx = int(sys.argv[1])
    # read experiment config from yaml file
    args = yaml.load(open(sys.argv[2]), Loader = yaml.FullLoader)
    # read method_id from command line input if specified 
    # (run different methods in different slurm jobs)
    if len(sys.argv) > 2: # this means I specified method_id in command line
        methods = METHODS[int(sys.argv[3])]
    else:
        methods = args["methods"]

    print("Trial idx: ", trial_idx)
    print("Experiment args: ", args)


    for config_name in args["experiment_configs"]:
        print(f"================ Running {config_name} ================")

        matrix_id, input_dim, latent_dim, PCls, util_type, problem_seed = args["experiment_configs"][config_name]
        if isinstance(PCls, float):
            PCls = [PCls]*latent_dim
        else:
            assert len(PCls) == latent_dim, \
                "PCls must be a float or a list of floats of length latent_dim"

        CSVData = open(
            f"/home/yz685/low_rank_BOPE/low_rank_BOPE/test_problems/real_metric_corr/metric_corr_exp_{matrix_id}.csv")
        metric_corr = torch.tensor(
            np.loadtxt(CSVData, delimiter=","), dtype=torch.double)
        outcome_dim = len(metric_corr)
        U, S, V = torch.linalg.svd(
            metric_corr + torch.diag(torch.ones(len(metric_corr))) * 1e-10)
        true_axes = V[:latent_dim]
        scaling = torch.sqrt(S[:latent_dim])

        state_dict_str = f"PTS={matrix_id}_input={input_dim}_outcome={outcome_dim}_latent={latent_dim}_PCls={PCls}_seed={problem_seed}"

        problem = make_problem(
            input_dim = input_dim,
            outcome_dim = outcome_dim,
            noise_std = args["noise_std"],
            true_axes = true_axes,
            PC_lengthscales = PCls,
            PC_scaling_factors = scaling,
            problem_seed = problem_seed,
            state_dict_str = state_dict_str
        )

        for alpha in args["alphas"]:

            # alpha = component of coefficient in true outcome subspace
            # we vary the alphas here among [0, 0.5, 1]
            # a smaller alpha means the high-varying outcome directions matter less for utility

            beta = make_controlled_coeffs(
                full_axes=V,
                latent_dim=latent_dim,
                alpha=alpha,
                n_reps = 1,
                dtype=torch.double,
                seed = problem_seed
            ) # shape is 1 x outcome_dim

            if util_type == "linear":
                util_func = LinearUtil(beta=beta.transpose(-2, -1))
            elif util_type == "pwlinear":
                util_func = PiecewiseLinear(
                    beta1=5*beta,
                    beta2=beta,
                    thresholds=torch.tensor([0.]*outcome_dim, dtype=torch.double)
                )
            
            output_path = "/home/yz685/low_rank_BOPE/experiments/synthetic_pts/" + \
                f"{config_name}_alpha={alpha}/"
            
            if all(os.path.exists(output_path + f"BO_data_trial={trial_idx}_{method}.th") \
                for method in methods):
                
                print(f"Already ran all methods in {methods} for trial {trial_idx}, skipping...")
            
            else:

                experiment = RetrainingBopeExperiment(
                    problem, 
                    util_func, 
                    methods = args["methods"],
                    pe_strategies = args["pe_strategies"],
                    trial_idx = trial_idx,
                    output_path = output_path,
                    initial_experimentation_batch = args["init_exp_batch"],
                    n_check_post_mean = args["n_check_post_mean"], 
                    pca_var_threshold = args["pca_var_threshold"],
                    BO_batch_size = args.get("BO_batch_size", 4),
                    n_BO_iters = args.get("n_BO_iters", 20),
                )
                experiment.run_BOPE_loop()
