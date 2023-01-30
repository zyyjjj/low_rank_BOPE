import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import (ChainedOutcomeTransform,
                                               Standardize)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.src.pref_learning_helpers import gen_initial_real_data
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, generate_principal_axes, make_controlled_coeffs, make_problem)

experiment_configs = {
    "rank_1_linear": [2],
    "rank_2_linear": [2,1],
    "rank_4_linear": [4,2,2,1],
    # "rank_6_linear": [8,4,4,2,2,1],
    # "rank_8_linear": [16,8,8,4,4,2,2,1],
    # TODO later: add nonlinear utility functions
}



def run_pipeline(config_name, trial_idx, outcome_dim = 20, input_dim = 1):

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

    for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:

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
            num_initial_samples = input_dim*outcome_dim,
            true_axes = true_axes,
            PC_lengthscales = [0.5]*rank,
            PC_scaling_factors = experiment_configs[config_name]
        )

        experiment = BopeExperiment(
            problem, 
            util_func, 
            methods = ["st", "pca", "pcr", "true_proj"],
            pe_strategies = [
                "EUBO-zeta", 
                "Random-f"
            ],
            trial_idx = trial_idx,
            output_path = "/home/yz685/low_rank_BOPE/experiments/"+config_name+"/"
        )
        experiment.run_BOPE_loop()


if __name__ == "__main__":

    # experiment-running params -- read from command line input
    trial_idx = int(sys.argv[1])

    for config_name in experiment_configs:
        run_pipeline(
            config_name = config_name,
            trial_idx = trial_idx,
            outcome_dim = 20,
            input_dim = 1
        )

    # TODO: can I replace absolute path with script directory, like
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # what's the difference between this and 
    # file_dir = os.path.dirname(__file__) ??
    # if output_path is None:
        # output_path = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp_output"
        # )
