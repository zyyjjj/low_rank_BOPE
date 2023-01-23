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
from low_rank_BOPE.src.diagnostics import check_util_model_fit_wrapper
from low_rank_BOPE.src.pref_learning_helpers import (fit_util_models_wrapper,
                                                     gen_initial_real_data)
from low_rank_BOPE.src.transforms import PCAOutcomeTransform
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, generate_principal_axes, make_controlled_coeffs, make_problem)

experiment_configs = {
    "rank_1_linear": [1],
    "rank_2_linear": [2,1],
    "rank_4_linear": [4,2,2,1],
    "rank_6_linear": [8,4,4,2,2,1],
    "rank_8_linear": [16,8,8,4,4,2,2,1],
    # TODO later: add nonlinear utility functions
}


def run_pipeline(config_name, seed, outcome_dim = 20, input_dim = 5):

    _, rank, util_type = config_name.split('_')
    rank = int(rank)
    print('rank', rank)

    torch.manual_seed(seed)

    full_axes = generate_principal_axes(
        output_dim=outcome_dim,
        num_axes=outcome_dim,
        dtype=torch.double
    )

    pca_acc_dict_alphas, st_acc_dict_alphas = [], []

    for alpha in [0, 0.4, 0.6, 1.0]:

        beta = make_controlled_coeffs(
            full_axes=full_axes,
            latent_dim=rank,
            alpha=alpha,
            n_reps = 1,
            dtype=torch.double
        ).transpose(-2, -1)
        print('beta shape', beta.shape)

        util_func = LinearUtil(beta=beta)

        ground_truth_principal_axes = full_axes[: rank]

        problem = make_problem(
            input_dim = input_dim,
            outcome_dim = outcome_dim,
            ground_truth_principal_axes = ground_truth_principal_axes,
            PC_lengthscales = [0.5]*rank,
            PC_scaling_factors = experiment_configs[config_name]
        )

        train_X, train_Y, util_vals, comps = gen_initial_real_data(n=100, problem=problem, util_func=util_func)
        print(
            'train X, train Y, util vals, comps shape', 
            train_X.shape, train_Y.shape, util_vals.shape, comps.shape
        )

        pca_model = SingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=ChainedOutcomeTransform(
                **{
                    "standardize": Standardize(outcome_dim, min_stdv=100),
                    "pca": PCAOutcomeTransform(
                        variance_explained_threshold=0.9
                    ), 
                }
            ),
            likelihood=GaussianLikelihood(noise_prior=GammaPrior(0.9, 10)),
        )
        pca_mll = ExactMarginalLogLikelihood(pca_model.likelihood, pca_model)

        fit_gpytorch_mll(pca_mll)

        pca_axes_dict = {
            "learned": pca_model.outcome_transform['pca'].axes_learned,
            "true": ground_truth_principal_axes,
            "oracle": beta.transpose(-2, -1)
        }
 
        pca_models_dict = fit_util_models_wrapper(
            train_Y, comps, util_vals, 
            method="pca", 
            axes_dict = pca_axes_dict, 
            modify_kernel = True
        )

        n_test = 400
        while n_test >= 50:
            try:
                pca_acc_dict = check_util_model_fit_wrapper(
                    problem, util_func, pca_models_dict, n_test=n_test
                )
                break
            except RuntimeError as e:
                print(str(e))
                n_test /= 2
                pca_acc_dict = check_util_model_fit_wrapper(
                    problem, util_func, pca_models_dict, n_test=n_test
                )

        st_models_dict = fit_util_models_wrapper(
            train_Y, comps, util_vals, 
            method="st"
        )

        n_test = 400
        while n_test >= 50:
            try:
                st_acc_dict = check_util_model_fit_wrapper(
                    problem, util_func, st_models_dict, n_test=n_test
                )
                break
            except RuntimeError as e:
                print(str(e))
                n_test /= 2
                st_acc_dict = check_util_model_fit_wrapper(
                    problem, util_func, st_models_dict, n_test=n_test
                )
        
        pca_acc_dict['learned_latent_dim'] = pca_model.outcome_transform['pca'].axes_learned.shape[0]

        print("pca_acc_dict", pca_acc_dict)
        print("st_acc_dict", st_acc_dict)
        pca_acc_dict_alphas.append(pca_acc_dict)
        st_acc_dict_alphas.append(st_acc_dict)

    return pca_acc_dict_alphas, st_acc_dict_alphas


if __name__ == "__main__":

    output_path = "../experiments/util_fit/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for config_name in experiment_configs:
        pca_acc_dict_l, st_acc_dict_l = [], []
        for seed in range(1):
            pca_acc_dicts, st_acc_dicts = run_pipeline(
                config_name=config_name, seed=seed
            )
            pca_acc_dict_l.append(pca_acc_dicts)
            st_acc_dict_l.append(st_acc_dicts)

        torch.save(pca_acc_dict_l, output_path + config_name + "_pca_acc.pt")
        torch.save(st_acc_dict_l, output_path + config_name + "_st_acc.pt")
