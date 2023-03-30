import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import multiprocessing
from collections import defaultdict

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
    "rank_1_linear": [2],
    "rank_2_linear": [4,2],
    "rank_4_linear": [8,4,4,2],
    # "rank_6_linear": [8,4,4,2,2,1],
    # "rank_8_linear": [16,8,8,4,4,2,2,1],
    # TODO later: add nonlinear utility functions
}


def run_pipeline(config_name, seed, noise_std, outcome_dim = 20, input_dim = 5):
    """
    Args:
        config_name: string specifying the outcome rank and utility type
        seed: integer for the random seed
        noise_std: magnitude of independent noise to add to outcome dimensions
        outcome_dim: dimensionality of the outcome
        input_dim: dimensionality of the input
    Returns:
        pca_acc_dict_alphas: dictionary with alpha values as keys and each value
            being a dictionary of the pca-fitted utility model's test accuracy
            for the simulated utility with the particular alpha value
        st_acc_dict_alphas: same but for utility model without dim reduction
    """


    _, rank, util_type = config_name.split('_')
    rank = int(rank)
    print('rank', rank)

    torch.manual_seed(seed)

    full_axes = generate_principal_axes(
        output_dim=outcome_dim,
        num_axes=outcome_dim,
        seed = seed,
        dtype=torch.double
    )

    pca_acc_dict_alphas, st_acc_dict_alphas = defaultdict(dict), defaultdict(dict)

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
            noise_std = noise_std, 
            num_initial_samples = input_dim*outcome_dim,
            true_axes = true_axes,
            PC_lengthscales = [0.5]*rank,
            PC_scaling_factors = experiment_configs[config_name]
        )

        train_X, train_Y, util_vals, comps = gen_initial_real_data(
            n=100, problem=problem, util_func=util_func
        )
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
        print('pca transform properties', pca_model.outcome_transform['pca'].__dict__)

        # check pca model fit (with different linear transformations)
        pca_axes_dict = {
            "learned": pca_model.outcome_transform['pca'].axes_learned,
            "true": true_axes,
            "oracle": beta.transpose(-2, -1)
        }
        pca_models_dict = fit_util_models_wrapper(
            train_Y, comps, util_vals, 
            method="pca", 
            axes_dict = pca_axes_dict, 
            modify_kernel = True
        )
        check_pca_model_fit_succeed = False
        n_test = 400
        while n_test >= 50:
            try:
                pca_acc_dict = check_util_model_fit_wrapper(
                    problem, util_func, pca_models_dict, n_test=n_test
                )
                check_pca_model_fit_succeed = True
                break
            except RuntimeError as e:
                n_test = int(n_test / 2)
                print(str(e), f'updated n_test = {n_test}')
                continue
        if not check_pca_model_fit_succeed:
            pca_acc_dict = {}

        # check independent GP model fit
        st_models_dict = fit_util_models_wrapper(
            train_Y, comps, util_vals, 
            method="st"
        )
        check_st_model_fit_succeed = False
        n_test = 400
        while n_test >= 50:
            try:
                st_acc_dict = check_util_model_fit_wrapper(
                    problem, util_func, st_models_dict, n_test=n_test
                )
                check_st_model_fit_succeed = True
                break
            except RuntimeError as e:
                print(str(e))
                n_test /= 2
        if not check_st_model_fit_succeed:
            st_acc_dict = {}
        
        print('PCA learned axes shape', pca_model.outcome_transform['pca'].axes_learned.shape)
        pca_acc_dict['learned_latent_dim'] = pca_model.outcome_transform['pca'].axes_learned.shape[0]
        pca_acc_dict["alpha"] = alpha
        st_acc_dict["alpha"] = alpha

        print("pca_acc_dict", pca_acc_dict)
        print("st_acc_dict", st_acc_dict)
        pca_acc_dict_alphas[alpha] = pca_acc_dict
        st_acc_dict_alphas[alpha] = st_acc_dict

    return pca_acc_dict_alphas, st_acc_dict_alphas


if __name__ == "__main__":

    trial_idx = int(sys.argv[1])
    input_dim = int(sys.argv[2])
    outcome_dim = int(sys.argv[3])
    noise_std = float(sys.argv[4])

    output_path = f"/home/yz685/low_rank_BOPE/experiments/util_fit_{input_dim}_{outcome_dim}_{noise_std}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for config_name in experiment_configs:

        pca_acc_dict_alphas, st_acc_dict_alphas = run_pipeline(
            config_name=config_name, seed=trial_idx, 
            noise_std = noise_std,
            outcome_dim = outcome_dim, input_dim = input_dim
        ) 

        # pool = multiprocessing.Pool(n_replications)
        # params = [(config_name, seed, outcome_dim, input_dim) for seed in range(n_replications)]
        # print(params)
        # pca_acc_dict_l, st_acc_dict_l = zip(*pool.map(run_pipeline, params))
        # pca_acc_dict_l, st_acc_dict_l = zip(*pool.map(partial(run_pipeline, config_name = config_name, outcome_dim = outcome_dim, input_dim = input_dim), list(range(5))))

        torch.save(
            pca_acc_dict_alphas, 
            output_path + config_name + f"_pca_acc_trial={trial_idx}.pt"
        )
        torch.save(
            st_acc_dict_alphas, 
            output_path + config_name + f"_st_acc_trial={trial_idx}.pt"
        )
