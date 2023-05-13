import copy
import random
import time
from collections import defaultdict
import multiprocessing
from typing import Any, Dict, List, NamedTuple
import gpytorch
import numpy as np
import torch
import sys
sys.path.append('..')


from low_rank_BOPE.src.lunar_lander import LunarLander
from low_rank_BOPE.src.pref_learning_helpers import (
    check_outcome_model_fit,
    check_pref_model_fit,
    find_max_posterior_mean,
    fit_outcome_model,
    fit_pref_model,
    gen_exp_cand,
    generate_random_exp_data,
    generate_random_pref_data,
    run_pref_learn
)
from low_rank_BOPE.src.transforms import (
    generate_random_projection,
    InputCenter,
    LinearProjectionInputTransform,
    LinearProjectionOutcomeTransform,
    PCAInputTransform,
    PCAOutcomeTransform,
    SubsetOutcomeTransform,
)
from botorch.acquisition.objective import GenericMCObjective, LearnedObjective
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    Normalize,
)

from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
# from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem

from low_rank_BOPE.src.models import make_modified_kernel, MultitaskGPModel
from low_rank_BOPE.src.diagnostics import (
    empirical_max_outcome_error,
    empirical_max_util_error,
    mc_max_outcome_error,
    mc_max_util_error,
)

from gpytorch.kernels import LCMKernel, MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior

N_BOPE_REPS = 30 # TODO: make arg
N_BOPE_REPS = sys.argv[1]
PCA_VAR_THRESHOLD = 0.95
AUGMENTED_DIMS_NOISE = 0.1
MIN_STD = 100000
N_TEST = 1000

tkwargs = {"dtype": torch.double}

BASE_CONFIG = {
    "initial_experimentation_batch": 16,
    "n_check_post_mean": 8, # TODO: make arg; use yaml to load
    # "n_check_post_mean": 1,
    "every_n_comps": 3,
}

INPUT_DIM = 12
NUM_ENVS = [20]  # this is the number of scenarios, so number of outcomes # TODO: make arg
BASELINE_REWARD = -200
MIN_REWARD_DIFF = 0
SIGMOID_COEFF = 0.01\

# problem-specific params -- put into a yaml file
# 
# 
# experiment-specific params -- read from command line input


METHODS = [
        "st",
        "pca", # TODO: add back
        # "random_linear_proj",
        # "random_subset",
        "lmc1",
        "lmc2"
    ]

# design sigmoid utility function reflecting distance from constraint
# torch.sigmoid( coeff * (outcome - MIN_REWARD) ),
# where coeff is a positive number; the larger coeff, the steeper the sigmoid

class Sigmoid(torch.nn.Module):
    def __init__(self, scale_coeff, threshold):
        super().__init__()
        self.scale_coeff = scale_coeff
        self.threshold = threshold

    def forward(self, Y, X=None)->torch.Tensor:
        if len(Y.shape) == 1:
            Y = Y.unsqueeze(0)

        return torch.sum(torch.sigmoid(self.scale_coeff * (Y - self.threshold)), dim = 1)

sigmoid_util_func = Sigmoid(SIGMOID_COEFF, MIN_REWARD_DIFF)

# workflow
def main(
    num_envs_list: List[int] = [50],
    n_trials: int = N_BOPE_REPS,
    save_file_name: str = 'lunar_lander_results_lkj'
) -> Dict[int, List]:
# TODO: main() should take in trial_idx as an argument

    all_results = defaultdict(list)

    for num_envs in num_envs_list:

        config = copy.deepcopy(BASE_CONFIG)
        config["input_dim"] = INPUT_DIM
        config["outcome_dim"] = num_envs
        problem = LunarLander(
            num_envs=num_envs,
            min_reward=BASELINE_REWARD
        )

        # make this part multiprocessing
        # TODO: if submit to cluster, just submit each one as a separate job
        # and then recover the multiprocessing part in computing one instance of lunar lander
        # also make sure to save files with names specific to each trial index
        # format for saving data: dict
        mpc_args = [(problem, sigmoid_util_func, int(i), config) for i in range(n_trials)]
        pool = multiprocessing.Pool()
        all_results[num_envs] = pool.starmap(run_one_trial, mpc_args)
        torch.save(all_results, save_file_name)

        # for i in range(N_BOPE_REPS):
        #     print(f'==========running BOPE rep {i}==========')
        #     all_results[num_envs].append(
        #         run_one_trial(
        #         problem=problem,
        #         util_func=sigmoid_util_func,
        #         trial_idx=i,
        #         config=config,
        #         # **tkwargs,
        #     ))

    return all_results


def run_one_trial(
    problem: torch.nn.Module,
    util_func: torch.nn.Module,
    trial_idx: int,
    config: Dict[str, Any],
    verbose=True,
    # **tkwargs,
) -> Dict:

    print(f"Running trial number {trial_idx} for the problem config:")
    print(problem, util_func, config)

    torch.manual_seed(trial_idx)
    np.random.seed(trial_idx)

    within_session_results = []
    exp_candidate_results = []

    # ======= Experimentation stage =======
    # initial exploration batch

    X, Y = generate_random_exp_data(problem, config["initial_experimentation_batch"], batch_eval = False)
    print("X,Y dtypes", X.dtype, Y.dtype)

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
            # TODO: if we want to do the transforms explicitly
            # use Qing's code for map-saas,
            # but also don't forget to center the outcomes
            "outcome_tf": ChainedOutcomeTransform(
                **{
                    "standardize": Standardize(
                        config["outcome_dim"],
                        min_stdv=MIN_STD,  # TODO: try standardize again
                    ),
                    # "pca": PCAOutcomeTransform(num_axes=config["lin_proj_latent_dim"]),
                    "pca": PCAOutcomeTransform(
                        variance_explained_threshold=PCA_VAR_THRESHOLD
                    ),
                }
            ),
        },
        # "mtgp": {
        #     "outcome_tf": Standardize(config["outcome_dim"]),
        #     "input_tf": Normalize(config["outcome_dim"]),
        #     "covar_module": make_modified_kernel(ard_num_dims=config["outcome_dim"]),
        # },
        "lmc1": {
            "outcome_tf": Standardize(config["outcome_dim"]),
            "input_tf": Normalize(config["outcome_dim"]),
            "covar_module": make_modified_kernel(ard_num_dims=config["outcome_dim"]),
        },
        "lmc2": {
            "outcome_tf": Standardize(config["outcome_dim"]),
            "input_tf": Normalize(config["outcome_dim"]),
            "covar_module": make_modified_kernel(ard_num_dims=config["outcome_dim"]),
        },
    }

    for method in METHODS:

        print(f"=====Running method {method}=====")

        lin_proj_latent_dim = 1  # define variable, placeholder

        time_start = time.time()  # log outcome model fitting time
        if method == "mtgp":
            outcome_model = KroneckerMultiTaskGP(
                X,
                Y,
                outcome_transform=copy.deepcopy(
                    transforms_covar_dict[method]["outcome_tf"]
                ),
                rank=3,
            )
            icm_mll = ExactMarginalLogLikelihood(
                outcome_model.likelihood, outcome_model
            )
            fit_gpytorch_model(icm_mll)

        elif method in ("lmc1", "lmc2"):

            # TODO: check covariance LKJ prior here
            sd_prior = GammaPrior(1.0, 0.15)
            eta = 0.5
            task_covar_prior = LKJCovariancePrior(config["outcome_dim"], eta, sd_prior)

            if method == "lmc1":
                lcm_kernel = LCMKernel(
                    base_kernels=[MaternKernel()] * lin_proj_latent_dim,
                    num_tasks=config["outcome_dim"],
                    rank=1,
                    task_covar_prior=task_covar_prior, # TODO: fix
                )
            else:
                lcm_kernel = LCMKernel(
                    base_kernels=[MaternKernel()] * lin_proj_latent_dim,
                    num_tasks=config["outcome_dim"],
                    rank=2,
                    task_covar_prior=task_covar_prior, # TODO: fix
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
            ).to(**tkwargs)
            fit_gpytorch_scipy(lcm_mll, options={"maxls": 30})

        else:
            outcome_model = fit_outcome_model(
                X,
                Y,
                outcome_transform=transforms_covar_dict[method]["outcome_tf"],
            )

        outcome_model_fitting_time = time.time() - time_start

        if method == "pca":

            axes_learned = outcome_model.outcome_transform["pca"].axes_learned

            transforms_covar_dict["pca"]["input_tf"] = ChainedInputTransform(
                **{
                    # "standardize": InputStandardize(config["outcome_dim"]),
                    # TODO: was trying standardize again
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
                    batch_eval = False
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
                    outcome_model=outcome_model, problem=problem, n_test=N_TEST,
                    batch_eval = False
                )
                print(f"outcome model mse: {outcome_model_mse}")
                pref_model_acc = check_pref_model_fit(
                    pref_model, problem=problem, util_func=util_func, n_test=N_TEST,
                    batch_eval = False
                )
                print(f"final pref model acc: {pref_model_acc}")
                sampler = SobolQMCNormalSampler(1)
                pref_obj = LearnedObjective(pref_model=pref_model, sampler=sampler)
                exp_cand_X = gen_exp_cand(
                    outcome_model, pref_obj, problem=problem, q=1, acqf_name="qNEI", X=X
                )  # noqa
                qneiuu_util = util_func(problem.evaluate_true(exp_cand_X)).item()
                print(
                    f"{method}-{pe_strategy} qNEIUU candidate utility: {qneiuu_util:.5f}"
                )
            else:
                qneiuu_util = None
                pref_model_acc = None

            time_consumed = time.time() - time_start

            # log the true utility of the selected candidate experimental design
            exp_result = {
                "candidate": exp_cand_X,
                "candidate_util": qneiuu_util,
                "method": method,
                "strategy": pe_strategy,
                "run_id": trial_idx,
                "time_consumed": time_consumed,
                "outcome_model_mse": outcome_model_mse,
                "pref_model_acc": pref_model_acc,
                "outcome_model_fit_time": outcome_model_fitting_time,
            }
            if method == "pca":
                print("Updating extra recovery diagnostics for PCA")
                # exp_result["empirical_max_outcome_error"] = empirical_max_outcome_error(
                #     Y, axes_learned  # Y here is the initial experiment data
                # )
                # exp_result["mc_max_outcome_error"] = mc_max_outcome_error(
                #     problem, axes_learned, N_TEST
                # )
                # exp_result["empirical_max_util_error"] = empirical_max_util_error(
                #     train_Y, axes_learned, util_func
                # )
                # exp_result["mc_max_util_error"] = mc_max_util_error(
                #     problem, axes_learned, util_func, N_TEST
                # )
                exp_result["num_axes"] = lin_proj_latent_dim
            exp_candidate_results.append(exp_result)

        # Baseline 3: Oracle / Assume knowing true utility
        # -- this depends on the outcome model, which in turn depends on the method
        # true_obj = GenericMCObjective(util_func)
        # true_obj_cand_X = gen_exp_cand(
        #     outcome_model, true_obj, problem=problem, q=1, acqf_name="qNEI", X=X
        # )
        # true_obj_util = util_func(problem.evaluate_true(true_obj_cand_X)).item()
        # print(f"True objective utility: {true_obj_util:.5f}")
        # exp_result = {
        #     "candidate": true_obj_cand_X,
        #     "candidate_util": true_obj_util,
        #     "method": method,
        #     "strategy": "True Utility",
        #     "run_id": trial_idx,
        # }
        # exp_candidate_results.append(exp_result)

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

    # return OneRun(
    #     exp_candidate_results=exp_candidate_results,
    #     within_session_results=within_session_results,
    # )

    return {
        "exp_candidate_results": exp_candidate_results,
        "within_session_results": within_session_results
    }


if __name__ == '__main__':
    main(num_envs_list = NUM_ENVS, save_file_name = '1203_lunar_lander_results_lkj_20dim')
