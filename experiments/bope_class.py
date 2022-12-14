#!/usr/bin/env python3

from collections import defaultdict
import gpytorch
import numpy as np
import copy
import random
import torch

from gpytorch.kernels import LCMKernel, MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior

from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    Normalize,
)
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.objective import GenericMCObjective, LearnedObjective
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.sampling import draw_sobol_samples

from sklearn.linear_model import LinearRegression

from low_rank_BOPE.src.pref_learning_helpers import (
    check_outcome_model_fit,
    check_pref_model_fit,
    # find_max_posterior_mean, # TODO: later see if we want the error-handled version
    fit_outcome_model,
    # fit_pref_model, # TODO: later see if we want the error-handled version
    gen_exp_cand,
    generate_random_exp_data,
    generate_random_pref_data,
    gen_comps,
    ModifiedFixedSingleSampleModel
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
from low_rank_BOPE.src.models import make_modified_kernel, MultitaskGPModel



class BopeExperiment:

    attr_list = {
        "pca_var_threshold": 0.95,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 20,
        "every_n_comps": 3,
        "verbose": True,
        "dtype": torch.double,
        "noise_std": 0.01,  # TODO: figure out how to set this for different probs
        "num_restarts": 20,
        "raw_samples": 128,
        "batch_limit": 4,
        "sampler_num_outcome_samples": 64,
        "maxiter": 1000,
        "use_PCR": False,
        # maybe num_axes, default to None?
        "min_stdv": 100000
    }


    def __init__(
        self,
        problem,
        util_func,
        methods,
        pe_strategies,
        trial_idx,
        save_dir,
        **kwargs
    ) -> None:
        """
        specify experiment settings
            problem:
            util_func:
            methods: list of statistical methods, each denoted using a str
            pe_strategies: PE strategies to use, could be {"EUBO-zeta", "Random-f"}
        one run should handle one problem and >=1 methods and >=1 pe_strategies
        """

        # pre-specified experiment metadata
        self.problem = problem
        self.util_func = util_func
        self.outcome_dim = problem.outcome_dim
        self.input_dim = problem.dim

        # self.attr_list stores default values, then overwrite with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.methods = methods
        self.pe_strategies = pe_strategies

        # logging models and results
        self.outcome_models_dict = {}
        self.pref_data_dict = defaultdict(dict)
        self.PE_session_results = defaultdict(dict)
        self.final_candidate_results = defaultdict(dict)

        # specify outcome and input transforms, covariance modules
        self.transforms_covar_dict = {
            "st": {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.outcome_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.outcome_dim),
            },
            "pca": {
                "outcome_tf": ChainedOutcomeTransform(
                    **{
                        "standardize": Standardize(
                            self.outcome_dim,
                            min_stdv=self.min_stdv,
                        ),
                        # "pca": PCAOutcomeTransform(num_axes=config["lin_proj_latent_dim"]),
                        "pca": PCAOutcomeTransform(
                            variance_explained_threshold=self.pca_var_threshold
                        ),
                    }
                ),
            },
            "mtgp": {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.outcome_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.outcome_dim),
            },
            "lmc": {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.outcome_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.outcome_dim),
            },
            "lmc2": {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.outcome_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.outcome_dim),
            },
        }

        # TODO: error handling -- maybe not needed
        self.passed = False
        self.fit_count = 0


    def generate_random_experiment_data(self, n, compute_util: False):
        r"""Generate n observations of experimental designs and outcomes.
        Args:
            problem: a TestProblem
            n: number of samples
        Computes:
            X: `n x problem input dim` tensor of sampled inputs
            Y: `n x problem outcome dim` tensor of noisy evaluated outcomes at X
            util_vals: `n x 1` tensor of utility value of Y
            comps: `n/`2` x `2`` tensor of pairwise comparisons # TODO: confirm
        """

        self.X = (
            draw_sobol_samples(bounds=self.problem.bounds, n=1, q=n)
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        self.Y = self.problem(self.X).detach()

        if compute_util:
            self.util_vals = self.util_func(self.Y).detach()
            self.comps = gen_comps(self.util_vals)


    def fit_outcome_model(self, method):

        if method == "mtgp":
            outcome_model = KroneckerMultiTaskGP(
                self.X,
                self.Y,
                outcome_transform=copy.deepcopy(
                    self.transforms_covar_dict[method]["outcome_tf"]
                ),
                rank=3, # TODO: update
            )
            icm_mll = ExactMarginalLogLikelihood(
                outcome_model.likelihood, outcome_model
            )
            fit_gpytorch_model(icm_mll)

        elif method == "lmc":

            # TODO: check covariance LKJ prior here
            sd_prior = GammaPrior(1.0, 0.15)
            eta = 0.5
            task_covar_prior = LKJCovariancePrior(self.outcome_dim, eta, sd_prior)

            lcm_kernel = LCMKernel(
                base_kernels=[MaternKernel()] * self.lin_proj_latent_dim,
                # TODO: Qing's comment: Here the base kernel is MaternKernel without setting ard_dim and prior. Is this intended?
                num_tasks=self.outcome_dim,
                rank=1, # rank is `2` if method is lmc2
                task_covar_prior=task_covar_prior,
            )
            lcm_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.outcome_dim
            )
            outcome_model = MultitaskGPModel(
                self.X,
                self.Y,
                lcm_likelihood,
                num_tasks=self.outcome_dim,
                multitask_kernel=lcm_kernel,
                outcome_transform=copy.deepcopy(
                    self.transforms_covar_dict[method]["outcome_tf"]
                ),
            ).to(self.dtype)
            lcm_mll = ExactMarginalLogLikelihood(
                outcome_model.likelihood, outcome_model
            )
            fit_gpytorch_scipy(lcm_mll, options={"maxls": 30})

        elif method == "pcr":
            P, S, V = torch.svd(self.Y)
            # then run regression from P (PCs) onto util_vals
            # retain the corresponding columns in P

            reg = LinearRegression().fit(np.array(P), np.array(self.util_vals))
            # select top k entries of PC_coeff
            # TODO: check abs() correctness
            dims_to_keep = np.argpartition(np.abs(reg.coef_), -self.outcome_dim // 3)[
                -self.outcome_dim // 3 :    # TODO: update
            ]

            # transform corresponding to dims_to_keep
            self.pcr_axes = torch.tensor(torch.transpose(V[:, dims_to_keep], -2, -1))
            # then plug these into LinearProjection O/I transforms
            self.transforms_covar_dict["pcr"] = {
                "outcome_tf": LinearProjectionOutcomeTransform(self.pcr_axes),
                "input_tf": LinearProjectionInputTransform(self.pcr_axes),
                "covar_module": make_modified_kernel(
                    ard_num_dims=self.outcome_dim // 3 # TODO: update
                ),
            }

        else:
            outcome_model = fit_outcome_model(
                self.X,
                self.Y,
                outcome_transform=self.transforms_covar_dict[method]["outcome_tf"],
            )

            if method == "pca":

                self.pca_axes = outcome_model.outcome_transform["pca"].axes_learned

                self.transforms_covar_dict["pca"]["input_tf"] = ChainedInputTransform(
                    **{
                        # "standardize": InputStandardize(config["outcome_dim"]),
                        # TODO: was trying standardize again
                        "center": InputCenter(self.outcome_dim),
                        "pca": PCAInputTransform(axes=self.pca_axes),
                    }
                )

                self.lin_proj_latent_dim = self.pca_axes.shape[0]

                print(
                    f"amount of variance explained by {self.lin_proj_latent_dim} axes: {outcome_model.outcome_transform['pca'].PCA_explained_variance}"
                )

                # here we first see how many latent dimensions PCA learn
                # then we create a random linear projection mapping to the same dimensionality

                self.transforms_covar_dict["pca"]["covar_module"] = make_modified_kernel(
                    ard_num_dims=self.lin_proj_latent_dim
                )

                random_proj = generate_random_projection(
                    self.outcome_dim, self.lin_proj_latent_dim, dtype = self.dtype
                )
                self.transforms_covar_dict["random_linear_proj"] = {
                    "outcome_tf": LinearProjectionOutcomeTransform(random_proj),
                    "input_tf": LinearProjectionInputTransform(random_proj),
                    "covar_module": make_modified_kernel(ard_num_dims=self.lin_proj_latent_dim),
                }
                random_subset = random.sample(
                    range(self.outcome_dim), self.lin_proj_latent_dim
                )
                self.transforms_covar_dict["random_subset"] = {
                    "outcome_tf": SubsetOutcomeTransform(
                        outcome_dim=self.outcome_dim, subset=random_subset
                    ),
                    "input_tf": FilterFeatures(
                        feature_indices=torch.Tensor(random_subset).to(int)
                    ),
                    "covar_module": make_modified_kernel(ard_num_dims=self.lin_proj_latent_dim),
                }

        self.outcome_models_dict[method] = outcome_model


    def fit_pref_model(
        self,
        Y,
        comps,
        **model_kwargs
    ):
        util_model = PairwiseGP(datapoints=Y, comparisons=comps, **model_kwargs)

        mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)
        fit_gpytorch_model(mll_util)

        return util_model


    def run_pref_learning(self, method, pe_strategy):

        acqf_vals = []
        for i in range(self.every_n_comps):

            train_Y, train_comps = self.pref_data_dict[method][pe_strategy]

            print(f"Running {i+1}/{self.every_n_comps} preference learning using {pe_strategy}")

            fit_model_succeed = False
            pref_model_acc = None

            for _ in range(3):
                try:
                    pref_model = self.fit_pref_model(
                        train_Y,
                        train_comps,
                        input_transform=self.transforms_covar_dict[method]["input_tf"],
                        covar_module=self.transforms_covar_dict[method]["covar_module"],
                        # likelihood=likelihood,
                    )
                    # TODO: commented out to accelerate things
                    # pref_model_acc = check_pref_model_fit(
                    #     pref_model, problem=problem, util_func=util_func, n_test=1000, batch_eval=batch_eval
                    # )
                    print("Pref model fitting successful")
                    fit_model_succeed = True
                    break
                except (ValueError, RuntimeError):
                    continue
            if not fit_model_succeed:
                print(
                    "fit_pref_model() failed 3 times, stop current call of run_pref_learn()"
                )
                # return train_Y, train_comps, None, acqf_vals
                # TODO we don't want to return, just not change the current values

            if pe_strategy == "EUBO-zeta":
                # EUBO-zeta
                one_sample_outcome_model = ModifiedFixedSingleSampleModel(
                    model=self.outcome_model_dict[method],
                    outcome_dim=train_Y.shape[-1]
                )
                acqf = AnalyticExpectedUtilityOfBestOption(
                    pref_model=pref_model,
                    outcome_model=one_sample_outcome_model
                )
                found_valid_candidate = False
                for _ in range(3):
                    try:
                        cand_X, acqf_val = optimize_acqf(
                            acq_function=acqf,
                            q=2,
                            bounds=self.problem.bounds,
                            num_restarts=8,
                            raw_samples=64,  # used for intialization heuristic
                            options={"batch_limit": 4, "seed": self.trial_idx},
                        )
                        cand_Y = one_sample_outcome_model(cand_X)
                        acqf_vals.append(acqf_val.item())

                        found_valid_candidate = True
                        break
                    except (ValueError, RuntimeError):
                        continue

                if not found_valid_candidate:
                    print(
                        "optimize_acqf() failed 3 times for EUBO, stop current call of run_pref_learn()"
                    )
                    return train_Y, train_comps, None, acqf_vals

            elif pe_strategy == "Random-f":
                # Random-f
                cand_X = draw_sobol_samples(
                    bounds=self.problem.bounds, 
                    n=1, 
                    q=2,
                ).squeeze(0).to(torch.double)
                cand_Y = self.outcome_model_dict[method].posterior(cand_X).rsample().squeeze(0).detach()
            else:
                raise RuntimeError("Unknown preference exploration strategy!")

            cand_Y = cand_Y.detach().clone()
            cand_comps = gen_comps(self.util_func(cand_Y))

            train_comps = torch.cat((train_comps, cand_comps + train_Y.shape[0]))
            train_Y = torch.cat((train_Y, cand_Y))

        self.pref_data_dict[method][pe_strategy] = (train_Y, train_comps)

        # return train_Y, train_comps, pref_model_acc, acqf_vals
        # TODO: not logging pref_model_acc and acqf_vals now


    def find_max_posterior_mean(self, method, pe_strategy, num_pref_samples = 1):
        # TODO: understand whether we should increase num_pref_samples from 1!

        train_Y, train_comps = self.pref_data_dict[method][pe_strategy]
        # seed = self.trial_idx

        within_result = {}

        pref_model = self.fit_pref_model(
            Y = train_Y,
            comps = train_comps,
            input_transform = self.transforms_covar_dict[method]["input_tf"],
            covar_module = self.transforms_covar_dict[method]["covar_module"]
        )
        sampler = SobolQMCNormalSampler(num_pref_samples)
        pref_obj = LearnedObjective(pref_model=pref_model, sampler=sampler)

        # find experimental candidate(s) that maximize the posterior mean utility
        post_mean_cand_X = gen_exp_cand(
            outcome_model=self.outcome_model_dict[method],
            objective=pref_obj,
            problem=self.problem,
            q=1,
            acqf_name="posterior_mean",
            seed=self.trial_idx
        )

        post_mean_util = self.util_func(self.problem.evaluate_true(post_mean_cand_X)).item()
        print(f"True utility of posterior mean utility maximizer: {post_mean_util:.3f}")

        within_result = {
            "n_comps": train_comps.shape[0],
            "util": post_mean_util,
            "run_id": self.trial_idx,
            "pe_strategy": pe_strategy,
            "method": method,
        }

        return within_result


    def generate_final_candidate(self, method, pe_strategy):

        train_Y, train_comps = self.pref_data_dict[method][pe_strategy]

        pref_model = self.fit_pref_model(
            Y = train_Y,
            comps = train_comps,
            input_transform = self.transforms_covar_dict[method]["input_tf"],
            covar_module = self.transforms_covar_dict[method]["covar_module"]
        )
        sampler = SobolQMCNormalSampler(1)
        pref_obj = LearnedObjective(pref_model=pref_model, sampler=sampler)

        # find experimental candidate(s) that maximize the posterior mean utility
        cand_X = gen_exp_cand(
            outcome_model=self.outcome_model_dict[method],
            objective=pref_obj,
            problem=self.problem,
            q=1,
            acqf_name="qNEI",
            X=self.X,
            seed=self.trial_idx
        )

        qneiuu_util = self.util_func(self.problem.evaluate_true(cand_X)).item()
        print(
            f"{method}-{pe_strategy} qNEIUU candidate utility: {qneiuu_util:.5f}"
        )

        exp_result = {
            "candidate": cand_X,
            "candidate_util": qneiuu_util,
            "method": method,
            "strategy": pe_strategy,
            "run_id": self.trial_idx,
            # "time_consumed": time_consumed,
            # "outcome_model_mse": outcome_model_mse,
            # "pref_model_acc": pref_model_acc,
            # "outcome_model_fit_time": outcome_model_fitting_time,
        }

        # TODO: later log PCA subspace recovery diagnostics

        self.final_candidate_results[method][pe_strategy] = exp_result # TODO: doublecheck


    def generate_random_pref_data(self, method, n):

        X = (
            draw_sobol_samples(
                bounds=self.problem.bounds,
                n=1,
                q=2*n,
                seed=self.trial_idx # TODO: confirm
            )
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        Y = self.outcome_models_dict[method].posterior(X).rsample().squeeze(0).detach()
        util = self.util_func(Y)
        comps = gen_comps(util)

        for pe_strategy in self.pe_strategies:
            self.pref_data_dict[method][pe_strategy] = (Y, comps)


# ======== Putting it together into steps ========

    def run_first_experimentation_stage(self, method):

        self.fit_outcome_model(method)


    def run_PE_stage(self, method):
        # initial result stored in self.pref_data_dict
        self.generate_random_pref_data(method, n=1)

        for pe_strategy in self.pe_strategies:

            self.PE_session_results[method][pe_strategy] = []

            self.PE_session_results[method][pe_strategy].append(
                self.find_max_posterior_mean(method, pe_strategy)
            )
            for j in range(self.n_check_post_mean):
                self.run_pref_learning(method, pe_strategy)
                self.PE_session_results[method][pe_strategy].append(
                    self.find_max_posterior_mean(method, pe_strategy)
                )

    def run_second_experimentation_stage(self, method):
        for pe_strategy in self.pe_strategies:
            self.generate_final_candidate(method, pe_strategy)
        

# ======== BOPE loop ========

    def run_BOPE_loop(self):
        # handle multiple trials
        # have a flag for whether the fitting is successful or not

        # all methods use the same initial experimentation data
        self.generate_random_experiment_data(
            self.initial_experimentation_batch,
            compute_util = True
        )

        for method in self.methods:
            self.run_first_experimentation_stage(method)
            self.run_PE_stage(method)
            self.run_second_experimentation_stage(method)

        # TODO: double check this is correct way to save stuff
        torch.save(self.PE_session_results, self.save_dir + 'PE_session_results_trial='+ self.trial_idx + '.th')
        torch.save(self.final_candidate_results, self.save_dir + 'final_candidate_results_trial' + self.trial_idx + '.th')

