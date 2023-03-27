#!/usr/bin/env python3

import copy
import os
import random
import time
from collections import defaultdict
from typing import List

import botorch
import gpytorch
import numpy as np
import torch
from botorch.acquisition.objective import LearnedObjective
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.pairwise_gp import (PairwiseGP,
                                        PairwiseLaplaceMarginalLogLikelihood)
from botorch.models.transforms.input import (ChainedInputTransform,
                                             FilterFeatures, Normalize)
from botorch.models.transforms.outcome import (ChainedOutcomeTransform,
                                               Standardize)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import \
    ExactMarginalLogLikelihood
from sklearn.linear_model import LinearRegression

from low_rank_BOPE.src.diagnostics import (check_outcome_model_fit,
                                           check_util_model_fit,
                                           get_function_statistics,
                                           mc_max_outcome_error,
                                           mc_max_util_error)
from low_rank_BOPE.src.models import MultitaskGPModel, make_modified_kernel
from low_rank_BOPE.src.pref_learning_helpers import (
    ModifiedFixedSingleSampleModel, find_true_optimal_utility, gen_comps,
    gen_exp_cand)
from low_rank_BOPE.src.transforms import (LinearProjectionInputTransform,
                                          LinearProjectionOutcomeTransform,
                                          SubsetOutcomeTransform,
                                          compute_weights, fit_pca,
                                          generate_random_projection)


class RetrainingBopeExperiment:

    attr_list = {
        "pca_var_threshold": 0.95,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 20,
        "every_n_comps": 3,
        "verbose": True,
        "dtype": torch.double,
        "num_restarts": 20,
        "raw_samples": 128,
        "batch_limit": 4,
        "sampler_num_outcome_samples": 64,
        "maxiter": 1000,
        "initial_latent_dim": None,
        "min_stdv": 100000,
        "true_axes": None, # specify these for synthetic problems
    }

    def __init__(
        self,
        problem: torch.nn.Module,
        util_func: torch.nn.Module,
        methods: List[str],
        pe_strategies: List[str],
        trial_idx: int,
        output_path: str,
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

        # self.attr_list stores default values, then overwrite with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        print("BopeExperiment settings: ", self.__dict__)

        # pre-specified experiment metadata
        self.problem = problem.double()
        self.util_func = util_func
        self.pe_strategies = pe_strategies
        self.outcome_dim = problem.outcome_dim
        self.input_dim = problem._bounds.shape[-1]
        self.trial_idx = trial_idx
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if hasattr(self.problem, "true_axes"):
            self.true_axes = self.problem.true_axes
        self.subspace_methods_headers = ["uwpca", "wpca", "random_linear_proj", "spca"]

        if "uwpca" in methods:
            # run unweighted pca first, so that the learned latent_dim
            # informs the other methods with dim reduction
            self.methods = ["uwpca"] + [m for m in methods if m != "uwpca"]
            print('self.methods, ', self.methods)
        else:
            # if "uw_pca" is not run and initial_latent_dim is not specified
            # set self.initial_latent_dim by hand
            self.methods = methods
            if not self.initial_latent_dim:
                self.initial_latent_dim = self.outcome_dim // 3
        # NOTE: method is suffixed with "_rt" if retraining is to be done
        # e.g., "uwpca", "uwpca_rt"
        # the subspace is updated during PE only if the method ends with "rt"

        # logging model specifics
        self.outcome_models_dict = {}  # by (method, pe_strategy)
        self.util_models_dict = {} # by (method, pe_strategy)
        self.pref_data_dict = defaultdict(dict)  # (Y, util_vals, comps) by (method, pe_strategy)
        self.projections_dict = {} # stores projections to subspaces, by (method, pe_strategy)
        self.transforms_covar_dict = {} # specify outcome and input transforms, covariance modules, by (method, pe_strategy)

        # log results
        self.PE_time_dict = {}
        self.PE_session_results = defaultdict(dict) # [method][pe_strategy]
        self.final_candidate_results = defaultdict(dict) # [method][pe_strategy]
        self.subspace_diagnostics = defaultdict(lambda: defaultdict(list)) # [(method, pe_strategy)]
       
        # estimate true optimal utility through sampling
        self.true_opt = find_true_optimal_utility(self.problem, self.util_func, n=5000)
        # TODO: replace this with a more cmprehensive function statistics profile (e.g., min, max, median, quantiles)
        # get_function_statistics()

    def generate_random_experiment_data(self, n, compute_util = True):
        r"""Generate n observations of experimental designs and outcomes.
        This is shared by all methods. 
        Args:
            n: number of samples
            compute_util: if True, also observe comparisons between observed 
                outcome as well as save their true utility values
        Computes:
            X: `n x problem input dim` tensor of sampled inputs
            Y: `n x problem outcome dim` tensor of noisy evaluated outcomes at X
            util_vals: `n x 1` tensor of utility value of Y
            comps: `n/2 x 2` tensor of pairwise comparisons 
        """

        self.X = (
            draw_sobol_samples(bounds=self.problem.bounds, n=1, q=n, seed=self.trial_idx)
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        self.Y = self.problem(self.X).detach()

        if compute_util:
            util_vals = self.util_func(self.Y).detach()
            comps = gen_comps(util_vals)

            # put into pref_data_dict for different (method, pe_strategy) tuples
            for method in self.methods:
                for pe_strategy in self.pe_strategies:
                    self.pref_data_dict[(method, pe_strategy)] = {
                        "Y": self.Y,
                        "util_vals": util_vals,
                        "comps": comps
                    }

    def compute_projections(self, method, pe_strategy):

        projection = None
        Y = self.pref_data_dict[(method, pe_strategy)]["Y"]

        if method.startswith("uwpca"): 
            # could be "uwpca" or "uwpca_rt"

            projection = fit_pca(
                Y, 
                var_threshold=self.pca_var_threshold, 
                weights=None
            ) 

            if method == "uwpca": # i.e., no retraining, just fit subspace once
                self.initial_latent_dim = projection.shape[0]

        elif method.startswith("wpca_true"):
            # could be "wpca_true" or "wpca_true_rt"
            
            util_vals = self.pref_data_dict[(method, pe_strategy)]["util_vals"]
            weights = compute_weights(util_vals.squeeze(1))

            projection = fit_pca(
                Y, 
                var_threshold=self.pca_var_threshold,
                weights=weights
            ) 

        # TODO: currently not running well, come back to this
        elif method.startswith("wpca_est"):
            if "w_pca_est" in self.util_models_dict:
                # use posterior mean as utility value estimate, if a model exists
                util_vals_est = self.util_models_dict["w_pca_est"].posterior(Y).mean
                weights = compute_weights(util_vals_est.squeeze(1))
            else:
                # otherwise, use uniform weights
                weights = torch.ones((Y.shape[0],1))
            projection = fit_pca(
                Y, 
                var_threshold=self.pca_var_threshold, 
                weights=weights
            ) 

        elif method == "st":
            self.transforms_covar_dict[(method, pe_strategy)] = {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.outcome_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.outcome_dim),
            }
        
        elif method.startswith("spca_true"):
            # this is what we used to call "pcr"
            # could be "spca_true", "spca_true_rt"

            P, _, V = torch.svd(Y - Y.mean(dim=0))

            # then run regression from P (PCs) onto util_vals
            reg = LinearRegression().fit(
                np.array(P), 
                np.array(self.pref_data_dict[(method, pe_strategy)]["util_vals"])
            ) 
            
            # select top `self.initial_latent_dim` entries of PC_coeff
            # TODO: not sure this is the best thing to do, 
            # alternative is to have it always align with uwpca
            dims_to_keep = np.argsort(np.abs(reg.coef_))[-self.initial_latent_dim:]
            print('dims_to_keep: ', dims_to_keep)
            if len(dims_to_keep.shape) == 2:
                dims_to_keep = dims_to_keep[0]
            print('dims_to_keep after processing: ', dims_to_keep) 
            # retain the corresponding columns in V
            self.projections_dict[(method, pe_strategy)] = torch.tensor(np.transpose(V[:, dims_to_keep]))

        elif method == "spca_est":
            pass


        elif method == "random_linear_proj":
            projection = generate_random_projection(
                self.outcome_dim, self.initial_latent_dim, dtype=self.dtype
            )
        
        elif method == "true_proj":
            projection = self.true_axes

        elif method == "random_subset":
            random_subset = random.sample(
                range(self.outcome_dim), self.initial_latent_dim
            )
            self.transforms_covar_dict["random_subset"] = {
                "outcome_tf": SubsetOutcomeTransform(
                    outcome_dim=self.outcome_dim, subset=random_subset
                ),
                "input_tf": FilterFeatures(
                    feature_indices=torch.Tensor(random_subset).to(int)
                ),
                "covar_module": make_modified_kernel(ard_num_dims=self.initial_latent_dim),
            }

        if projection is not None:
            self.projections_dict[(method, pe_strategy)] = projection

            self.transforms_covar_dict[(method, pe_strategy)] = {
                "outcome_tf": LinearProjectionOutcomeTransform(projection),
                "input_tf": LinearProjectionInputTransform(projection),
                "covar_module": make_modified_kernel(ard_num_dims=projection.shape[0]),
            }

            # save subspace diagnostics
            self.subspace_diagnostics[(method, pe_strategy)]["latent_dim"].append(projection.shape[0])

            
    def fit_outcome_model(self, method, pe_strategy): 
        r"""Fit outcome model based on specified method.
        Args:
            method: string specifying the statistical model
            pe_strategy
        
        """

        print(f"Fitting outcome model using {method} and {pe_strategy}")

        start_time = time.time()

        outcome_model = SingleTaskGP(
            train_X=self.X, 
            train_Y=self.Y, 
            outcome_transform = copy.deepcopy(
                self.transforms_covar_dict[(method, pe_strategy)]["outcome_tf"]
            )
        )
        mll_outcome = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
        fit_gpytorch_mll(mll_outcome)        

        # should we observe comparisons in first stage? suppose we do, then we can initialize the util model w more data
        
        model_fitting_time = time.time() - start_time
        rel_mse = check_outcome_model_fit(outcome_model, self.problem, n_test=1000) # TODO: double check MSE metric we are using
        self.subspace_diagnostics[(method, pe_strategy)]["model_fitting_time"].append(model_fitting_time)
        self.subspace_diagnostics[(method, pe_strategy)]["rel_mse"].append(rel_mse)

        self.outcome_models_dict[(method, pe_strategy)] = outcome_model


    def fit_util_model(
        self,
        Y,
        comps,
        **model_kwargs
    ):
        r"""Fit utility model based on given data and model_kwargs
        Args:
            Y: `num_samples x outcome_dim` tensor of outcomes
            comps: `num_samples/2 x 2` tensor of pairwise comparisons of Y data
            model_kwargs: input transform and covar_module
        """
        util_model = PairwiseGP(
            datapoints=Y, comparisons=comps, **model_kwargs)

        mll_util = PairwiseLaplaceMarginalLogLikelihood(
            util_model.likelihood, util_model)
        fit_gpytorch_mll(mll_util)

        return util_model

    def run_pref_learning(self, method, pe_strategy):

        acqf_vals = []
        for i in range(self.every_n_comps):

            train_Y = self.pref_data_dict[(method, pe_strategy)]["Y"]
            train_comps = self.pref_data_dict[(method, pe_strategy)]["comps"]
            train_util_vals = self.pref_data_dict[(method, pe_strategy)]["util_vals"]

            print(
                f"== Running {i+1}/{self.every_n_comps} preference learning using {pe_strategy}")

            fit_model_succeed = False
            util_model_acc = None

            for _ in range(3):
                try:
                    util_model = self.fit_util_model(
                        train_Y,
                        train_comps,
                        input_transform=self.transforms_covar_dict[(method, pe_strategy)]["input_tf"],
                        covar_module=self.transforms_covar_dict[(method, pe_strategy)]["covar_module"],
                    )
                    # TODO: commented out to accelerate things
                    # util_model_acc = check_util_model_fit(
                    #     util_model, problem=problem, util_func=util_func, n_test=1000, batch_eval=batch_eval
                    # )
                    print("Pref model fitting successful")
                    fit_model_succeed = True
                    break
                except (ValueError, RuntimeError):
                    continue
            if not fit_model_succeed:
                print(
                    "fit_util_model() failed 3 times, stop current call of run_pref_learn()"
                )

            if pe_strategy == "EUBO-zeta":
                with botorch.settings.debug(state=True):
                    # EUBO-zeta
                    one_sample_outcome_model = ModifiedFixedSingleSampleModel(
                        model=self.outcome_models_dict[(method, pe_strategy)],
                        outcome_dim=train_Y.shape[-1]
                    ).to(torch.double) 
                    acqf = AnalyticExpectedUtilityOfBestOption(
                        pref_model=util_model,
                        outcome_model=one_sample_outcome_model
                    ).to(torch.double) 
                    acqf_landscape = get_function_statistics(
                        function=acqf, bounds=self.problem.bounds)
                    found_valid_candidate = False
                    for _ in range(3):
                        try:
                            cand_X, acqf_val = optimize_acqf(
                                acq_function=acqf,
                                q=2,
                                bounds=self.problem.bounds,
                                num_restarts=self.num_restarts,
                                raw_samples=self.raw_samples,  # used for intialization heuristic
                                options={"batch_limit": 4, "seed": self.trial_idx},
                            )
                            cand_Y = one_sample_outcome_model(cand_X)
                            acqf_vals.append(acqf_val.item())

                            found_valid_candidate = True
                            print("EUBO mean, sd, min_val, max_val, quantile_vals: ", acqf_landscape)
                            print("EUBO candidate acqf value: ", acqf_val)
                            break
                        except (ValueError, RuntimeError) as error:
                            print("error in optimizing EUBO: ", error)
                            continue
                    if not found_valid_candidate:
                        print(
                            f"optimize_acqf() failed 3 times for EUBO with {method},", 
                            "stop current call of run_pref_learn()"
                        )
                        # return train_Y, train_comps, None, acqf_vals
                        return

            elif pe_strategy == "Random-f":
                # Random-f
                cand_X = draw_sobol_samples(
                    bounds=self.problem.bounds,
                    n=1,
                    q=2,
                ).squeeze(0).to(torch.double)
                cand_Y = self.outcome_models_dict[(method, pe_strategy)].posterior(
                    cand_X).rsample().squeeze(0).detach()
            else:
                raise RuntimeError("Unknown preference exploration strategy!")

            cand_Y = cand_Y.detach().clone()
            cand_util_val = self.util_func(cand_Y)
            cand_comps = gen_comps(cand_util_val)
            print("EUBO selected candidate util val: ", cand_util_val)
            

            train_comps = torch.cat(
                (train_comps, cand_comps + train_Y.shape[0])
            )
            train_Y = torch.cat((train_Y, cand_Y))
            train_util_vals = torch.cat((train_util_vals, cand_util_val))
            print('train_Y, train_comps shape: ', train_Y.shape, train_comps.shape)

            self.pref_data_dict[(method, pe_strategy)] = {
                "Y": train_Y,
                "util_vals": train_util_vals,
                "comps": train_comps
            }
            

    def find_max_posterior_mean(self, method, pe_strategy, num_pref_samples=1):

        train_Y = self.pref_data_dict[(method, pe_strategy)]["Y"]
        train_comps = self.pref_data_dict[(method, pe_strategy)]["comps"]

        within_result = {}

        # NOTE: I thought about whether we could save computation by avoiding double-fitting
        # the same pref model here and in the next call of run_pref_learning()
        # But this function is called every three times run_pref_learning() is called
        # so the amount of duplicated effort is not large 
        util_model = self.fit_util_model(
            Y=train_Y,
            comps=train_comps,
            input_transform=self.transforms_covar_dict[(method, pe_strategy)]["input_tf"],
            covar_module=self.transforms_covar_dict[(method, pe_strategy)]["covar_module"]
        )
        sampler = SobolQMCNormalSampler(num_pref_samples)
        pref_obj = LearnedObjective(pref_model=util_model, sampler=sampler)

        # find experimental candidate(s) that maximize the posterior mean utility
        post_mean_cand_X = gen_exp_cand(
            outcome_model=self.outcome_models_dict[(method, pe_strategy)],
            objective=pref_obj,
            problem=self.problem,
            q=1,
            acqf_name="posterior_mean",
            seed=self.trial_idx
        )

        post_mean_util = self.util_func(
            self.problem.evaluate_true(post_mean_cand_X)).item()
        print(
            f"True utility of posterior mean utility maximizer: {post_mean_util:.3f}")
        util_posterior_landscape = get_function_statistics(
            function=util_model, bounds=self.problem.outcome_bounds)
        print("util posterior mean function mean, sd, min_val, max_val, quantile_vals: ", 
              util_posterior_landscape)

        within_result = {
            "n_comps": train_comps.shape[0],
            "util": post_mean_util,
            "run_id": self.trial_idx,
            "pe_strategy": pe_strategy,
            "method": method,
            "candidate": post_mean_cand_X
        }

        # check util model fit here
        util_model_acc = check_util_model_fit(
            util_model, self.problem, self.util_func, 
            n_test=1000, batch_eval=True)
        within_result["util_model_acc"] = util_model_acc

        return within_result

    def generate_final_candidate(self, method, pe_strategy):

        train_Y = self.pref_data_dict[(method, pe_strategy)]["Y"]
        train_comps = self.pref_data_dict[(method, pe_strategy)]["comps"]

        util_model = self.fit_util_model(
            Y=train_Y,
            comps=train_comps,
            input_transform=self.transforms_covar_dict[(method, pe_strategy)]["input_tf"],
            covar_module=self.transforms_covar_dict[(method, pe_strategy)]["covar_module"]
        )
        # log accuracy of final utility model
        util_model_acc = check_util_model_fit(
            util_model, self.problem, self.util_func, 
            n_test=1000, batch_eval=True)

        sampler = SobolQMCNormalSampler(1)
        pref_obj = LearnedObjective(pref_model=util_model, sampler=sampler)

        # find experimental candidate(s) that maximize the posterior mean utility
        cand_X = gen_exp_cand(
            outcome_model=self.outcome_models_dict[(method, pe_strategy)],
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
            "PE_time": self.PE_time_dict[(method, pe_strategy)],
            "util_model_acc": util_model_acc,
        }
        
        # log the true optimal utility computed in __init__()
        exp_result["true_opt"] = self.true_opt

        self.final_candidate_results[method][pe_strategy] = exp_result

    def compute_subspace_diagnostics(self, method, pe_strategy, n_test = 1000):
        # log diagnostics for recovering outcome and utility from the subspace

        projection = self.projections_dict[(method, pe_strategy)] 

        max_outcome_error = mc_max_outcome_error(
            problem=self.problem,
            axes_learned=projection,
            n_test=n_test
        )
        
        max_util_error = mc_max_util_error(
            problem=self.problem,
            axes_learned=projection,
            util_func=self.util_func,
            n_test=n_test
        )
        
        self.subspace_diagnostics[(method, pe_strategy)]["max_util_error"].append(max_util_error)
        self.subspace_diagnostics[(method, pe_strategy)]["max_outcome_error"].append(max_outcome_error)
        

    def generate_random_pref_data(self, method, pe_strategy, n):

        X = (
            draw_sobol_samples(
                bounds=self.problem.bounds,
                n=1,
                q=2*n,
                seed=self.trial_idx 
            )
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        Y = self.outcome_models_dict[(method, pe_strategy)].posterior(
            X).rsample().squeeze(0).detach()
        util_val = self.util_func(Y)
        comps = gen_comps(util_val)

        self.pref_data_dict[(method, pe_strategy)]["comps"] = torch.cat(
            (self.pref_data_dict[(method, pe_strategy)]["comps"],
            comps + self.pref_data_dict[(method, pe_strategy)]["Y"].shape[0])
        )
        self.pref_data_dict[(method, pe_strategy)]["Y"] = torch.cat(
            (self.pref_data_dict[(method, pe_strategy)]["Y"], Y)
        )
        self.pref_data_dict[(method, pe_strategy)]["util_vals"] = torch.cat(
            (self.pref_data_dict[(method, pe_strategy)]["util_vals"], util_val)
        )



# ======== Putting it together into steps ========


    def run_first_experimentation_stage(self, method):

        for pe_strategy in self.pe_strategies:
            self.compute_projections(method, pe_strategy)
            self.fit_outcome_model(method, pe_strategy)        

    def run_PE_stage(self, method):
        # initial result stored in self.pref_data_dict
        

        for pe_strategy in self.pe_strategies:

            self.generate_random_pref_data(method, pe_strategy, n=1)
            # TODO: if we also gather comps in the first stage, is this step needed?

            if any(method.startswith(header) for header in self.subspace_methods_headers):
                self.compute_subspace_diagnostics(method, pe_strategy, n_test=1000)

            start_time = time.time()

            print(f"===== Running PE using {method} with {pe_strategy} =====")

            self.PE_session_results[method][pe_strategy] = []
            self.PE_session_results[method][pe_strategy].append(
                self.find_max_posterior_mean(method, pe_strategy)
            )

            for j in range(self.n_check_post_mean):

                self.run_pref_learning(method, pe_strategy)
                self.PE_session_results[method][pe_strategy].append(
                    self.find_max_posterior_mean(method, pe_strategy)
                )
                # relearn subspace if method calls for retraining
                # TODO: check correctness
                if method.endswith("rt"):
                    print("Retraining")
                    self.compute_projections(method, pe_strategy)
                    self.fit_outcome_model(method, pe_strategy)
                    # run diagnostics on the updated subspaces
                    self.compute_subspace_diagnostics(method, pe_strategy, n_test=1000)
            
            # log time required to do PE
            PE_time = time.time() - start_time
            self.PE_time_dict[(method, pe_strategy)] = PE_time # will be logged later


    def run_second_experimentation_stage(self, method):
        for pe_strategy in self.pe_strategies:
            print(f"===== Generating final candidate using {method} with {pe_strategy} =====")
            self.generate_final_candidate(method, pe_strategy)


# ======== BOPE loop ========


    def run_BOPE_loop(self):
        # handle multiple trials
        # have a flag for whether the fitting is successful or not

        # all methods use the same initial experimentation data
        self.generate_random_experiment_data(
            self.initial_experimentation_batch,
            compute_util=True
        )

        for method in self.methods:
            try:
                print(f"============= Running {method} =============")
                self.run_first_experimentation_stage(method)
                self.run_PE_stage(method)
                self.run_second_experimentation_stage(method)

                torch.save(self.PE_session_results, self.output_path +
                        'PE_session_results_trial=' + str(self.trial_idx) + '.th')
                torch.save(self.final_candidate_results, self.output_path +
                        'final_candidate_results_trial=' + str(self.trial_idx) + '.th')
                torch.save(self.pref_data_dict, self.output_path +
                        'pref_data_trial=' + str(self.trial_idx) + '.th')
                torch.save(self.subspace_diagnostics, self.output_path +
                        'subspace_diagnostics_trial=' + str(self.trial_idx) + '.th')
            
            except:
                print(f"============= {method} failed, skipping =============")
                continue
