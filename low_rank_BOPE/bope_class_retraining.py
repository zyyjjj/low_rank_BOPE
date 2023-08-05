#!/usr/bin/env python3

import copy
import os
import random
import time
from collections import defaultdict
from typing import List, Optional
import traceback

import botorch
import numpy as np
import torch
from botorch.acquisition.objective import LearnedObjective
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.logging import logger
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

from low_rank_BOPE.src.diagnostics import (best_and_avg_util_in_subspace,
                                           check_outcome_model_fit,
                                           check_overall_fit,
                                           check_util_model_fit,
                                           get_function_statistics,
                                           mc_max_outcome_error,
                                           mc_max_util_error, 
                                           compute_grassmannian)
from low_rank_BOPE.src.models import MultitaskGPModel, make_modified_kernel
from low_rank_BOPE.src.pref_learning_helpers import (
    ModifiedFixedSingleSampleModel, find_true_optimal_utility, gen_comps,
    gen_exp_cand)
from low_rank_BOPE.src.transforms import (LinearProjectionInputTransform,
                                          LinearProjectionOutcomeTransform,
                                          SubsetOutcomeTransform,
                                          compute_weights, fit_pca,
                                          generate_random_projection)


def defaultdict_list():
    return defaultdict(list)

class RetrainingBopeExperiment:

    attr_list = {
        "pca_var_threshold": 0.95,
        "num_PCs": None, # if num_PCs is specified, ignore pca_var_threshold
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 20,
        "every_n_comps": 3,
        "n_BO_iters": 10,
        "BO_batch_size": 1,
        "n_meta_iters": 1, 
        "dtype": torch.double,
        "num_restarts": 20,
        "raw_samples": 128,
        "batch_limit": 4,
        "sampler_num_outcome_samples": 128,
        "maxiter": 1000,
        "initial_latent_dim": None,
        "min_stdv": 100000,
        "true_axes": None, # specify these for synthetic problems
        "standardize": True,
        "wpca_type": "rank_cts",
        "wpca_options": {"k": 10, "num_points_to_discard": 2},
        "compute_true_opt": False,
        "save_results": True,
    }

    def __init__(
        self,
        problem: torch.nn.Module,
        util_func: torch.nn.Module,
        methods: List[str],
        pe_strategies: List[str],
        trial_idx: int,
        output_path: Optional[str] = None,
        **kwargs
    ) -> None:
        r"""
        Initialize RetrainingBopeExperiment class.
        Args:
            problem: outcome function
            util_func: utility function
            methods: list of methods to use
            pe_strategies: list of pe_strategies to use
            trial_idx: index of the trial; controls randomness
            output_path: path to save the experiment results to
        """

        # self.attr_list stores default values, then overwrite with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        logger.info(f"RetrainingBopeExperiment settings: {self.__dict__}")

        # pre-specified experiment metadata
        self.problem = problem.double()
        self.util_func = util_func
        self.pe_strategies = pe_strategies
        self.outcome_dim = problem.outcome_dim
        self.input_dim = problem.bounds.shape[-1]
        self.trial_idx = trial_idx
        self.output_path = output_path
        if (self.output_path is not None) and (not os.path.exists(self.output_path)):
            os.makedirs(self.output_path, exist_ok=True)
        if hasattr(self.problem, "true_axes"):
            self.true_axes = self.problem.true_axes
        self.subspace_methods_headers = ["pca", "wpca", "random_linear_proj", "spca"]

        if "pca" in methods:
            # run unweighted pca first, so that the learned latent_dim
            # informs the other methods with dim reduction
            # (random_linear_proj, random_subset, spca)
            self.methods = ["pca"] + [m for m in methods if m != "pca"]
            logger.info(f"self.methods: {self.methods}")
        else:
            # if "pca" is not run and initial_latent_dim is not specified
            # set self.initial_latent_dim by hand
            self.methods = methods
            if self.initial_latent_dim is None:
                self.initial_latent_dim = self.outcome_dim // 3
        # NOTE: method is suffixed "_rt" if retraining
        # the subspace is updated during PE only if the method ends with "rt"

        # logging model specifics
        self.outcome_models_dict = {}  # by (method, pe_strategy)
        self.util_models_dict = {} # by (method, pe_strategy)
        self.pref_data_dict = defaultdict(dict)  # (Y, util_vals, comps) by (method, pe_strategy)
        self.projections_dict = defaultdict(list) # stores list of projections to subspaces, by (method, pe_strategy)
        self.transforms_covar_dict = {} # specify outcome and input transforms, covariance modules, by (method, pe_strategy)
        self.subspace_training_Y = {} # outcome data used for subspace learning, by (method, pe_strategy)
        self.BO_data_dict = defaultdict(dict) # stores observed (X,Y) in initial batch and BO, by (method, pe_strategy)

        # save results
        self.PE_session_results = defaultdict(defaultdict_list) # [method][pe_strategy] 
        self.final_candidate_results = defaultdict(defaultdict_list) # [method][pe_strategy]
        self.subspace_diagnostics = defaultdict(defaultdict_list) # [(method, pe_strategy)][diag_metric] = list
        # self.util_postmean_landscape = defaultdict(list) # [(method, pe_strategy)] = list of statistics tuples
        self.time_consumption = defaultdict(defaultdict_list) # [(method, pe_strategy)][time_metric] = list
        
        # initialize progress checkpoint dict
        self.progress = {}
        for method in self.methods:
            self.progress[method] = {"meta_iter": -1, "stage": "initial"}

        # estimate true optimal utility through sampling
        if self.compute_true_opt:
            self.true_opt = find_true_optimal_utility(self.problem, self.util_func, n=5000)
        else:
            self.true_opt = None

        # load partial experiment data if it exists
        if self.save_results:
            self.load_experiment_data()


    def generate_random_experiment_data(self, n: int, compute_util: bool = True):
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

        self.initial_X = (
            draw_sobol_samples(bounds=self.problem.bounds, n=1, q=n, seed=self.trial_idx)
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        self.initial_Y = self.problem(self.initial_X).detach()

        if compute_util:
            util_vals = self.util_func(self.initial_Y).detach()
            comps = gen_comps(util_vals)

            # put into pref_data_dict for different (method, pe_strategy) tuples
            for method in self.methods:
                for pe_strategy in self.pe_strategies[method]:
                    if pe_strategy == "TS": # TODO: is this the right way to distinguish things? 
                        self.pref_data_dict[(method, pe_strategy)] = {
                            "X": self.initial_X,
                            "Y": self.initial_Y,
                            "util_vals": util_vals,
                            "comps": comps
                        }
                    else:
                        self.pref_data_dict[(method, pe_strategy)] = {
                            "Y": self.initial_Y,
                            "util_vals": util_vals,
                            "comps": comps
                        }
                    self.subspace_training_Y[(method, pe_strategy)] = self.initial_Y
        
        # save the data to BO_data_dict
        for method in self.methods:
            for pe_strategy in self.pe_strategies[method]:
                self.BO_data_dict[(method, pe_strategy)] = {
                    "X": self.initial_X,
                    "Y": self.initial_Y
                }


    def compute_projections(self, method: str, pe_strategy: str):
        r"""
        - Compute projections to subspace for given method and pe_strategy. 
        - Store the projection in self.projections_dict (append to list).
        - Update self.transforms_covar_dict with the computed projection.
        - Save subspace diagnostics in self.subspace_diagnostics.
        """

        logger.info(f"    -- Computing subspace for method [{method}] and pe_strategy [{pe_strategy}]")

        if method == "random_search":
            return
        
        projection = None

        train_Y = self.BO_data_dict[(method, pe_strategy)]["Y"]

        if method.startswith("pca"): 

            if method in ("pca", "pca_rt", "pca_norefit_rt"):
                Y_selected = train_Y
            else:
                # methods that include fake points from outcome model posterior 
                # pca_all_rt, pca_eubo_rt, pca_postquant_rt, pca_postmax_rt
                # these are deprecated for now
                Y_selected = self.subspace_training_Y[(method, pe_strategy)]
            
            logger.debug(f"        -- shape of Y_selected for computing subspace: {Y_selected.shape}")
            
            projection = fit_pca(
                Y_selected,
                var_threshold=self.pca_var_threshold, 
                num_PCs=self.num_PCs,
                weights=None,
                standardize=self.standardize
            ) 

            if method == "pca": # i.e., no retraining, just fit subspace once
                self.initial_latent_dim = projection.shape[0]

        elif method.startswith("wpca_true"):
            # could be "wpca_true" or "wpca_true_rt"
            # though weighting without retraining isn't really reasonable

            util_vals = self.util_func(train_Y).detach()
            weights = compute_weights(
                    util_vals.squeeze(1), 
                    weights_type=self.wpca_type,
                    wpca_options=self.wpca_options
                )
            logger.debug(f"weights: {weights}")

            projection = fit_pca(
                train_Y, 
                var_threshold=self.pca_var_threshold,
                num_PCs=self.num_PCs,
                weights=weights,
                standardize=self.standardize
            ) 

        elif method.startswith("wpca_est"):

            if (method, pe_strategy) in self.util_models_dict:
                # use posterior mean as utility value estimate, if a model exists
                logger.info("        -- Using posterior mean of util model to compute weights")

                util_vals_est = self.util_models_dict[(method, pe_strategy)].posterior(train_Y).mean.detach()

                logger.info(f"        -- train_Y.shape: {train_Y.shape}, util_vals_est.shape: {util_vals_est.shape}")

                weights = compute_weights(
                    util_vals_est.squeeze(1), 
                    weights_type=self.wpca_type,
                    wpca_options=self.wpca_options
                )
            else:
                # otherwise, use uniform weights
                weights = torch.ones((train_Y.shape[0],1))
            
            projection = fit_pca(
                train_Y, 
                var_threshold=self.pca_var_threshold, 
                num_PCs=self.num_PCs,
                weights=weights,
                standardize=self.standardize
            ) 

        elif method == "st":
            self.transforms_covar_dict[(method, pe_strategy)] = {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.outcome_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.outcome_dim),
            }
        
        elif method == "pbo":
            self.transforms_covar_dict[(method, pe_strategy)] = {
                "outcome_tf": Standardize(self.outcome_dim),
                "input_tf": Normalize(self.input_dim),
                "covar_module": make_modified_kernel(ard_num_dims=self.input_dim),
            }
        
        elif method.startswith("spca_true"):
            # this is what we used to call "pcr"
            # could be "spca_true", "spca_true_rt"

            train_Y_centered = train_Y - train_Y.mean(dim=0)
            if self.standardize:
                P, _, V = torch.svd(train_Y_centered/train_Y_centered.std(dim=0))
            else:
                P, _, V = torch.svd(train_Y_centered)

            util_vals = self.util_func(train_Y).detach()

            # then run regression from P (PCs) onto util_vals
            reg = LinearRegression().fit(
                np.array(P), 
                np.array(util_vals)
            ) 
            
            # select top `self.initial_latent_dim` entries of PC_coeff
            dims_to_keep = np.argsort(np.abs(reg.coef_))[-self.initial_latent_dim:]
            logger.info(f"        -- dims_to_keep: {dims_to_keep}")
            if len(dims_to_keep.shape) == 2:
                dims_to_keep = dims_to_keep[0]
            logger.info(f"        -- dims_to_keep after processing: {dims_to_keep}") 
            # retain the corresponding columns in V
            projection = torch.tensor(np.transpose(V[:, dims_to_keep]))

        elif method.startswith("spca_est"): 
            if (method, pe_strategy) in self.util_models_dict:
                util_vals_est = self.util_models_dict[(method, pe_strategy)].posterior(train_Y).mean.detach()
                train_Y_centered = train_Y - train_Y.mean(dim=0)
                if self.standardize:
                    P, _, V = torch.svd(train_Y_centered/train_Y_centered.std(dim=0))
                else:
                    P, _, V = torch.svd(train_Y_centered)

                reg = LinearRegression().fit(
                    np.array(P), 
                    np.array(util_vals_est)
                ) 

                dims_to_keep = np.argsort(np.abs(reg.coef_))[-self.initial_latent_dim:]
                if len(dims_to_keep.shape) == 2:
                    dims_to_keep = dims_to_keep[0]
                logger.info(f"        -- dims_to_keep: {dims_to_keep}") 
                # retain the corresponding columns in V
                projection = torch.tensor(np.transpose(V[:, dims_to_keep]))

            else:
                # initialize with just PCA
                projection = fit_pca(
                    train_Y,
                    var_threshold=self.pca_var_threshold, 
                    weights=None,
                    standardize=self.standardize
                ) 

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
            self.projections_dict[(method, pe_strategy)].append(projection)

            self.transforms_covar_dict[(method, pe_strategy)] = {
                "outcome_tf": ChainedOutcomeTransform(
                    **{
                        "projection": LinearProjectionOutcomeTransform(projection),
                        "standardize": Standardize(projection.shape[0])
                    }
                ), 
                "input_tf": ChainedInputTransform(
                    **{
                        "projection": LinearProjectionInputTransform(projection),
                        "normalize": Normalize(projection.shape[0])
                    }
                ),
                "covar_module": make_modified_kernel(ard_num_dims=projection.shape[0]),
            }

            # save subspace diagnostics
            self.subspace_diagnostics[(method, pe_strategy)]["latent_dim"].append(projection.shape[0])

            
    def fit_outcome_model(
            self, method: str, pe_strategy: str, 
            save_model_fit_time: bool = True, save_diagnostics: bool = True): 
        r"""
        - Fit outcome model for given method and pe_strategy.
        - Save the outcome model in self.outcome_models_dict.
        - Save diagnostics such as fitting error in self.subspace_diagnostics.
        """

        if method == "random_search":
            return

        # get existing data from self.BO_data_dict 
        train_X = self.BO_data_dict[(method, pe_strategy)]["X"]
        train_Y = self.BO_data_dict[(method, pe_strategy)]["Y"]

        logger.info(f"    -- Fitting outcome model using [{method}] and [{pe_strategy}]")
        logger.debug(f"        -- train_X.shape: {train_X.shape}", f"train_Y.shape: {train_Y.shape}")
        
        outcome_model = SingleTaskGP(
            train_X=train_X, 
            train_Y=train_Y, 
            outcome_transform = copy.deepcopy(
                self.transforms_covar_dict[(method, pe_strategy)]["outcome_tf"]
            )
        )
        mll_outcome = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
        
        start_time = time.time()
        fit_gpytorch_mll(mll_outcome)        
        model_fitting_time = time.time() - start_time
        if save_model_fit_time:
            self.time_consumption[(method, pe_strategy)]["outcome_model_fitting_time"].append(model_fitting_time)
        logger.info(f"        -- Outcome model fitting time: {model_fitting_time:.2f} seconds")

        if save_diagnostics:
            rel_mse = check_outcome_model_fit(outcome_model, self.problem, n_test=1000)
            self.subspace_diagnostics[(method, pe_strategy)]["rel_mse"].append(rel_mse)

        self.outcome_models_dict[(method, pe_strategy)] = outcome_model


    def fit_util_model(
            self, method: str, pe_strategy: str, 
            fit_on_X: bool=False,
            save_model: bool = False, save_model_fit_time: bool=True):
        r"""Fit and return utility model for given method and pe_strategy."""

        logger.info(f"    -- Fitting util model")

        # train_X = self.BO_data_dict[(method, pe_strategy)]["X"] # TODO: should be pref_data_dict, not BO_data_dict
        train_X = self.pref_data_dict[(method, pe_strategy)].get("X", None)
        train_Y = self.pref_data_dict[(method, pe_strategy)]["Y"]
        train_comps = self.pref_data_dict[(method, pe_strategy)]["comps"]

        if fit_on_X:
            datapoints = train_X
        else:
            datapoints = train_Y

        util_model = PairwiseGP(
            datapoints=datapoints, 
            comparisons=train_comps, 
            input_transform=copy.deepcopy(
                self.transforms_covar_dict[(method, pe_strategy)]["input_tf"]
            ),
            covar_module=copy.deepcopy(
                self.transforms_covar_dict[(method, pe_strategy)]["covar_module"],
            )        
        )

        mll_util = PairwiseLaplaceMarginalLogLikelihood(
            util_model.likelihood, util_model)
        start_time = time.time()
        fit_gpytorch_mll(mll_util)
        model_fitting_time = time.time() - start_time
        logger.info(f"        -- Util model fitting time: {model_fitting_time:.2f} seconds")

        if save_model:
            self.util_models_dict[(method, pe_strategy)] = util_model # TODO: will this increase memory consumption a lot?

        if save_model_fit_time:
            self.time_consumption[(method, pe_strategy)]["util_model_fitting_time"].append(model_fitting_time)

        return util_model


    def run_pref_learning(self, method: str, pe_strategy: str, micro_iter: int):
        r"""
        - Run preference learning for given method and pe_strategy 
        for self.every_n_comps times.
        - Save the new preference data in self.pref_data_dict.
        """

        acqf_vals = []
        for i in range(self.every_n_comps):

            train_X = self.pref_data_dict[(method, pe_strategy)].get("X", None)
            train_Y = self.pref_data_dict[(method, pe_strategy)]["Y"]
            train_comps = self.pref_data_dict[(method, pe_strategy)]["comps"]
            train_util_vals = self.pref_data_dict[(method, pe_strategy)]["util_vals"]

            logger.info(
                f"=== Running meta_iter {self.progress[method]['meta_iter']} round {i+micro_iter*self.every_n_comps} pref learning using [{pe_strategy}] ===")

            fit_model_succeed = False

            fit_on_X = True if method=="pbo" else False

            for _ in range(3):
                try:
                    util_model = self.fit_util_model(
                        method, pe_strategy, save_model=False,
                        fit_on_X=fit_on_X,
                        save_model_fit_time=True
                    )
                    fit_model_succeed = True
                    break
                except (ValueError, RuntimeError):
                    continue
            if not fit_model_succeed:
                logger.info(
                    "    -- fit_util_model() failed 3 times, stop current call of run_pref_learn()"
                )
            
            gen_pe_cand_start_time = time.time()

            if pe_strategy == "EUBO-zeta":
                with botorch.settings.debug(state=True):
                    # EUBO-zeta
                    one_sample_outcome_model = ModifiedFixedSingleSampleModel(
                        model=self.outcome_models_dict[(method, pe_strategy)],
                        outcome_dim=self.outcome_dim
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
                                raw_samples=self.raw_samples,
                                options={"batch_limit": 4, "seed": self.trial_idx},
                            )
                            cand_Y = one_sample_outcome_model(cand_X) 
                            acqf_vals.append(acqf_val.item())

                            found_valid_candidate = True
                            logger.debug(f"        -- EUBO mean, sd, min_val, max_val, quantile_vals: {acqf_landscape}")
                            logger.debug(f"        -- EUBO candidate acqf value: {acqf_val.item()}")
                            break
                        except (ValueError, RuntimeError) as error:
                            logger.info(f"        -- error in optimizing EUBO: {error}")
                            continue
                    if not found_valid_candidate:
                        logger.info(
                            f"        -- optimize_acqf() failed 3 times for EUBO with {method},", 
                            "stop current call of run_pref_learn()"
                        )
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
            
            elif method=="pbo" and pe_strategy=="TS":
                # TODO: check correctness!!!
                # use Thompson sampling to draw comparisons
                if train_X is None:
                    raise RuntimeError("train_X in pref_data_dict must not be None for Thompson Sampling")
                # get the sample with the largest posterior value
                comp1 = util_model.posterior(train_X).sample().argmax(dim=-2)
                # exclude the first sample, take the sample with the second largest posterior value
                sample2 = util_model.posterior(train_X).sample()
                sample2[:, comp1.squeeze(), :] = -float("Inf")
                comp2 = sample2.argmax(dim=-2)
                # Create candidate comparisons
                cand_comps = torch.cat((comp1, comp2), dim=-1)
                # reorder so that the larger-util one comes first
                if train_util_vals[comp2] > train_util_vals[comp1]:
                    cand_comps = cand_comps.flip(-1) # TODO: double check
                train_comps = torch.cat((train_comps, cand_comps))
                self.pref_data_dict[(method, pe_strategy)]["comps"] = train_comps
                return
            else:
                raise RuntimeError("Not supported: method={}, pe_strategy={}".format(method, pe_strategy))
        
            gen_pe_cand_time = time.time() - gen_pe_cand_start_time
            self.time_consumption[(method, pe_strategy)]["gen_pe_cand_time"].append(gen_pe_cand_time)
            logger.info(f"        -- gen_pe_cand_time: {gen_pe_cand_time:.2f} seconds")

            cand_Y = cand_Y.detach().clone()
            cand_util_val = self.util_func(cand_Y)
            cand_comps = gen_comps(cand_util_val)
            logger.info(f"        -- EUBO selected candidate util val: {cand_util_val.squeeze(-1).tolist()}")
            
            train_comps = torch.cat(
                (train_comps, cand_comps + train_Y.shape[0])
            )
            train_Y = torch.cat((train_Y, cand_Y))
            train_util_vals = torch.cat((train_util_vals, cand_util_val))
            logger.debug(f'        -- train_Y, train_comps shape: {train_Y.shape}, {train_comps.shape}')

            if method == "pca_eubo_rt":
                self.subspace_training_Y[(method, pe_strategy)] = torch.cat(
                    (self.subspace_training_Y[(method, pe_strategy)],
                     cand_Y[cand_comps[0][0].item()].unsqueeze(0))
                )

            self.pref_data_dict[(method, pe_strategy)] = {
                "Y": train_Y,
                "util_vals": train_util_vals,
                "comps": train_comps
            }
            

    def find_max_posterior_mean(
        self, 
        method: str, 
        pe_strategy: str, 
        num_pref_samples: int =1
    ):
        r"""
        Find the candidate design that maximizes the posterior mean utility.
        Compute its true utility as well as other diagnostics, return in a 
        dictionary called `within_result`.
        """

        logger.info(f"=== Finding posterior util maximizer using [{method}] with [{pe_strategy}] ===")

        n_comps = self.pref_data_dict[(method, pe_strategy)]["comps"].shape[0]

        within_result = {}

        # NOTE: I thought about whether we could save computation by avoiding double-fitting
        # the same pref model here and in the next call of run_pref_learning()
        # But this function is called every three times run_pref_learning() is called
        # so the amount of duplicated effort is not large 
        if method == "pbo":
            util_model = self.fit_util_model(
                method, pe_strategy, save_model=True, save_model_fit_time=False,
                fit_on_X=True
            )
        else:
            util_model = self.fit_util_model(
                method, pe_strategy, save_model=True, save_model_fit_time=False
            )
        sampler = SobolQMCNormalSampler(num_pref_samples)
        if method=="pbo":
            model = self.util_models_dict[(method, pe_strategy)]
        else:
            model = self.outcome_models_dict[(method, pe_strategy)]
        pref_obj = None if method=="pbo" else LearnedObjective(pref_model=util_model, sampler=sampler)

        # find experimental candidate(s) that maximize the posterior mean utility
        post_mean_cand_X, _ = gen_exp_cand(
            model=model,
            objective=pref_obj,
            problem=self.problem,
            q=1,
            acqf_name="posterior_mean",
            seed=self.trial_idx,
            sampler_num_outcome_samples=self.sampler_num_outcome_samples,
        )
        post_mean_util = self.util_func(
            self.problem.evaluate_true(post_mean_cand_X)).item()
        logger.info(
            f"        -- True utility of posterior mean utility maximizer: {post_mean_util:.3f}")

        if method == "pca_postmax_rt":
            outcome_to_add = self.outcome_models_dict[(method, pe_strategy)].posterior(post_mean_cand_X).mean
            self.subspace_training_Y[(method, pe_strategy)] = torch.cat(
                (self.subspace_training_Y[(method, pe_strategy)],
                outcome_to_add)
            )

        # look at util_model(outcome_model(x)) for a large number of sampled x
        # util_posterior_landscape = get_function_statistics(
        #     function=util_model, 
        #     # bounds=self.problem.outcome_bounds,
        #     bounds=self.problem.bounds,
        #     inner_function=self.outcome_models_dict[(method, pe_strategy)]
        # )
        # logger.debug(
        #     "        -- util posterior mean function mean, sd, min_val, max_val, quantile_vals: "
        #     f"{util_posterior_landscape}"
        # )

        # self.util_postmean_landscape[(method, pe_strategy)].append(util_posterior_landscape)

        within_result = {
            "n_comps": n_comps,
            "util": post_mean_util,
            "run_id": self.trial_idx,
            "pe_strategy": pe_strategy,
            "method": method,
            "candidate": post_mean_cand_X.tolist()
        }

        # check util model fit here
        # util_model_acc = check_util_model_fit(
        #     util_model=util_model, 
        #     problem=self.problem, util_func=self.util_func, 
        #     n_test=1024, batch_eval=True)
        # within_result["util_model_acc"] = util_model_acc
        # util_model_acc_top_half = check_util_model_fit(
        #     util_model=util_model, 
        #     problem=self.problem, util_func=self.util_func, 
        #     n_test=1024, batch_eval=True, top_quantile=0.5)
        # within_result["util_model_acc_top_half"] = util_model_acc_top_half
        # util_model_acc_top_quarter = check_util_model_fit(
        #     util_model=util_model, 
        #     problem=self.problem, util_func=self.util_func, 
        #     n_test=1024, batch_eval=True, top_quantile=0.25)
        # within_result["util_model_acc_top_quarter"] = util_model_acc_top_quarter

        # overall_model_acc = check_overall_fit(
        #     outcome_model=self.outcome_models_dict[(method, pe_strategy)],
        #     util_model=util_model,
        #     problem=self.problem,
        #     util_func=self.util_func,
        #     n_test=1000,
        #     batch_eval=True
        # )
        # within_result["overall_model_acc"] = overall_model_acc

        return within_result


    def generate_BO_candidate( 
        self, 
        method: str, 
        pe_strategy: str, 
        BO_iter: int, 
        best_so_far: float,
        cum_n_BO_iters_so_far: int
    ):
        r"""
        Generate `self.BO_batch_size` number of BO candidate designs.
        Args:
            BO_iter: iteration index in the current BO stage
            best_so_far: best observed utility value in all BO stages so far
            cum_n_BO_iters_so_far: cumulative number of BO iterations in all BO 
                stages so far (not including the current BO stage)
        """

        if method == "random_search":
            rs_X = torch.rand([self.BO_batch_size, self.input_dim], dtype = torch.double) 
            rs_X = rs_X * (self.problem.bounds[1] - self.problem.bounds[0]) + self.problem.bounds[0]
            rs_util = self.util_func(self.problem.evaluate_true(rs_X)) # shape (q, 1)
            if best_so_far is None:
                best_so_far = rs_util.max().item()
            else: 
                best_so_far = max(best_so_far, rs_util.max().item())
            exp_result = {
                "candidate": rs_X.tolist(),
                "candidate_util": rs_util.squeeze(-1).tolist(),
                "best_util_so_far": best_so_far,
                "BO_iter": BO_iter + cum_n_BO_iters_so_far + 1,
                "method": method,
                "strategy": pe_strategy,
                "run_id": self.trial_idx,
            }

        else:
            # read model from self.util_models_dict
            # for non-retraining methods, it's the one saved during find_max_posterior_mean
            # for retraining methods, it's the one saved after the previous BO iteration
            util_model = self.util_models_dict[(method, pe_strategy)]
            # util_model_acc = check_util_model_fit(
            #     util_model, self.problem, self.util_func, n_test=1024, batch_eval=True)
            util_model_acc = None

            sampler = SobolQMCNormalSampler(4)
            pref_obj = LearnedObjective(pref_model=util_model, sampler=sampler)

            baseline_X = self.BO_data_dict[(method, pe_strategy)]["X"]

            gen_bo_cand_start_time = time.time()

            # find experimental candidate(s) that maximize the noisy EI acqf
            # TODO: find a way to adapt this to PBO-TS
            if method == "pbo":
                new_cand_X, acqf_val = gen_exp_cand(
                    model=self.util_models_dict[(method, pe_strategy)], # TODO: is this right? 
                    objective=None,
                    problem=self.problem,
                    q=self.BO_batch_size, 
                    acqf_name="qNEI",
                    X=baseline_X, 
                    seed=self.trial_idx
                )
            else:
                new_cand_X, acqf_val = gen_exp_cand(
                    model=self.outcome_models_dict[(method, pe_strategy)],
                    objective=pref_obj,
                    problem=self.problem,
                    q=self.BO_batch_size, 
                    acqf_name="qNEI",
                    X=baseline_X, 
                    seed=self.trial_idx
                )
            # new_cand_X_posterior = self.outcome_models_dict[(method, pe_strategy)].posterior(new_cand_X)
            # new_cand_X_posterior_mean_util = util_model.posterior(new_cand_X_posterior.mean).mean.detach()
            # new_cand_X_posterior_var = new_cand_X_posterior.variance
            # logger.info(f"        -- new_cand_X_posterior_mean_util: {new_cand_X_posterior_mean_util}")
            # print('average new_cand_X_posterior_var: ', torch.mean(new_cand_X_posterior_var))
            new_cand_Y = self.problem(new_cand_X).detach()

            gen_bo_cand_time = time.time() - gen_bo_cand_start_time
            self.time_consumption[(method, pe_strategy)]["gen_bo_cand_time"].append(gen_bo_cand_time)
            logger.info(f"    -- time consumption for generating BO candidates: {gen_bo_cand_time:.2f} seconds")

            qneiuu_util = self.util_func(new_cand_Y).detach() # shape (q, 1) or q
            logger.info(
                f"        -- ({method}, {pe_strategy})-qNEIUU candidate utility: {qneiuu_util.squeeze(-1).tolist()}"
            )

            if best_so_far is None:
                best_so_far = torch.max(qneiuu_util).item()
            else:
                best_so_far = max(best_so_far, torch.max(qneiuu_util).item())

            logger.info(f"    ** best so far at BO iter {BO_iter}: {best_so_far}")

            exp_result = {
                "candidate": new_cand_X.tolist(),
                "acqf_val": acqf_val.tolist(),
                # "candidate_posterior_mean_util": new_cand_X_posterior_mean_util.squeeze(-1).tolist(),
                # "candidate_posterior_variance": new_cand_X_posterior.variance,
                "candidate_util": qneiuu_util.squeeze(-1).tolist(),
                "best_util_so_far": best_so_far,
                "BO_iter": BO_iter + cum_n_BO_iters_so_far + 1,
                "method": method,
                "strategy": pe_strategy,
                "run_id": self.trial_idx,
                "util_model_acc": util_model_acc,
            }
            
            # log the true optimal utility computed in __init__()
            exp_result["true_opt"] = self.true_opt

            self.BO_data_dict[(method, pe_strategy)]["X"] = torch.cat(
                (self.BO_data_dict[(method, pe_strategy)]["X"], new_cand_X),
                dim=0
            )
            self.BO_data_dict[(method, pe_strategy)]["Y"] = torch.cat(
                (self.BO_data_dict[(method, pe_strategy)]["Y"], new_cand_Y),
                dim=0
            )

        self.final_candidate_results[method][pe_strategy].append(exp_result)

        return best_so_far


    def compute_subspace_diagnostics(self, method: str, pe_strategy: str, n_test = 1000):
        r"""Compute and save diagnostics for the method and pe_strategy:
        - outcome reconstruction in subspace
        - utility reconstruction in subspace
        - best achievable utility value in subspace max_x g(VV^T f(x))
        - average utility value in subspace E[g(VV^T f(x))]
        """
        projection = self.projections_dict[(method, pe_strategy)][-1] 

        max_outcome_error = mc_max_outcome_error(
            problem=self.problem,
            projection=projection,
            n_test=n_test
        )
        
        max_util_error = mc_max_util_error(
            problem=self.problem,
            projection=projection,
            util_func=self.util_func,
            n_test=n_test
        )

        max_subspace_util, avg_subspace_util = best_and_avg_util_in_subspace(
            problem=self.problem, 
            projection=projection, 
            util_func=self.util_func
        )

        self.subspace_diagnostics[(method, pe_strategy)]["max_util_error"].append(max_util_error)
        self.subspace_diagnostics[(method, pe_strategy)]["max_outcome_error"].append(max_outcome_error)
        self.subspace_diagnostics[(method, pe_strategy)]["best_util"].append(max_subspace_util)
        self.subspace_diagnostics[(method, pe_strategy)]["avg_util"].append(avg_subspace_util)


# ======== Putting together into stages ========

    def run_initial_experimentation_stage(self, method: str):
        r"""Run initial experimentation stage for the method.
        Compute projection (if needed), fit outcome model, save data in 
        self.pref_data_dict.
        """

        if method == "random_search":
        # if method in ("random_search", "pbo"):
            return

        for pe_strategy in self.pe_strategies[method]:
            self.compute_projections(method, pe_strategy)
            if method == "pbo":
                return
            else:
                self.fit_outcome_model(method, pe_strategy)        


    def run_PE_stage(self, method: str, meta_iter: int):
        r"""Run preference exploration stage. """

        if method == "random_search": 
            return

        for pe_strategy in self.pe_strategies[method]:

            if any(method.startswith(header) for header in self.subspace_methods_headers):
                self.compute_subspace_diagnostics(method, pe_strategy, n_test=1000)

            pe_start_time = time.time()

            logger.info(f"===== Running PE meta_iter {meta_iter} using [{method}] with [{pe_strategy}] =====")

            # compute max utility from initial data if meta_iter == 0
            if meta_iter == 0:
                self.PE_session_results[method][pe_strategy].append(
                    self.find_max_posterior_mean(method, pe_strategy)
                )

            for j in range(self.n_check_post_mean):

                self.run_pref_learning(method, pe_strategy, j)
                self.PE_session_results[method][pe_strategy].append(
                    self.find_max_posterior_mean(method, pe_strategy)
                )
                # relearn subspace if method calls for retraining
                # in fact, only the methods that require utility model posterior 
                # to fit the subspace needs this step
                if method.endswith("rt"):
                    logger.info(f"    ~~ Retraining subspace using [{method}] with [{pe_strategy}] during PE")
                    prev_projection = self.projections_dict[(method, pe_strategy)][-1]
                    self.compute_projections(method, pe_strategy)
                    _, _, g = compute_grassmannian(
                        prev_projection, 
                        self.projections_dict[(method, pe_strategy)][-1]
                    )
                    self.subspace_diagnostics[(method, pe_strategy)]["grassmannian"].append(g)
                    if method != "pca_norefit_rt":
                        self.fit_outcome_model(method, pe_strategy)
                    # run diagnostics on the updated subspaces
                    self.compute_subspace_diagnostics(method, pe_strategy, n_test=1024)
            
            # log time required to do PE 
            PE_time = time.time() - pe_start_time
            self.time_consumption[(method, pe_strategy)]["PE_time"].append(PE_time)
            logger.info(f"    -- time consumption for PE stage: {PE_time:.2f} seconds")


    def run_BO_experimentation_stage(self, method: str, meta_iter: int): 
        r"""Run experimentation stage where candidates are generated using BO. """
        for pe_strategy in self.pe_strategies[method]:
            logger.info(f"===== Running BO meta_iter {meta_iter} using [{method}] with [{pe_strategy}] =====")

            if len(self.final_candidate_results[method][pe_strategy]) > 0:
                best_so_far = \
                    self.final_candidate_results[method][pe_strategy][-1]["best_util_so_far"]
                logger.info(f"    -- final_candidate_results[{method}][{pe_strategy}] already exists, taking the last best_so_far = {best_so_far:.4f}")
            else:
                # equivalently, if meta_iter == 0
                # start with best value in initial experimentation batch, and save it
                best_so_far = max(
                    self.util_func(self.BO_data_dict[(method, pe_strategy)]["Y"])
                ).item()
                init_exp_result = {
                    "best_util_so_far": best_so_far,
                    "BO_iter": 0,
                    "method": method,
                    "strategy": pe_strategy,
                    "run_id": self.trial_idx,
                }
                self.final_candidate_results[method][pe_strategy].append(init_exp_result)
                logger.info(f"    -- final_candidate_results[{method}][{pe_strategy}] does not exist, initializing with best util in initial batch best_so_far = {best_so_far:.4f}")
                 
            logger.info(f"    ** best so far before BO exp stage: {best_so_far}") 

            bo_start_time = time.time()

            for iter in range(self.n_BO_iters): 
                logger.info(f"== Running meta_iter {meta_iter} BO iteration {iter} using [{method}] with [{pe_strategy}] ==")
                # total number of BO iterations before the current meta-stage
                cum_n_BO_iters_so_far = meta_iter * (self.n_BO_iters) 
                best_so_far = self.generate_BO_candidate(
                    method, pe_strategy, iter, best_so_far, cum_n_BO_iters_so_far)

                # update subspace and refit util model for retraining methods
                if method.endswith("rt"):
                    logger.info(f"    ~~ Retraining subspace using [{method}] with [{pe_strategy}] during BO")
                    prev_projection = self.projections_dict[(method, pe_strategy)][-1]
                    self.compute_projections(method, pe_strategy)
                    _, _, g = compute_grassmannian(
                        prev_projection,
                        self.projections_dict[(method, pe_strategy)][-1]
                    )
                    self.subspace_diagnostics[(method, pe_strategy)]["grassmannian"].append(g)
                    if method != "pca_norefit_rt": # TODO: check
                        self.fit_util_model(method, pe_strategy, save_model=True)
                    self.compute_subspace_diagnostics(method, pe_strategy, n_test=1024)
                
                # refit outcome model for all methods, whether retraining or not
                self.fit_outcome_model(method, pe_strategy)
            
            bo_time = time.time() - bo_start_time
            self.time_consumption[(method, pe_strategy)]["BO_time"].append(bo_time)
            logger.info(f"    -- time consumption for BO stage: {bo_time:.2f} seconds")



# ======== Further putting together into the BOPE loop ========


    def run_BOPE_loop(self):
        r"""
        Run the full BOPE pipeline for all specified (method, pe_strategy) pairs.
        - Generate random initial experimentation data
        - Run initial experimentation stage
        - Alternate running PE stage and BO stage for self.n_meta_iters times
        """

        if all(self.progress[method]["meta_iter"] == -1 for method in self.methods):
            # all methods use the same initial experimentation data
            logger.info("============= Generating initial experimentation data for all methods =============")
            self.generate_random_experiment_data(
                self.initial_experimentation_batch,
                compute_util=True
            )
            for method in self.methods:
                self.progress[method]["meta_iter"] = 0
                self.save_experiment_data()

        for method in self.methods:
            try:
                logger.info(f"============= Running {method} =============")

                if self.progress[method]["stage"] == "initial":
                    self.run_initial_experimentation_stage(method)
                    self.progress[method]["stage"] = "PE"
                    self.save_experiment_data()

                while self.progress[method]["meta_iter"] < self.n_meta_iters:

                    logger.info(f"========== Running PE-BO meta-iter {self.progress[method]['meta_iter']} ==========")
                    if self.progress[method]["stage"] == "PE": 
                        self.run_PE_stage(method, self.progress[method]["meta_iter"])
                        self.progress[method]["stage"] = "BO"
                        self.save_experiment_data()
                    if self.progress[method]["stage"] == "BO":
                        self.run_BO_experimentation_stage(method, self.progress[method]["meta_iter"])
                        self.progress[method]["stage"] = "PE"
                        self.progress[method]["meta_iter"] += 1
                        self.save_experiment_data()

                print('\n\n')

            except Exception as e:
                logger.info(f"Error occurred: {e}")
                traceback.print_exc()
                logger.info(f"============= {method} failed, skipping =============")
                
                continue


    def save_experiment_data(self):

        if (self.output_path is not None) and self.save_results:

            results_dict = {
                "PE_session_results": self.PE_session_results,
                "final_candidate_results": self.final_candidate_results,
                "pref_data": self.pref_data_dict,
                "subspace_diagnostics": self.subspace_diagnostics,
                "BO_data": self.BO_data_dict,
                "projections": self.projections_dict,
                "transforms_covar_dict": self.transforms_covar_dict,
                "time_consumption": self.time_consumption,
                "progress": self.progress,
                "random_states": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                },
            }

            file_dir = self.output_path + f"{self.methods}/"

            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            torch.save(
                results_dict,
                file_dir + f'results_trial={self.trial_idx}.th'
            )

            logger.info(
                f"At progress {self.progress}, " 
                f"saved results to {file_dir}results_trial={self.trial_idx}.th")


    def load_experiment_data(self):
        # read from file path
        file_path = self.output_path + f"{self.methods}/" + f'results_trial={self.trial_idx}.th'
        if not os.path.exists(file_path):
            return
        
        logger.info(f"Loading existing results from {file_path}...")
        res = torch.load(file_path)

        # load saved data
        self.PE_session_results = res["PE_session_results"]
        self.final_candidate_results = res["final_candidate_results"]
        self.pref_data_dict = res["pref_data"]
        self.subspace_diagnostics = res["subspace_diagnostics"]
        self.BO_data_dict = res["BO_data"]
        self.projections_dict = res["projections"]
        self.transforms_covar_dict = res["transforms_covar_dict"] 
        self.time_consumption = res["time_consumption"]
        self.progress = res["progress"]

        # reset the random seed
        torch.set_rng_state(res["random_states"]["torch"])
        np.random.set_state(res["random_states"]["numpy"])
        random.setstate(res["random_states"]["random"])

        logger.info(f"Current progress: {self.progress}")

        # fit the models
        for method in self.methods:
            if self.progress[method]["stage"] != "initial":
                for pe_strategy in self.pe_strategies[method]:
                    self.fit_util_model(
                        method, pe_strategy, 
                        save_model=True, save_model_fit_time=False
                    )
                    self.fit_outcome_model(
                        method, pe_strategy, 
                        save_model_fit_time=False, save_diagnostics=False
                    )
            
        logger.info("Refit outcome and utility models from loaded results.")