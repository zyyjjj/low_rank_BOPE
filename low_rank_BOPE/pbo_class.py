
import os
from collections import defaultdict
from typing import List

import botorch
import gpytorch
import numpy as np
import torch
from botorch.acquisition.monte_carlo import (qNoisyExpectedImprovement,
                                             qSimpleRegret)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import (PairwiseGP,
                                        PairwiseLaplaceMarginalLogLikelihood)
from botorch.models.transforms.input import (ChainedInputTransform,
                                             FilterFeatures, Normalize)
from botorch.models.transforms.outcome import (ChainedOutcomeTransform,
                                               Standardize)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from sklearn.linear_model import LinearRegression
from torch import Tensor

from low_rank_BOPE.src.diagnostics import check_util_model_fit
from low_rank_BOPE.src.models import MultitaskGPModel, make_modified_kernel
from low_rank_BOPE.src.pref_learning_helpers import (  # find_max_posterior_mean, # TODO: later see if we want the error-handled version;;; fit_pref_model, # TODO: later see if we want the error-handled version
    ModifiedFixedSingleSampleModel, find_true_optimal_utility,
    fit_outcome_model, gen_comps, gen_exp_cand)
from low_rank_BOPE.src.transforms import (InputCenter,
                                          LinearProjectionInputTransform,
                                          LinearProjectionOutcomeTransform,
                                          PCAInputTransform,
                                          PCAOutcomeTransform,
                                          SubsetOutcomeTransform,
                                          compute_weights, fit_pca,
                                          generate_random_projection,
                                          get_latent_ineq_constraints)


class PboExperiment:

    # add weighted PCA, add explicit computation in latent space

    attr_list = {
        "pca_var_threshold": 0.95,
        "initial_pref_batch_size": 16,
        "n_check_post_mean": 20,
        "every_n_comps": 3,
        "verbose": True,
        "dtype": torch.double,
        "num_restarts": 20,
        "raw_samples": 128,
        "batch_limit": 4,
        "sampler_num_outcome_samples": 64,
        "maxiter": 1000,
        "latent_dim": None,
        "min_stdv": 100000,
        "true_axes": None, # specify these for synthetic problems
    }

    def __init__(
        self,
        problem: torch.nn.Module,
        util_func: torch.nn.Module,
        methods: List[str],
        trial_idx: int,
        output_path: str,
        **kwargs
    ) -> None:
        
        # self.attr_list stores default values, then overwrite with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        print("PboExperiment settings: ", self.__dict__)

        # pre-specified experiment metadata
        self.problem = problem.double()
        self.util_func = util_func
        self.outcome_dim = problem.outcome_dim
        self.input_dim = problem._bounds.shape[-1]
        # allow problem to have optional bounds on outcome vectors
        self.outcome_bounds = problem.outcome_bounds.to(torch.double) if hasattr(problem, "outcome_bounds") else None
        self.trial_idx = trial_idx
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if hasattr(self.problem, "true_axes"):
            self.true_axes = self.problem.true_axes

        if "uw_pca" in methods:
            # run unweighted pca first, so that the learned latent_dim
            # informs the other methods with dim reduction
            self.methods = ["uw_pca"] + [m for m in methods if m != "uw_pca"]
            print('self.methods, ', self.methods)
        else:
            # if "unweighted_pca" is not run and latent_dim is not specified
            # set self.latent_dim by hand
            self.methods = methods
            if not self.latent_dim:
                self.latent_dim = self.outcome_dim // 3
 
        # log model specifics
        self.util_models_dict = {} 
        self.pref_data_dict = defaultdict(dict)  # (Y, util_vals, comps), also latent values for dim-reduction methods
        self.projections_dict = {} # projections to subspaces
        self.acqf_bounds_dict = defaultdict(dict) # (bounds, inequality_constraints)

        # log results
        self.PE_time_dict = {}
        self.PE_session_results = defaultdict(dict)
        self.final_candidate_results = defaultdict(dict)


    def generate_initial_data(self, n):
        r"""Generate n observations of outcomes and n/2 pairwise comparisons.
        This is shared by all methods initially. 
        Args:
            n: number of samples
        Computes and stores into data dict:
            Y: `n x outcome_dim` tensor of noisy evaluated outcomes
            util_vals: `n x 1` tensor of utility values of Y
            comps: `n/2 x 2` tensor of pairwise comparisons 
        """

        X = (
            draw_sobol_samples(bounds=self.problem.bounds, n=1, q=n, seed=self.trial_idx)
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        Y = self.problem(X).detach()

        util_vals = self.util_func(Y).detach()
        comps = gen_comps(util_vals)

        for method in self.methods:
            self.pref_data_dict[method] = {
                "Y": Y,
                "util_vals": util_vals,
                "comps": comps
            }

    def compute_projection_and_acqf_bounds(self, method):

        Y = self.pref_data_dict[method]["Y"]
        util_vals = self.pref_data_dict[method]["util_vals"]
        
        if method == "w_pca_true":
            weights = compute_weights(util_vals.squeeze(1))
            projection = fit_pca(
                Y, 
                var_threshold=self.pca_var_threshold,
                weights=weights
            ) 

        # TODO: currently not running well
        elif method == "w_pca_est":
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
        
        elif method == "uw_pca":
            projection = fit_pca(
                Y, 
                var_threshold=self.pca_var_threshold, 
                weights=None
            ) 
        
        elif method == "st":
            projection = None

        elif method in ("random_linear_proj"):
            pass

        if projection is not None:
            # compute and save latent variables, save projection
            L = torch.matmul(
                Y, 
                torch.transpose(projection, -2, -1)
            )
            self.pref_data_dict[method]["L"] = L
            self.projections_dict[method] = projection

            # compute and save acquisition function bounds and ineq constraints
            self.acqf_bounds_dict[method]["bounds"] = \
                torch.tensor([[-10000]*projection.shape[0], [10000]*projection.shape[0]], dtype=torch.double)
            
            if self.outcome_bounds is not None:
                latent_ineq_constraints = get_latent_ineq_constraints(
                    projection=projection,
                    original_bounds=self.outcome_bounds
                )
                self.acqf_bounds_dict[method]["ineq_constraints"] = latent_ineq_constraints
            else:
                self.acqf_bounds_dict[method]["ineq_constraints"] = None
        
        else:
            # no dimensionality reduction
            # compute and save acquisition function bounds and ineq constraints
            self.acqf_bounds_dict[method]["ineq_constraints"] = None
            if self.outcome_bounds is not None:
                self.acqf_bounds_dict[method]["bounds"] = self.outcome_bounds
            else:
                self.acqf_bounds_dict[method]["bounds"] = \
                    torch.tensor([[-10000]*self.outcome_dim, [10000]*projection.self.outcome_dim], dtype=torch.double)
        

    def fit_util_model(self, method, **model_kwargs):
        r"""
        Fit utility model from outcomes / latent variables to utility values.
        """

        if "L" in self.pref_data_dict[method]:
            train_YorL = self.pref_data_dict[method]["L"]
        else:
            train_YorL = self.pref_data_dict[method]["Y"]
        train_comps = self.pref_data_dict[method]["comps"]
        
        util_model = PairwiseGP(
            datapoints=train_YorL, 
            comparisons=train_comps, 
            **model_kwargs
        )

        mll_util = PairwiseLaplaceMarginalLogLikelihood(
            util_model.likelihood, util_model)
        fit_gpytorch_mll(mll_util)

        self.util_models_dict[method] = util_model

        
    def run_pref_learning(self, method):

        latent = True if method in self.projections_dict else False
        print(method, latent)

        acqf_vals = []
        for i in range(self.every_n_comps):

            print(
                f"Running {i+1}/{self.every_n_comps} preference learning")

            fit_model_succeed = False
            pref_model_acc = None

            for _ in range(3):
                try:
                    self.compute_projection_and_acqf_bounds(method)
                    self.fit_util_model(method)
                    print("Pref model fitting successful")
                    fit_model_succeed = True
                    break
                except (ValueError, RuntimeError):
                    continue
            if not fit_model_succeed:
                print(
                    "fit_pref_model() failed 3 times, stop current call of run_pref_learn()"
                )

            if method == "random":
                pass
                # TODO: randomly sample cand_Y from utility model input domain

            else:
                # use statistical method + EUBO-zeta strategy

                acqf = AnalyticExpectedUtilityOfBestOption(
                    pref_model=self.util_models_dict[method],
                ).to(torch.double) 

                found_valid_candidate = False
                for _ in range(3):
                    try:
                        cand, acqf_val = optimize_acqf(
                            acq_function=acqf,
                            q=2,
                            bounds=self.acqf_bounds_dict[method]["bounds"],
                            inequality_constraints=self.acqf_bounds_dict[method]["ineq_constraints"],
                            num_restarts=self.num_restarts,
                            raw_samples=self.raw_samples,
                            options={"batch_limit": 4, "seed": self.trial_idx},
                        )
                        acqf_vals.append(acqf_val.item())

                        found_valid_candidate = True
                        break
                    except (ValueError, RuntimeError) as error:
                        print("error in optimizing EUBO: ", error)
                        continue
                if not found_valid_candidate:
                    print(
                        f"optimize_acqf() failed 3 times for EUBO with {method},", 
                        "stop current call of run_pref_learn()"
                    )
                    return

            if latent:
                cand_Y = torch.matmul(cand.to(torch.double), self.projections_dict[method])
            else:
                cand_Y = cand.to(torch.double).detach().clone()
            cand_util_val = self.util_func(cand_Y)
            cand_comps = gen_comps(cand_util_val)

            self.pref_data_dict[method]["comps"] = torch.cat(
                (self.pref_data_dict[method]["comps"],
                cand_comps + self.pref_data_dict[method]["Y"].shape[0])
            )
            self.pref_data_dict[method]["Y"] = torch.cat(
                (self.pref_data_dict[method]["Y"], cand_Y)
            )
            self.pref_data_dict[method]["util_vals"] = torch.cat(
                (self.pref_data_dict[method]["util_vals"], cand_util_val)
            )
            if latent:
                self.pref_data_dict[method]["L"] = torch.cat(
                    (self.pref_data_dict[method]["L"], cand)
                )
            

    def find_max_posterior_mean(self, method):
        # compute posterior-util-maximizing candidate images

        if method in self.projections_dict:
            latent = True
            projection = self.projections_dict[method]
        else:
            latent = False
            projection = None

        sampler = SobolQMCNormalSampler(64)
        postmean_acqf = qSimpleRegret(
            model = self.util_models_dict[method], 
            sampler = sampler,
        ).to(torch.double)

        cand, _ = optimize_acqf(
            acq_function=postmean_acqf,
            q=1,
            bounds=self.acqf_bounds_dict[method]["bounds"],
            inequality_constraints=self.acqf_bounds_dict[method]["ineq_constraints"],
            num_restarts=8,
            raw_samples=64,
            options={"batch_limit": 4, "seed": 0},
            sequential=True,
        )

        if latent:
            cand_Y = torch.matmul(cand.to(torch.double), projection)
        else:
            cand_Y = cand.to(torch.double).detach().clone()
        cand_util_val = self.util_func(cand_Y).item()

        within_result = {
            "n_comps": self.pref_data_dict[method]["comps"].shape[0],
            "util": cand_util_val,
            "run_id": self.trial_idx,
            "method": method,
            "candidate": cand_Y
        }

        # check util model fit
        util_model_acc = check_util_model_fit(
            self.util_models_dict[method], self.problem, self.util_func, 
            n_test=1000, batch_eval=True, projection=projection)
        within_result["util_model_acc"] = util_model_acc

        return within_result


    def run_PE_stage(self, method):

        print(f"===== Running PE using {method} =====")

        self.PE_session_results[method] = []

        self.compute_projection_and_acqf_bounds(method)
        self.fit_util_model(method)
        self.PE_session_results[method].append(
            self.find_max_posterior_mean(method)
        )

        for j in range(self.n_check_post_mean):
            self.run_pref_learning(method)
            self.PE_session_results[method].append(
                self.find_max_posterior_mean(method)
            )            

    def run_PBO_loop(self):
        self.generate_initial_data(n=self.initial_pref_batch_size)
        for method in self.methods:
            try:
                # run
                # self.generate_initial_data()
                self.run_PE_stage(method)

                # save, something like
                torch.save(self.PE_session_results, self.output_path +
                        'PE_session_results_trial=' + str(self.trial_idx) + '.th')
                # torch.save(self.final_candidate_results, self.output_path +
                #         'final_candidate_results_trial=' + str(self.trial_idx) + '.th')
                # torch.save(self.outcome_model_fitting_results, self.output_path +
                #         'outcome_model_fitting_results_trial=' + str(self.trial_idx) + '.th')
                
            except:
                print(f"============= {method} failed, skipping =============")
                continue