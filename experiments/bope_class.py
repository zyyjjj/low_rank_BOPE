#!/usr/bin/env python3
import torch

from botorch.utils.sampling import draw_sobol_samples


# TODO: nested dictionary
methods_settings = {
    "st": None,
    "pca": None,
    "random_linear_proj": None,
    "random_subset": None,
    "lmc1": None,
    "lmc2": None
}
# one run should handle one problem and >=1 methods




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
        "use_PCR": False
        # maybe num_axes, default to None?
        "min_stdv": 100000
    }


    def __init__(
        self,
        problem,
        util_func,
        methods,
        PE_strategy,
        trial_idx,
        **kwargs
    ) -> None:
        """
        specify experiment settings
            problem:
            util_func:
            methods: list of statistical methods, each denoted using a str
            PE_strategies: PE strategies to use, could be {"EUBO-zeta", "Random-f"}
        """


        # pre-specified experiment metadata
        self.problem = problem
        self.util_func = util_func
        # TODO: implement this in problem class
        self.outcome_dim = problem.outcome_dim
        self.input_dim = problem.dim

        # self.attr_list stores default values, then overwrite with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        # results logging
        # but should have a list for each method
        # TODO: maybe make into dicts?
        self.within_session_results = []
        self.exp_candidate_results = []

        self.methods = methods
        self.pe_strategies = pe_strategies

        # TODO: for each method, have a dict of model class, transforms, other hyperparameters
        # make this dictionary a class attribute?
        # or maybe store this dict in an external yaml file for cleanness
        # then make self.methods_dict = {}

        self.outcome_models_dict = {}
        self.util_models_dict = {}
        self.pref_data_dict = {}

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



        # error handling
        self.passed = False
        self.fit_count = 0

    def run_first_experimentation_stage(self, method):

        self.fit_outcome_model(method)


    def run_PE_stage(self, method):
        # initial result stored in self.pref_data_dict
        self.generate_random_pref_data(method, n)

        for pe_strategy in self.pe_strategies:
            self.find_max_posterior_mean()
            for j in range(self.n_check_post_mean):
                self.run_pref_learning()
                self.find_max_posterior_mean()
                # TODO: then update result
        pass

    def run_second_experimentation_stage(self, method):
        for pe_strategy in self.pe_strategies:
            self.generate_experiment_candidate(method, pe_strategy)
        pass

    def generate_random_experiment_data(self, n, compute_util: False):
        r"""Generate n observations of experimental designs and outcomes.
        Args:
            problem: a TestProblem
            n: number of samples
        Computes:
            X: `n x problem input dim` tensor of sampled inputs
            Y: `n x problem outcome dim` tensor of noisy evaluated outcomes at X
            util_vals: `n x 1` tensor of utility value of Y
            comps: `n/2 x 2` tensor of pairwise comparisons # TODO: confirm
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
        # use self.X, self.Y, self.outcome_transform
        # maybe input model class and other kwargs hyperparameters
        # handle different methods differently

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
                rank=1, # rank is 2 if method is lmc2
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
            ).to(**tkwargs)
            lcm_mll = ExactMarginalLogLikelihood(
                outcome_model.likelihood, outcome_model
            )
            # fit_gpytorch_mll(lcm_mll)
            fit_gpytorch_scipy(lcm_mll, options={"maxls": 30})

        elif method == "pcr":
            P, S, V = torch.svd(self.Y)
            # then run regression from P (PCs) onto util_vals
            # select top k entries of PC_coeff
            # retain the corresponding columns in P

            reg = LinearRegression().fit(np.array(P), np.array(self.util_vals))
            dims_to_keep = np.argpartition(reg.coef_, -config["outcome_dim"] // 3)[
                -config["outcome_dim"] // 3 :    # TODO: update
            ]

            # transform corresponding to dims_to_keep
            self.pcr_axes = torch.tensor(torch.transpose(V[:, dims_to_keep], -2, -1))
            # then plug these into LinearProjection O/I transforms
            self.transforms_covar_dict["pcr"] = {
                "outcome_tf": LinearProjectionOutcomeTransform(self.pcr_axes),
                "input_tf": LinearProjectionInputTransform(self.pcr_axes),
                "covar_module": make_modified_kernel(
                    ard_num_dims=config["outcome_dim"] // 3 # TODO: update
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
                        "center": InputCenter(config["outcome_dim"]),
                        "pca": PCAInputTransform(axes=self.pca_axes),
                    }
                )

                self.lin_proj_latent_dim = self.pca_axes.shape[0]

                print(
                    f"amount of variance explained by {self.lin_proj_latent_dim} axes: {outcome_model.outcome_transform['pca'].PCA_explained_variance}"
                )

                # here we first see how many latent dimensions PCA learn
                # then we create a random linear projection mapping to the same dimensionality

                transforms_covar_dict["pca"]["covar_module"] = make_modified_kernel(
                    ard_num_dims=self.lin_proj_latent_dim
                )

                random_proj = generate_random_projection(
                    self.outcome_dim, self.lin_proj_latent_dim, **tkwargs
                )
                transforms_covar_dict["random_linear_proj"] = {
                    "outcome_tf": LinearProjectionOutcomeTransform(random_proj),
                    "input_tf": LinearProjectionInputTransform(random_proj),
                    "covar_module": make_modified_kernel(ard_num_dims=self.lin_proj_latent_dim),
                }
                random_subset = random.sample(
                    range(self.outcome_dim), self.lin_proj_latent_dim
                )
                transforms_covar_dict["random_subset"] = {
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
        # input_transform,
        # covar_module,
        # utility_aware
    ):
        util_model = PairwiseGP(datapoints=Y, comparisons=comps, **model_kwargs)

        mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)
        fit_gpytorch_model(mll_util)

        return util_model

    def run_pref_learning(self, method, pe_strategy):
        # TODO

        acqf_vals = []
        for i in range(self.every_n_comps):
            if verbose:
                print(f"Running {i+1}/{self.every_n_comps} preference learning using {pe_strategy}")

            fit_model_succeed = False
            pref_model_acc = None
            for _ in range(3):
                try:
                    pref_model = fit_pref_model(
                        train_Y,
                        train_comps,
                        input_transform=input_transform,
                        covar_module=covar_module,
                        likelihood=likelihood,
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
                return train_Y, train_comps, None, acqf_vals

            if pe_strategy == "EUBO-zeta":
                # EUBO-zeta
                one_sample_outcome_model = ModifiedFixedSingleSampleModel(
                    model=outcome_model, outcome_dim=train_Y.shape[-1]
                )
                acqf = AnalyticExpectedUtilityOfBestOption(
                    pref_model=pref_model, outcome_model=one_sample_outcome_model
                )
                found_valid_candidate = False
                for _ in range(3):
                    try:
                        cand_X, acqf_val = optimize_acqf(
                            acq_function=acqf,
                            q=2,
                            bounds=problem.bounds,
                            num_restarts=8,
                            raw_samples=64,  # used for intialization heuristic
                            options={"batch_limit": 4, "seed": seed},
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
                cand_X = generate_random_inputs(problem, n=2)
                cand_Y = outcome_model.posterior(cand_X).rsample().squeeze(0).detach()
            else:
                raise RuntimeError("Unknown preference exploration strategy!")

            cand_Y = cand_Y.detach().clone()
            cand_comps = gen_comps(util_func(cand_Y))

            train_comps = torch.cat((train_comps, cand_comps + train_Y.shape[0]))
            train_Y = torch.cat((train_Y, cand_Y))

        return train_Y, train_comps, pref_model_acc, acqf_vals



        pass

    def find_max_posterior_mean(self, method, pe_strategy, num_pref_samples = 1):
        # TODO: understand whether we should increase num_pref_samples from 1!

        self.outcome_model_dict[method]
        train_Y, train_comps = self.pref_data_dict[method][pe_strategy]
        # seed = self.trial_idx

        pref_model = fit_pref_model(
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
        )

        post_mean_util = self.util_func(self.problem.evaluate_true(post_mean_cand_X)).item()
        if verbose:
            print(f"True utility of posterior mean utility maximizer: {post_mean_util:.3f}")

        # TODO: update results
        # within_result = {
        #     "n_comps": train_comps.shape[0],
        #     "util": post_mean_util,
        # }
        # return within_result



    def generate_experiment_candidate(self):
        # TODO
        self.fit_pref_model()
        # note that this is method and pe-strategy specific
        # maybe could store data / pref model in a dict
        # like self.pref_data[method][strategy] has train_Y, train_comps, pref_model
        # then define qneiuu, and find its maximizer
        pass

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


    def run_BOPE_loop(self):
        # handle repetitions
        # have a flag for whether the fitting is successful or not

        # maybe cache this after first time
        self.generate_random_experiment_data(
            self.initial_experimentation_batch,
            compute_util = True
        )

        for method in self.methods:
            # TODO: make sure to fix the seed for first exp stage for different methods
            self.run_first_experimentation_stage(method)
            self.run_PE_stage(method)
            self.run_second_experimentation_stage(method)

        # how to store the outcome / util models for the different methods
        # maybe have an outcome_model_dict?

        # TODO: then have a way to store the optimization outcomes
