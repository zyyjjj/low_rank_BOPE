# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

# TODO: also find a way to enable PCR

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
    }


    def __init__(self, problem, methods, PE_strategy, **kwargs) -> None:
        """
        specify experiment settings
            problem:
            methods: list of statistical methods
            PE_strategy: PE strategy to use
        """

        # TODO: if we pass in a list of methods
        # where in this class do we loop over the methods?


        # pre-specified experiment metadata
        self.problem = (
            problem  # if we pass in a module; do we want to pass in a string?
        )
        # self.outcome_dim = # TODO: can we get this from problem?
        self.input_dim = problem.dim

        # self.attr_list stores default values
        # then overwrite with kwargs
        for key in self.attr_list.keys():
            setattr(self, key, self.attr_list[key])
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        # results logging
        # but should have a list for each method
        self.within_session_results = []
        self.exp_candidate_results = []

        # specify outcome and input transforms, covariance modules

        # TODO: design question: do we want to pass in one method,
        # or test a bunch of methods at once?
        # recall that some benchmarks may depend on PCA number of axes


        # TODO: for each method, have a dict of model class, transforms, other hyperparameters
        # make this dictionary a class attribute?
        # or maybe store this dict in an external yaml file for cleanness
        # then make self.methods_dict = {}





        # error handling
        self.passed = False
        self.fit_count = 0

    def run_first_experimentation_stage(self):
        self.generate_random_experiment_data(self.problem, self.initial_experimentation_batch)
        self.fit_outcome_model()


    def run_PE_stage(self):
        self.generate_random_pref_data()
        self.fit_pref_model()
        self.run_pref_learning()
        self.find_max_posterior_mean()
        pass

    def run_second_experimentation_stage(self):
        self.generate_experiment_candidate()
        pass

    def generate_random_experiment_data(self, n):
        r"""Generate n observations of experimental designs and outcomes.
        Args:
            problem: a TestProblem
            n: number of samples
        Returns:
            X: `n x problem input dim` tensor of sampled inputs
            Y: `n x problem outcome dim` tensor of noisy evaluated outcomes at X
        """

        X = (
            draw_sobol_samples(bounds=self.problem.bounds, n=1, q=n)
            .squeeze(0)
            .to(torch.double)
            .detach()
        )
        Y = self.problem(X).detach()

        return X, Y

    def fit_outcome_model(self):
        # use self.X, self.Y, self.outcome_transform
        # maybe input model class and other kwargs hyperparameters

        # if self.use_PCR: don't use the specified transforms
        # instead, run PCA, then run regression
        # perhaps this warrants another transform object
        # -- no, just compute the regression and make a LinearProjOutcomeTransform out of it

        # elif method == "pcr":
            # P, S, V = torch.svd(Y)
            # # then run regression from P (PCs) onto util_vals
            # # select top k entries of PC_coeff
            # # retain the corresponding columns in P

            # # torch.matmul(torch.transpose(P), P)
            # reg = LinearRegression().fit(np.array(P), np.array(util_vals))
            # dims_to_keep = np.argpartition(reg.coef_, -config["outcome_dim"] // 3)[
            #     -config["outcome_dim"] // 3 :
            # ]

            # # transform corresponding to dims_to_keep
            # pcr_axes = torch.tensor(torch.transpose(V[:, dims_to_keep], -2, -1))
            # # then plug these into LinearProjection O/I transforms
            # transforms_covar_dict["pcr"] = {
            #     "outcome_tf": LinearProjectionOutcomeTransform(pcr_axes),
            #     "input_tf": LinearProjectionInputTransform(pcr_axes),
            #     "covar_module": make_modified_kernel(
            #         ard_num_dims=config["outcome_dim"] // 3
            #     ),
            # }

        # if use PCA or PCR, update self.outcome_transform
        # and self.input_transform here



        pass

    def fit_pref_model(self, utility_aware: bool):
        # if utility aware, run PCR
        pass

    def run_pref_learning(self):
        # accumulate pairwise comparisons
        pass

    def find_max_posterior_mean(self):
        pass

    def generate_experiment_candidate(self):
        pass

    def generate_random_pref_data(self):
        pass

    def run_BOPE_loop(self):
        # have a flag for whether the fitting is successful or not
        self.run_first_experimentation_stage()
        self.run_BOPE_loop()
        self.run_second_experimentation_stage()

        # then have a way to store the optimization outcomes
