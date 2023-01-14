"""
This is a surrogate of the portfolio simulator, based on 3k samples found in port_evals.
credit to https://github.com/saitcakmak/BoRisk/blob/master/BoRisk/test_functions/portfolio_surrogate.py 
"""
import math
from typing import Optional
import torch
import gpytorch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_w_samples(
    bounds = torch.Tensor([[0,0], [1,1]]), 
    n = 50, 
    distribution = 'uniform'
):
    """
    Generate `n` samples of environmental variables from uniform distributions.
    (Later, can explore more complicated or data-driven distributions.)
    bounds: tensor
    n: number of samples we want to generate
    """

    if distribution == 'uniform':
        return torch.rand(n, bounds.shape[-1])
    # TODO: enable other distributions for w

    
# store w_samples in dict
w_samples_dict = {}
w_samples_dict['uniform'] = generate_w_samples()

class DistributionalPortfolioSurrogate(SyntheticTestFunction):
    r"""
    Surrogate of the portfolio simulator.
    User specifies the distributions for the environmental variables.
    Outputs statistics of the distribution of one design over the
        distribution of the environmental variables.
    """

    # Corresponding weights
    weights = None
    _optimizers = None
    dim = 3
    _bounds = [(0, 1) for _ in range(3)]

    def __init__(
        self, noise_std: Optional[float] = None, 
        negate: bool = False,
        w_distribution: str = 'uniform' 
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.model = None
        self.w_samples = w_samples_dict[w_distribution]
    
    def evaluate_true_one_design(self, x: Tensor) -> Tensor:
        """
        Evaluates the expected return of one design,
        over the distribution of environmental variables w.
        x: one design, shape 1x3
        """
        if self.model is not None:
            with torch.no_grad(), gpytorch.settings.max_cg_iterations(10000):
                # broadcast x to concatenate with all the w samples
                x_w = torch.cat(
                    (x.repeat(self.w_samples.shape[0], 1), self.w_samples),
                    dim = 1
                )
                posterior_mean = self.model.posterior(
                    x_w.to(dtype=torch.float32, device="cpu")
                ).mean.to(x)
                return torch.mean(posterior_mean, dim = 0)
        self.fit_model()
        return self.evaluate_true_one_design(x)

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        X: tensor of designs to evaluate; X.shape[-1]=3
        """

        results = torch.Tensor()

        for x in X:
            results = torch.cat((results, self.evaluate_true_one_design(x)), dim = 0)
        
        return results
        

    def fit_model(self):
        """
        If no state_dict exists, fits the model and saves the state_dict.
        Otherwise, constructs the model but uses the fit given by the state_dict.
        """
        # read the data
        data_list = list()
        for i in range(1, 31):
            data_file = os.path.join(script_dir, "port_evals", "port_n=100_seed=%d" % i)
            data_list.append(torch.load(data_file))

        # join the data together
        X = torch.cat(
            [data_list[i]["X"] for i in range(len(data_list))], dim=0
        ).squeeze(-2)
        Y = torch.cat(
            [data_list[i]["Y"] for i in range(len(data_list))], dim=0
        ).squeeze(-2)

        # fit GP
        noise_prior = GammaPrior(1.1, 0.5)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=[],
            noise_constraint=GreaterThan(
                0.000005,  # minimum observation noise assumed in the GP model
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

        # We save the state dict to avoid fitting the GP every time which takes ~3 mins
        try:
            state_dict = torch.load(
                os.path.join(script_dir, "portfolio_surrogate_state_dict.pt")
            )
            model = SingleTaskGP(X, Y, likelihood, outcome_transform=Standardize(m=1))
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            model = SingleTaskGP(X, Y, likelihood, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            from time import time

            start = time()
            fit_gpytorch_model(mll)
            print("fitting took %s seconds" % (time() - start))
            torch.save(
                model.state_dict(),
                os.path.join(script_dir, "portfolio_surrogate_state_dict.pt"),
            )
        self.model = model



# next TODO: utility function
# I know we said we want data from multiple time periods to constitute the high-dim outcome
# but what if we treat the outcomes from different w's as a high-dim outcome?



# Sait's original code



class PortfolioSurrogate(SyntheticTestFunction):
    r"""
    Surrogate of the portfolio simulator.
    """
    # The original set of random points
    w_samples = None
    # Corresponding weights
    weights = None
    _optimizers = None
    dim = 5
    _bounds = [(0, 1) for _ in range(5)]

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.model = None

    def evaluate_true(self, X: Tensor) -> Tensor:
        if self.model is not None:
            with torch.no_grad(), gpytorch.settings.max_cg_iterations(10000):
                return self.model.posterior(
                    X.to(dtype=torch.float32, device="cpu")
                ).mean.to(X)
        self.fit_model()
        return self.evaluate_true(X)

    def fit_model(self):
        """
        If no state_dict exists, fits the model and saves the state_dict.
        Otherwise, constructs the model but uses the fit given by the state_dict.
        """
        # read the data
        data_list = list()
        for i in range(1, 31):
            data_file = os.path.join(script_dir, "port_evals", "port_n=100_seed=%d" % i)
            data_list.append(torch.load(data_file))

        # join the data together
        X = torch.cat(
            [data_list[i]["X"] for i in range(len(data_list))], dim=0
        ).squeeze(-2)
        Y = torch.cat(
            [data_list[i]["Y"] for i in range(len(data_list))], dim=0
        ).squeeze(-2)

        # fit GP
        noise_prior = GammaPrior(1.1, 0.5)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=[],
            noise_constraint=GreaterThan(
                0.000005,  # minimum observation noise assumed in the GP model
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )

        # We save the state dict to avoid fitting the GP every time which takes ~3 mins
        try:
            state_dict = torch.load(
                os.path.join(script_dir, "portfolio_surrogate_state_dict.pt")
            )
            model = SingleTaskGP(X, Y, likelihood, outcome_transform=Standardize(m=1))
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            model = SingleTaskGP(X, Y, likelihood, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            from time import time

            start = time()
            fit_gpytorch_model(mll)
            print("fitting took %s seconds" % (time() - start))
            torch.save(
                model.state_dict(),
                os.path.join(script_dir, "portfolio_surrogate_state_dict.pt"),
            )
        self.model = model

