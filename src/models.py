import torch
from torch import Tensor
import gpytorch
from botorch.fit import fit_gpytorch_model, fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel, LCMKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.transforms.input import InputTransform

# LCM model class
class MultitaskGPModel(GPyTorchModel, ExactGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Likelihood,
        num_tasks: int,
        multitask_kernel: Kernel,
        outcome_transform: OutcomeTransform = None,
        input_transform: InputTransform = None,
    ):

        r"""
        Initialize model class for multi-output GP models.

        Args:
            train_X: `num_samples x input_dim` tensor
            train_Y: `num_samples x outcome_dim` tensor
            likelihood: Gpytorch likelihood
            num_tasks: number of outcomes
            multitask_kernel: a multi-output kernel
            outcome_transform: OutcomeTransform
            input_transform: InputTransform

        """

        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._num_outputs = num_tasks

        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = multitask_kernel

    def forward(self, x: Tensor):
        r"""
        Return posterior distribution at new point x
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def fit_LCM_model(
    train_X: Tensor,
    train_Y: Tensor,
    num_tasks: int,
    num_basis_kernels: int,
    rank: int = 1,
):
    r"""
    Fit LCM model with specified number of basis kernels and rank.
    Args:
        train_X: `num_samples x input_dim` tensor
        train_Y: `num_samples x outcome_dim` tensor
        num_tasks: number of outcomes
        num_basis_kernels: number of basis kernels in LCM kernel
        rank: rank of basis kernels, default to 1
    Returns:
        fitted model
    """
    lcm_kernel = LCMKernel(
        base_kernels=[MaternKernel()] * num_basis_kernels, num_tasks=num_tasks, rank=1
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    lcm_model = MultitaskGPModel(
        train_X, train_Y, likelihood, num_tasks=num_tasks, multitask_kernel=lcm_kernel
    )

    mll_lcm = ExactMarginalLogLikelihood(lcm_model.likelihood, lcm_model)
    fit_gpytorch_model(mll_lcm)

    return lcm_model


# modified kernel with change in hyperpriors
def make_modified_kernel(ard_num_dims):
    ls_prior = GammaPrior(1.2, 0.5)
    ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate

    covar_module = ScaleKernel(
        RBFKernel(
            # batch_shape=self.batch_shape,
            ard_num_dims=ard_num_dims,
            lengthscale_prior=ls_prior,
            lengthscale_constraint=GreaterThan(
                lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
            ),
        ),
        # outputscale_prior=SmoothedBoxPrior(a=1e-2, b=1e2),
        # outputscale_prior=SmoothedBoxPrior(a=0.1, b=10),
        # outputscale_constraint=Interval(lower_bound=0.1, upper_bound=10),
        outputscale_prior=SmoothedBoxPrior(a=0.2, b=5.0),
        outputscale_constraint=Interval(lower_bound=0.2, upper_bound=5.0),
    )

    return covar_module
