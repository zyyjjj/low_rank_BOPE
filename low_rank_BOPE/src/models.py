from typing import Optional

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import (Kernel, LCMKernel, MaternKernel, RBFKernel,
                              ScaleKernel)
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.torch_priors import GammaPrior, Prior
from torch import Tensor


def compute_variance_explained_per_axis(data, axes, **tkwargs) -> torch.Tensor:
    r"""
    Compute the fraction of variance explained with each axis supplied in the axes tensor

    Args:
        data: `num_datapoints x output_dim` tensor
        axes: `num_axes x output_dim` tensor where each row is a principal axis

    Returns:
        var_explained: `1 x num_axes` tensor with i-th entry being the fraction of variance explained by the i-th supplied axis
    """

    total_var = sum(torch.var(data, dim=0)).item()

    # check if each row of axes is normalized; if not, divide by L2 norm
    axes = torch.div(axes, torch.linalg.norm(axes, dim=1).unsqueeze(1))

    var_explained = torch.var(
        torch.matmul(data, axes.transpose(0, 1).to(**tkwargs)), dim=0
    ).detach()
    var_explained = var_explained / total_var

    return var_explained

# LCM model class
class MultitaskGPModel(GPyTorchModel, ExactGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        rank: Optional[int] = None,
        task_covar_prior: Optional[Prior] = None,
        likelihood: Optional[MultitaskGaussianLikelihood] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ):

        r"""
        Initialize model class for multi-output GP models.

        Args:
            train_X: `num_samples x input_dim` tensor
            train_Y: `num_samples x outcome_dim` tensor
            latent_dim: number of basis kernels to use
            outcome_transform: OutcomeTransform
            input_transform: InputTransform

        """

        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        if rank is None:
            rank = 1
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._num_outputs = train_Y.shape[-1]
        
        ard_num_dims, num_tasks = train_X.shape[-1], train_Y.shape[-1]

        if task_covar_prior is None:
            sd_prior = GammaPrior(1.0, 0.15)
            eta = 0.5
            task_covar_prior = LKJCovariancePrior(num_tasks, eta, sd_prior)
    
        if likelihood is None:            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_tasks)

        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        self.covar_module = LCMKernel(
            base_kernels=[MaternKernel(
                nu = 2.5,
                ard_num_dims = ard_num_dims,
                lengthscale_prior = GammaPrior(3.0, 6.0)
            )] * latent_dim,
            num_tasks=num_tasks,
            rank=rank,
            # task_covar_prior=task_covar_prior, # not using task covar prior for now 
            # related issue https://github.com/cornellius-gp/gpytorch/issues/2263
        )

        self.to(train_X)

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
def make_modified_kernel(ard_num_dims, a=0.2, b=5.0):
    # ls_prior = GammaPrior(1.2, 0.5)
    ls_prior = GammaPrior(2.4, 2.7) # consistent w Jerry's Mar22 update
    ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate

    covar_module = ScaleKernel(
        RBFKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=ls_prior,
            lengthscale_constraint=GreaterThan(
                lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
            ),
        ),
        outputscale_prior=SmoothedBoxPrior(a=a, b=b),
        outputscale_constraint=Interval(lower_bound=a, upper_bound=b),
        # outputscale_prior=SmoothedBoxPrior(a=1e-2, b=1e2),
        # outputscale_prior=SmoothedBoxPrior(a=0.1, b=10),
        # outputscale_constraint=Interval(lower_bound=0.1, upper_bound=10),
    )

    return covar_module




# Qing's code for map-saas on the PCs

    if pc_model_type == "map_saas":
        # Y_transformed: [n x num_axes]
        Y_transformed, _ = pca_transform(train_Y)

        Xs = [train_X for _ in range(Y_transformed.shape[-1])]
        Ys = [Y_transformed[:, [i]] for i in range(Y_transformed.shape[-1])]
        Yvars = [
            torch.full(Y_transformed[:, [i]].size(), torch.nan, **tkwargs)
            for i in range(Y_transformed.shape[-1])
        ]

        model_PC = get_and_fit_map_saas_model(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            task_features=[],
            fidelity_features=[],
            metric_names=[],
        )

        # load state dict if it is passed [Qing: not sure this is used in benchmark]
        if state_dict is not None:
            model_PC.load_state_dict(state_dict)

        return model_PC, pca_transform.axes_learned
