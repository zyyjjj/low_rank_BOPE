#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from typing import Optional

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.deterministic import DeterministicModel
from botorch.models.model import Model

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


def gen_comps(
    util_vals: Tensor, comp_noise_type: str = None, comp_noise: float = None
) -> Tensor:
    r"""Create pairwise comparisons.
    Args:
        util_vals: `num_outcomes x 1` tensor of utility values
        comp_noise_type: type of comparison noise to inject, one of {'constant', 'probit'}
        comp_noise: parameter related to probability of making a comparison mistake
    Returns:
        comp_pairs: `(num_outcomes // 2) x 2` tensor showing the preference,
            with the more preferable outcome followed by the other one in each row
    """
    assert comp_noise_type is None, "do not support comp noise now"

    cpu_util = util_vals.cpu()

    comp_pairs = []
    for i in range(cpu_util.shape[0] // 2):
        i1 = i * 2
        i2 = i * 2 + 1
        if cpu_util[i1] > cpu_util[i2]:
            new_comp = [i1, i2]
            util_diff = cpu_util[i1] - cpu_util[i2]
        else:
            new_comp = [i2, i1]
            util_diff = cpu_util[i2] - cpu_util[i1]

        new_comp = torch.tensor(new_comp, device=util_vals.device, dtype=torch.long)
        comp_pairs.append(new_comp)

    comp_pairs = torch.stack(comp_pairs)

    return comp_pairs


def generate_random_pref_data(problem, n, outcome_model, util_func):
    X = (
        draw_sobol_samples(
            bounds=problem.bounds,
            n=1,
            q=2 * n,
        )
        .squeeze(0)
        .to(torch.double)
        .detach()
    )

    # sampled from outcomes
    train_outcomes = outcome_model.posterior(X).rsample().squeeze(0).detach()
    util_val = util_func(train_outcomes)
    comps = gen_comps(util_val)
    return comps, train_outcomes, util_val


def fit_pca(
    train_Y: Tensor,
    var_threshold: float = 0.9,
    weights: Optional[Tensor] = None,
    standardize: Optional[bool] = True,
):
    r"""
    Perform PCA on supplied data with optional weights.
    Args:
        train_Y: `num_samples x outcome_dim` tensor of data
        var_threshold: threshold of variance explained
        weights: `num_samples x 1` tensor of weights to add on each data point
        standardize: whether to standardize train_Y before computing PCA
    Returns:
        pca_axes: `latent_dim x outcome_dim` tensor where each row is a pca axis
    """

    # TODO: maybe add optional arg num_axes

    if weights is not None:
        # weighted pca
        assert (
            weights.shape[0] == train_Y.shape[0]
        ), f"weights shape {weights.shape} does not match train_Y shape {train_Y.shape}, "
        assert (weights >= 0).all(), "weights must be nonnegative"

        # remove the entries with weight=0 # TODO: check this
        valid = torch.nonzero(weights.squeeze(1)).squeeze(1)
        print("valid: ", valid)

        print(
            "valid shape: ",
            valid.shape,
            "weights shape: ",
            weights.shape,
            "train_Y shape: ",
            train_Y.shape,
        )
        weighted_mean = (train_Y[valid] * weights[valid]).sum(dim=0) / weights[
            valid
        ].sum(0)
        train_Y_centered = weights[valid] * (train_Y[valid] - weighted_mean)

    else:
        # unweighted pca
        train_Y_centered = train_Y - train_Y.mean(dim=0)

    if standardize:
        # standardize
        U, S, V = torch.svd(train_Y_centered / train_Y_centered.std(dim=0))
    else:
        # don't standardize, just center
        U, S, V = torch.svd(train_Y_centered)

    S_squared = torch.square(S)
    explained_variance = S_squared / S_squared.sum()

    exceed_thres = np.cumsum(explained_variance.detach().numpy()) > var_threshold
    num_axes = len(exceed_thres) - sum(exceed_thres) + 1

    pca_axes = torch.tensor(
        torch.transpose(V[:, :num_axes], -2, -1), dtype=torch.double
    )

    return pca_axes


# modified kernel with change in hyperpriors
def make_modified_kernel(ard_num_dims, a=0.01, b=100):
    # ls_prior = GammaPrior(1.2, 0.5)
    ls_prior = GammaPrior(2.4, 2.7)  # consistent w Jerry's Mar22 update
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
    )
    return covar_module


class ModifiedFixedSingleSampleModel(DeterministicModel):
    r"""
    A deterministic model defined by a single sample `w`.

    Given a base model `f` and a fixed sample `w`, the model always outputs

        y = f_mean(x) + f_stddev(x) * w

    We assume the outcomes are uncorrelated here.

    This is modified from FixedSingleSampleModel to handle dimensionality reduction.
    For models with dim reduction, model.num_outputs is the reduced outcome dimension,
    whereas we want w to be in the original outcome dimension.
    In this modification, we define self.w within forward() rather than __init__(),
    where we fix the dimensionality of w to be posterior(X).event_shape[-1].
    """

    def __init__(
        self, model: Model, outcome_dim: int, w: Optional[torch.Tensor] = None
    ) -> None:
        r"""
        Args:
            model: The base model.
            outcome_dim: dimensionality of the outcome space
            w: A 1-d tensor with length = outcome_dim.
                If None, draw it from a standard normal distribution.
        """
        super().__init__()
        self.model = model
        self.w = torch.randn(outcome_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        post = self.model.posterior(X)

        # return post.mean + post.variance.sqrt() * self.w.to(X)
        # adding jitter to avoid numerical issues
        return post.mean + torch.sqrt(post.variance + 1e-8) * self.w.to(X)
