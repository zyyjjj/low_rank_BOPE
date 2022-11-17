import torch
from typing import Dict, Optional

import gpytorch

from pref_learning_helpers import (
    check_pref_model_fit,
    fit_pref_model,
    gen_comps,
    gen_exp_cand,
    generate_random_inputs,
    ModifiedFixedSingleSampleModel,
)
from botorch.acquisition import LearnedObjective
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, LCMKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


def subspace_recovery_error(
    axes_learned: Tensor, ground_truth_principal_axes: Tensor
) -> float:
    r"""
    Compute $(\|(I-VV^T)A\|_F / \|A\|_F)^2$, where
    A = ground_truth_principal_axes transposed, each column a ground truth
    principal axis for data generation;
    V = axes_learned transposed, each column a learned principal axis.

    This quantity serves as a diagnostic of ``how well axes_learned
    recovers the true subspace from which the outcomes are generated".
    This can be used in synthetic experiments where we know the
    underlying outcome generation process.

    Args:
        axes_learned: num_axes x outcome_dim tensor,
            each row a learned principal axis
        ground_truth_principal_axes: true_latent_dim x outcome_dim tensor,
            each row a ground truth principal axis
    Returns:
        squared Frobenius norm of ground_truth_principal_axes projected onto
            the orthogonal of PCA-learned subspace, divided by the squared
            Frobenius of ground_truth_principal_axes matrix, which is
            equal to the true latent dimension.
    """

    latent_dim, outcome_dim = ground_truth_principal_axes.shape

    # I-VV^T, projection onto orthogonal space of V, shape is outcome_dim x outcome_dim
    orth_proj = (
        torch.eye(outcome_dim) - torch.transpose(axes_learned, -2, -1) @ axes_learned
    )

    # (I-VV^T)A
    true_subspace_lost = orth_proj @ torch.transpose(
        ground_truth_principal_axes, -2, -1
    )
    true_subspace_lost_frobenius_norm = torch.linalg.norm(true_subspace_lost)

    # ||A||^2 = latent_dim
    frac_squared_norm = torch.square(true_subspace_lost_frobenius_norm) / latent_dim

    return frac_squared_norm.item()


def empirical_max_outcome_error(Y: Tensor, axes_learned: Tensor) -> float:
    r"""
    Compute the diagnostic $max_i \|(I-VV^T)y_i\|_2$,
    where $y_i$ is the ith row of Y, representing an outcome data point,
    and V = axes_learned transposed, each column a learned principal axis.

    This quantity serves as a data-dependent empirical estimate of
    ``worst case magnitude of unmodeled outcome component",
    which reflects PCA representation quality.

    Args:
        Y: num_samples x outcome_dim tensor,
            each row an outcome data point
        axes_learned: num_axes x outcome_dim tensor,
            each row a learned principal axis
    Returns:
        maximum norm, among the data points in Y, of the outcome component
            projected onto the orthogonal space of V
    """

    outcome_dim = Y.shape[-1]

    # I-VV^T, projection onto orthogonal space of V, shape is outcome_dim x outcome_dim
    orth_proj = (
        torch.eye(outcome_dim) - torch.transpose(axes_learned, -2, -1) @ axes_learned
    )

    # Y @ orth_proj is num_samples x outcome_dim
    Y_orth_proj_norm = torch.linalg.norm(Y @ orth_proj, dim=1)

    return torch.max(Y_orth_proj_norm).item()


def mc_max_outcome_error(problem, axes_learned, n_test) -> float:
    r"""
    Compute the diagnostic $\mathbb{E}[max_x \|(I-VV^T)f(x)\|_2]$,
    through Monte Carlo sampling. V = axes_learned transposed,
    where each column of V is a learned principal axis.

    This quantity is a Monte Carlo estimate of
    ``expected worst case magnitude of unmodeled outcome component",
    which reflects PCA representation quality.

    Args:
        problem: a TestProblem, maps inputs to outcomes
        axes_learned: num_axes x outcome_dim tensor,
            each row a learned principal axis
        n_test: number of Monte Carlo samples to take
    Returns:
        maximum norm, among the sampled data points, of the
            outcome component projected onto the orthogonal space of V
    """

    test_X = generate_random_inputs(problem, n_test).detach()
    test_Y = problem.evaluate_true(test_X).detach()

    outcome_dim = test_Y.shape[-1]

    # I-VV^T, projection onto orthogonal space of V, shape is outcome_dim x outcome_dim
    orth_proj = (
        torch.eye(outcome_dim) - torch.transpose(axes_learned, -2, -1) @ axes_learned
    )

    # test_Y @ orth_proj is num_samples x outcome_dim
    test_Y_orth_proj_norm = torch.linalg.norm(test_Y @ orth_proj, dim=1)

    return torch.max(test_Y_orth_proj_norm).item()


def empirical_max_util_error(Y, axes_learned, util_func) -> float:
    r"""
    Compute the diagnostic $max_i \|g(y_i) - g(VV^T y_i)\|_2$,
    where $y_i$ is the ith row of Y, representing an outcome data point,
    and V = axes_learned transposed, each column a learned principal axis.

    This quantity serves as a data-dependent empirical estimate of
    ``worst case magnitude of utility recovery error", which reflects
    how well we can learn the utility function with the input space being
    outcome spaced projected onto the subspace V.

    Args:
        Y: num_samples x outcome_dim tensor,
            each row an outcome data point
        axes_learned: num_axes x outcome_dim tensor,
            each row a learned principal axis
        util_func: ground truth utility function (outcome -> utility)
    Returns:
        maximum difference, among the data points in Y, of the
            true utility value and the utility value of the projection
            of Y onto the subpace V.
    """

    # VV^T, projection onto subspace spanned by V, shape is outcome_dim x outcome_dim
    proj = torch.transpose(axes_learned, -2, -1) @ axes_learned

    # compute util(Y) - util(VV^T Y)
    util_difference = torch.abs(util_func(Y) - util_func(Y @ proj))

    return torch.max(util_difference).item()


def mc_max_util_error(problem, axes_learned, util_func, n_test) -> float:
    r"""
    Compute the diagnostic $\mathbb{E}[max_x \|g(f(x)) - g(VV^T f(x))\|_2$,
    through Monte Carlo sampling. V = axes_learned transposed, where
    each column of V is a learned principal axis. f is the true outcome function
    and g is the true utility function.

    This quantity is a Monte Carlo estimate of ``expected worst case
    magnitude of utility recovery error", which reflects
    how well we can learn the utility function with the input space being
    outcome spaced projected onto the subspace V.

    Args:
        problem: a TestProblem, maps inputs to outcomes
        Y: num_samples x outcome_dim tensor,
            each row an outcome data point
        axes_learned: num_axes x outcome_dim tensor,
            each row a learned principal axis
        util_func: ground truth utility function (outcome -> utility)
    Returns:
        maximum difference, among the sampled data points, of the
            true utility value and the utility value of the projection
            of sampled outcome data onto the subpace V.
    """

    test_X = generate_random_inputs(problem, n_test).detach()
    test_Y = problem.evaluate_true(test_X).detach()

    # VV^T, projection onto subspace spanned by V, shape is outcome_dim x outcome_dim
    proj = torch.transpose(axes_learned, -2, -1) @ axes_learned

    # compute util(Y) - util(VV^T Y)
    test_util_difference = torch.abs(util_func(test_Y) - util_func(test_Y @ proj))

    return torch.max(test_util_difference).item()



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
