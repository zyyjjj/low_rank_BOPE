from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn
from ae.pca.transforms import PCAOutcomeTransform

from botorch.fit import fit_gpytorch_mll
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.decomposition import PCA


def compute_principal_axes(
    data: torch.Tensor, variance_explained_threshold: float, verbose: bool, **tkwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Perform PCA to get principal components and principal axes that
    explain over `variance_explained_threshold` of the variance in the data.

    Args:
        data: `num_samples x output_dim` tensor
        variance_explained_threshold: the fraction of variance in the data
            that we want the principal axes to explain
        verbose: boolean value, whether to print detailed information of PCA results
            (fraction of explained variance with each additional principal axis, fitted axes)

    Returns:
        learned_axes: `num_axes x original_output_dimension` tensor, with each row being a principal axis
        learned_PCs: `num_samples x num_axes` tensor,
            with the i-th row being the principal components for the i-th datapoint
    """

    # first compute all PCs
    pca = PCA()
    data_transformed = pca.fit_transform(data)

    # decide the number of principal axes to keep (that makes explained variance exceed the specified threshold)
    explained_variance = pca.explained_variance_ratio_
    exceed_thres = np.cumsum(explained_variance) > variance_explained_threshold
    num_axes = len(exceed_thres) - sum(exceed_thres) + 1

    learned_axes = pca.components_[:num_axes, :]
    learned_PCs = data_transformed[:, :num_axes]

    if verbose:
        print("explained variance", explained_variance)
        print("learned_axes", learned_axes)

    learned_axes = torch.tensor(learned_axes, **tkwargs)
    learned_PCs = torch.tensor(learned_PCs, **tkwargs)

    return learned_axes, learned_PCs


def initialize_model_PCA(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    variance_explained_threshold: float,
    ground_truth_principal_axes: torch.Tensor = None,  # for sanity check only
    PC_noise_level: Optional[float] = None,
    num_axes: Optional[int] = None,
    state_dict: Dict = None,
    likelihood: Type[Likelihood] = None,
    verbose: bool = False,
    **tkwargs,
) -> Tuple[GPyTorchModel, torch.Tensor]:
    r"""Initialize a GP model on the PCs inferred from a given dataset of inputs and observations.

    Args:
        train_X: `num_samples x input_dim` tensor where each row is an input datapoint
        train_Y: `num_samples x output_dim` tensor where each row is an output datapoint
        variance_explained_threshold: a number between 0 and 1; threshold for the fraction of variance we want explained by the principal components we include
        ground_truth_principal_axes: `num_axes x output_dim` tensor where rows are the ground truth principal axes underlying the data; for debugging purpose only
        PC_noise_level: nonnegative real number specifying the magnitude of PC noise, to plug into FixedNoiseGP();
            if None, fit a SingleTaskGP() and have it infer the noise
        num_axes: positive integer specifying the number of principal axes to keep;
            if None, use the smallest number of axes that together explain over `variance_explained_threshold` of variance
        state_dict: dictionary for model hyperparameters; if given, warm-starts the GP model fitting from the supplied hyperparameter values
        verbose: if True, print details about the PCA fitting
        likelihood: gpytorch Likelihood

    Returns:
        model_PC: initialized GP model on PCs
        axes_learned: `num_axes x output_dim` tensor where rows are fitted principal axes
    """

    if ground_truth_principal_axes is None:
        axes_learned, PC_learned = compute_principal_axes(
            train_Y.detach().numpy(), variance_explained_threshold, verbose
        )
    else:
        # for debugging only, supply the ground truth principal axes rather than fitting the axes from data
        assert (
            ground_truth_principal_axes.shape[1] == train_Y.shape[1]
        ), "dimension 1 of ground_truth_principal_axes should equal dimension 1 of train_Y"

        # directly compute principal components from the supplied ground truth principal axes
        axes_learned = ground_truth_principal_axes
        PC_learned = torch.matmul(train_Y, torch.transpose(axes_learned, 0, 1))

    # create OutcomeTransform that performs PCA
    pca_transform = PCAOutcomeTransform(
        variance_explained_threshold=variance_explained_threshold, num_axes=num_axes
    )

    # Here we have two options for which models to use
    if PC_noise_level is None:
        # Option 1: fit a SingleTaskGP(), and have the PC noise be inferred (in reality we typically don't know the PC noise)
        model_PC = SingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=ChainedOutcomeTransform(
                **{
                    "standardize": Standardize(train_Y.shape[-1]),
                    "pca": pca_transform,
                }
            ),
            likelihood=likelihood,
        )

    else:
        # Option 2: fit a FixedNoiseGP() using the supplied PC noise
        # TODO: how do we deal with the case where different PCs have different levels of noise?
        model_PC = FixedNoiseGP(
            train_X,
            train_Y,
            torch.ones_like(train_Y) * PC_noise_level,  # TODO: double check this
            outcome_transform=ChainedOutcomeTransform(
                **{
                    "standardize": Standardize(train_Y.shape[-1]),
                    "pca": pca_transform,
                }
            ),
            likelihood=likelihood,
        )

    # load state dict if it is passed
    if state_dict is not None:
        model_PC.load_state_dict(state_dict)

    return model_PC, axes_learned


# deprecated by PCAOutcomeTransform()
def get_posterior_predictions_PCA(
    test_X: torch.Tensor, model_PC: Type[Model], axes_learned: torch.Tensor, **tkwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fit GP model and compute posterior distributions for principal components and metrics.

    Args:
        test_X: `num_test_points x input_dim` tensor
        model_PC: GP model on PCs
        axes_learned: `num_axes x output_dim` tensor where rows are learned principal axes

    Returns:
        PC_posterior_mean: `num_test_points x num_PCs` tensor
        PC_posterior_variance: `num_test_points x num_PCs` tensor
        metric_posterior_mean: `num_test_points x output_dim` tensor
        metric_posterior_variance: `num_test_points x output_dim` tensor
    """

    mll_PC = ExactMarginalLogLikelihood(model_PC.likelihood, model_PC)

    # fit model
    fit_gpytorch_mll(mll_PC)

    # get PC predictive distributions
    PC_posterior_mean = model_PC.posterior(test_X).mean.to(**tkwargs)
    PC_posterior_variance = model_PC.posterior(test_X).variance.to(**tkwargs)

    # transform to metric predictive distributions
    metric_posterior_mean = torch.matmul(PC_posterior_mean, axes_learned)
    metric_posterior_variance = torch.matmul(
        PC_posterior_variance, torch.square(axes_learned)
    )

    return (
        PC_posterior_mean,
        PC_posterior_variance,
        metric_posterior_mean,
        metric_posterior_variance,
    )


def initialize_model_ST(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    train_Yvar: float = None,
    infer_noise: bool = False,
    state_dict: Dict = None,
    likelihood: Type[Likelihood] = None,
    # TODO: I should have an argument that allows specifying the noise hyperprior
    **tkwargs,
) -> GPyTorchModel:
    r"""Initialize a GP model on the metrics from a given dataset of inputs and observations.

    Args:
        train_X: `num_samples x input_dim` tensor where each row is an input datapoint
        train_Y: `num_samples x output_dim` tensor where each row is an output datapoint
        train_Yvar: nonnegative real number specifying the variance of observations (not SD)
        infer_noise: if True, fit a SingleTaskGP(); if False, fit a FixedNoiseGP()
        state_dict: dictionary for model hyperparameters; if given, warm-starts the GP model fitting from the supplied hyperparameter values
        likelihood: gpytorch Likelihood

    Returns:
        model_ST: initialized GP model on metric values, one independent single-task GP for each metric
    """

    # standardize outcome in GP fitting: https://botorch.org/api/_modules/botorch/models/transforms/outcome.html#Standardize

    if infer_noise:
        model_ST = SingleTaskGP(
            train_X,
            train_Y,  # TODO: handle noise hyperpriors more carefully, don't just always go with the default
            outcome_transform=Standardize(train_Y.shape[-1]),
            likelihood=likelihood,
        )
    else:
        model_ST = FixedNoiseGP(
            train_X,
            train_Y,
            torch.ones(train_Y.shape) * train_Yvar,
            # TODO: later enable specifying different noise levels for different metrics
            outcome_transform=Standardize(train_Y.shape[-1]),
            likelihood=likelihood,
        )

    # load state dict if it is passed
    if state_dict is not None:
        model_ST.load_state_dict(state_dict)

    return model_ST


def get_posterior_predictions_ST(
    test_X: torch.Tensor, model: Type[Model], **tkwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fit (independent) GP models on metrics and compute posterior distributions for metrics.

    Args:
        test_X: `num_test_points x input_dim` tensor
        model: GP model on metrics

    Returns:
        metric_posterior_mean: `num_test_points x output_dim` tensor
        metric_posterior_variance: `num_test_points x output_dim` tensor
    """

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # fit model
    fit_gpytorch_mll(mll)

    # get metric predictions
    metric_posterior_mean = model.posterior(test_X).mean.to(**tkwargs)
    metric_posterior_variance = model.posterior(test_X).variance.to(**tkwargs)

    return metric_posterior_mean, metric_posterior_variance


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
