from copy import deepcopy
from logging import Logger
import logging 

from typing import Optional, Tuple

import torch
import torch.nn as nn
# from ax.utils.common.logger import get_logger
from botorch.fit import fit_gpytorch_mll
from botorch.models import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
    SingleTaskGP,
)
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.likelihoods.pairwise import PairwiseLikelihood
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
)
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    OutcomeTransform,
    Standardize
)
from botorch.utils.sampling import draw_sobol_samples
from low_rank_BOPE.autoencoder.transforms import (
    LinearProjectionInputTransform, LinearProjectionOutcomeTransform,
)
from low_rank_BOPE.autoencoder.utils import (
    fit_pca,
    make_modified_kernel,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.module import Module
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

# logger: Logger = get_logger(__name__)
logger = logging.getLogger("botorch")

##############################################################################################################
##############################################################################################################
# Autoencoder model class

class Autoencoder(nn.Module):
    def __init__(self, latent_dims, output_dims, **tkwargs):
        super(Autoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = Encoder(latent_dims, output_dims, **tkwargs)
        self.decoder = Decoder(latent_dims, output_dims, **tkwargs)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, latent_dims, output_dims, **tkwargs):
        super(Encoder, self).__init__()
        # TODO start with one layer
        self.linear = nn.Linear(output_dims, latent_dims, **tkwargs)

    def forward(self, x):
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_dims, **tkwargs):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dims, output_dims, **tkwargs)

    def forward(self, z):
        z = torch.sigmoid(self.linear(z))
        return z


def get_autoencoder(
    train_Y: Tensor,
    latent_dims: int,
    pre_train_epoch: int,
) -> Autoencoder:
    """Instantiate an autoencoder."""
    output_dims = train_Y.shape[-1]
    tkwargs = {"dtype": train_Y.dtype, "device": train_Y.device}
    autoencoder = Autoencoder(latent_dims, output_dims, **tkwargs)
    if pre_train_epoch > 0:
        autoencoder = train_autoencoder(autoencoder, train_Y, pre_train_epoch)
    return autoencoder


def train_autoencoder(
    autoencoder: Autoencoder, train_outcomes: Tensor, epochs=200
) -> Autoencoder:
    """One can pre-train an AE with outcome data (via minimize L2 loss)."""
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        opt.zero_grad()
        outcomes_hat = autoencoder(train_outcomes)
        loss = ((train_outcomes - outcomes_hat) ** 2).sum()  # L2 loss functions
        if epoch % 100 == 0:
            logger.info(f"Pre-train autoencoder epoch {epoch}: loss func = {loss}")
        loss.backward()
        opt.step()
    return autoencoder


##############################################################################################################
##############################################################################################################
# Outcome model class and fitting outcome models

def initialize_outcome_model(
    train_X: Tensor,
    train_Y: Tensor,
    latent_dims: int
) -> Tuple[SingleTaskGP, Likelihood]:
    outcome_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(latent_dims),
    )
    mll = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
    return outcome_model, mll


def get_fitted_pca_outcome_model(
    train_X: Tensor,
    train_Y: Tensor,
    pca_var_threshold: float,
    standardize: bool = False,
) -> SingleTaskGP:
    projection = fit_pca(
        train_Y = train_Y,
        var_threshold = pca_var_threshold,
        weights = None,
        standardize=standardize,
    )
    outcome_tf = ChainedOutcomeTransform(
        **{
            "projection": LinearProjectionOutcomeTransform(projection_matrix=projection),
            "standardize": Standardize(projection.shape[0])
        }
    )
    logger.info(f"pca projection matrix shape: {projection.shape}")
    outcome_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=outcome_tf,
    )
    mll_outcome = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
    fit_gpytorch_mll(mll_outcome)

    return outcome_model


def get_fitted_standard_outcome_model(train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
    """Fit a single-task outcome model."""
    outcome_dim = train_Y.shape[-1]
    outcome_model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(train_X.shape[-1]),
        outcome_transform=Standardize(outcome_dim),
    )
    mll_outcome = ExactMarginalLogLikelihood(outcome_model.likelihood, outcome_model)
    fit_gpytorch_mll(mll_outcome)
    
    return outcome_model


def get_fitted_autoencoded_outcome_model(
    train_X: Tensor,
    train_Y: Tensor,
    latent_dims: int,
    num_joint_train_epochs: int,
    num_autoencoder_pretrain_epochs: int,
    # num_unlabeled_outcomes: int,
    autoencoder: Optional[Autoencoder] = None,
    fix_vae: bool = False
) -> Tuple[SingleTaskGP, Autoencoder]:
    
    # TODO: still want this? 
    # not very reasonable to gen posterior samples from the outcome model yet to be trained
    # if (
    #     (outcome_model is not None)
    #     and (bounds is not None)
    #     and (num_unlabeled_outcomes > 0)
    # ):
    #     unlabeled_train_Y = _get_unlabeled_outcomes(
    #         outcome_model=outcome_model,
    #         bounds=bounds,
    #         nsample=num_unlabeled_outcomes,
    #     )
    #     unlabeled_train_Y = torch.cat((train_Y, unlabeled_train_Y), dim=0)
    # else:
    #     unlabeled_train_Y = train_Y

    if autoencoder is None:
        autoencoder = get_autoencoder(
            # train_Y=unlabeled_train_Y,
            train_Y=train_Y,
            latent_dims=latent_dims,
            pre_train_epoch=num_autoencoder_pretrain_epochs,
        )
    # get latent embeddings for train_Y
    train_Y_latent = autoencoder.encoder(train_Y).detach()
    outcome_model, mll_outcome = initialize_outcome_model(
        train_X=train_X, train_Y=train_Y_latent, latent_dims=latent_dims
    )

    # train outcome model under fixed vae
    outcome_model_, _, autoencoder_ = jointly_optimize_models(
        train_X=train_X,
        train_Y=train_Y,
        autoencoder=autoencoder,
        num_epochs=num_joint_train_epochs,
        outcome_model=outcome_model,
        mll_outcome=mll_outcome,
        train_ae=not fix_vae,
        train_outcome_model=True,
        train_util_model=False
    )

    return outcome_model_, autoencoder_


##############################################################################################################
##############################################################################################################
# Utility model class and fitting util models

class HighDimPairwiseGP(PairwiseGP):
    """Pairwise GP for high-dim outcomes. A thin wrapper over PairwiseGP to take a trained
    auto-encoder to map high-dim outcomes to a low-dim outcome spaces.
    """

    def __init__(
        self,
        datapoints: Tensor,
        comparisons: Tensor,
        autoencoder: Optional[nn.Module] = None,
        likelihood: Optional[PairwiseLikelihood] = None,
        covar_module: Optional[ScaleKernel] = None,
        input_transform: Optional[InputTransform] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            datapoints=datapoints,
            comparisons=comparisons,
            likelihood=likelihood,
            covar_module=covar_module,
            input_transform=input_transform,
        )
        # avoid de-dup so hard-coded this to be 0
        self._consolidate_rtol = 0.0
        self._consolidate_atol = 0.0

        # a place-holder in the training stage
        # will load the trained autoencoder for eval
        self.autoencoder = None
        if autoencoder is not None:
            self.set_autoencoder(autoencoder)

    def set_autoencoder(self, autoencoder: nn.Module):
        assert autoencoder.latent_dims == self.covar_module.base_kernel.ard_num_dims
        self.autoencoder = deepcopy(autoencoder)
        self.autoencoder.eval()

    def forward(self, datapoints: Tensor) -> MultivariateNormal:
        return super().forward(datapoints)


def initialize_util_model(
    outcomes: Tensor, comps: Tensor, latent_dims: int
) -> Tuple[HighDimPairwiseGP, PairwiseLaplaceMarginalLogLikelihood]:
    util_model = HighDimPairwiseGP(
        datapoints=outcomes,
        comparisons=comps,
        autoencoder=None,
        input_transform=Normalize(latent_dims),
        covar_module=make_modified_kernel(ard_num_dims=latent_dims),
    )
    mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)
    return util_model, mll_util


def get_fitted_autoencoded_util_model(
    train_Y: Tensor,
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
    latent_dims: int,
    num_joint_train_epochs: int,
    num_autoencoder_pretrain_epochs: int,
    num_unlabeled_outcomes: int,
    outcome_model: Optional[GPyTorchModel] = None,
    bounds: Optional[Tensor] = None,
    autoencoder: Optional[Autoencoder] = None,
    fix_vae: bool = False,
) -> Tuple[HighDimPairwiseGP, Autoencoder]:
    r"""Fit utility model with auto-encoder
    Args:
        train_pref_outcomes: `num_samples x outcome_dim` tensor of outcomes
        train_comps: `num_samples/2 x 2` tensor of pairwise comparisons of Y data
        outcome_model: if not None, we will sample 100 fakes outcomes from it for
            training auto-encoder otherwiese, we use train_Y (labeled preference training data)
    """

    if (
        (outcome_model is not None)
        and (bounds is not None)
        and (num_unlabeled_outcomes > 0)
    ):
        unlabeled_train_Y = _get_unlabeled_outcomes(
            outcome_model=outcome_model,
            bounds=bounds,
            nsample=num_unlabeled_outcomes,
        )
        unlabeled_train_Y = torch.cat((train_Y, unlabeled_train_Y), dim=0)
    else:
        unlabeled_train_Y = train_Y

    if autoencoder is None:
        autoencoder = get_autoencoder(
            train_Y=unlabeled_train_Y,
            latent_dims=latent_dims,
            pre_train_epoch=num_autoencoder_pretrain_epochs,
        )
    # get latent embeddings for train_Y
    z = autoencoder.encoder(train_pref_outcomes).detach()
    util_model, mll_util = initialize_util_model(
        outcomes=z, comps=train_comps, latent_dims=latent_dims
    )

    if not fix_vae:
        # jointly optimize util model and vae parameters
        return jointly_opt_ae_util_model(
            util_model=util_model,
            mll_util=mll_util,
            autoencoder=autoencoder,
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            num_epochs=num_joint_train_epochs,
        )
    else:
        # train util model under fixed vae
        _, util_model_, autoencoder_ = jointly_optimize_models(
            train_Y=train_Y,
            train_pref_outcomes=train_pref_outcomes,
            train_comps=train_comps,
            autoencoder=autoencoder,
            num_epochs=num_joint_train_epochs,
            util_model=util_model,
            mll_util=mll_util,
            train_ae=False,
            train_util_model=True,
            train_outcome_model=False
        )
    
        return util_model_, autoencoder_


def get_fitted_pca_util_model(
    train_Y: Tensor,
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
    pca_var_threshold: float,
    num_unlabeled_outcomes: int,
    outcome_model: Optional[GPyTorchModel] = None,
    bounds: Optional[Tensor] = None,
    standardize: bool = False,
) -> PairwiseGP:
    r"""Fit utility model based on given data and model_kwargs
    Args:
        train_Y: `num_samples x outcome_dim` tensor of outcomes
        train_comps: `num_samples/2 x 2` tensor of pairwise comparisons of Y data
        model_kwargs: input transform and covar_module
        outcome_model: if not None, we will sample 100 fakes outcomes from it for PCA fitting
            otherwise, we use train_Y (labeled preference training data) to fit PCA
    """

    if (
        (outcome_model is not None)
        and (bounds is not None)
        and (num_unlabeled_outcomes > 0)
    ):
        unlabeled_train_Y = _get_unlabeled_outcomes(
            outcome_model=outcome_model,
            bounds=bounds,
            nsample=num_unlabeled_outcomes,
        )
        unlabeled_train_Y = torch.cat((train_Y, unlabeled_train_Y), dim=0)
    else:
        unlabeled_train_Y = train_Y

    projection = fit_pca(
        train_Y=unlabeled_train_Y,
        # need to check the selection of var threshold
        var_threshold=pca_var_threshold,
        weights=None,
        standardize=standardize,
    )

    input_tf = ChainedInputTransform(
        **{
            "projection": LinearProjectionInputTransform(projection),
            "normalize": Normalize(projection.shape[0]),
        }
    )
    covar_module = make_modified_kernel(ard_num_dims=projection.shape[0])
    logger.info(f"pca projection matrix shape: {projection.shape}")
    util_model = PairwiseGP(
        datapoints=train_pref_outcomes,
        comparisons=train_comps,
        input_transform=input_tf,
        covar_module=covar_module,
    )
    mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)
    fit_gpytorch_mll(mll_util)
    return util_model


def get_fitted_standard_util_model(
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
) -> PairwiseGP:
    r"""Fit standard utility model without dim reduction on outcome spaces
    Args:
        train_pref_outcomes: `num_samples x outcome_dim` tensor of outcomes
        train_comps: `num_samples/2 x 2` tensor of pairwise comparisons of Y data
    """
    util_model = PairwiseGP(
        datapoints=train_pref_outcomes,
        comparisons=train_comps,
        input_transform=Normalize(train_pref_outcomes.shape[1]),  # outcome_dim
        covar_module=make_modified_kernel(ard_num_dims=train_pref_outcomes.shape[1]),
    )
    mll_util = PairwiseLaplaceMarginalLogLikelihood(util_model.likelihood, util_model)
    fit_gpytorch_mll(mll_util)
    return util_model


##############################################################################################################
##############################################################################################################
# jointly optimize util model and fine-tune AE

def jointly_opt_ae_util_model(
    util_model: HighDimPairwiseGP,
    mll_util: PairwiseLaplaceMarginalLogLikelihood,
    autoencoder: Autoencoder,
    train_Y: Tensor,
    train_pref_outcomes: Tensor,
    train_comps: Tensor,
    num_epochs: int,
) -> Tuple[HighDimPairwiseGP, Autoencoder]:
    """Jointly optimize util model and fine-tune AE"""
    autoencoder.train()
    util_model.train()

    optimizer = torch.optim.Adam(
        [{"params": autoencoder.parameters()}, {"params": util_model.parameters()}]
    )

    for epoch in range(num_epochs):
        train_Y_recons = autoencoder(train_Y)
        z = autoencoder.encoder(train_pref_outcomes).detach()
        # TODO: weight by the uncertainty
        vae_loss = ((train_Y - train_Y_recons) ** 2).sum() / train_Y.shape[
            -1
        ]  # L2 loss functions
        if epoch % 100 == 0:
            logger.info(
                f"Pref model joint training epoch {epoch}: autoencoder loss func = {vae_loss}"
            )

        # update the util training data (datapoints) with the new latent-embed of outcomes
        util_model.set_train_data(
            comparisons=train_comps, datapoints=z, update_model=True
        )
        pred = util_model(z)
        util_loss = -mll_util(pred, train_comps)
        if epoch % 100 == 0:
            logger.info(
                f"Pref model joint training epoch {epoch}: util model loss func = {util_loss}"
            )
        # add losses and back prop
        loss = vae_loss + util_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    autoencoder.eval()
    util_model.eval()
    # update the trained autoencoder and attach to the util model
    util_model.set_autoencoder(autoencoder=autoencoder)
    return util_model, autoencoder


def jointly_optimize_models(
    autoencoder: Autoencoder,
    train_Y: Tensor,
    num_epochs: int,
    train_X: Optional[Tensor] = None,
    train_pref_outcomes: Optional[Tensor] = None,
    train_comps: Optional[Tensor] = None,
    outcome_model: Optional[SingleTaskGP] = None,
    mll_outcome: Optional[ExactMarginalLogLikelihood] = None,
    util_model: Optional[HighDimPairwiseGP] = None,
    mll_util: Optional[PairwiseLaplaceMarginalLogLikelihood] = None,
    train_ae: bool = True,
    train_outcome_model: bool = True,
    train_util_model: bool = True
):
    
    optimizer_params = []

    if train_ae:
        assert None not in {autoencoder, train_Y}, "Must provide train_Y and autoencoder to train"
        autoencoder.train()
        optimizer_params.append({"params": autoencoder.parameters()})
    if train_outcome_model:
        assert None not in {train_X, train_Y, outcome_model, mll_outcome} , \
            "Must provide train_X, train_Y, outcome model, and mll to train"
        outcome_model.train()
        optimizer_params.append({"params": outcome_model.parameters()})
    if train_util_model:
        assert None not in {train_pref_outcomes, train_comps, util_model, mll_util} , \
            "Must provide train_pref_outcomes, train_comps, util model, and mll to train"
        util_model.train()
        optimizer_params.append({"params": util_model.parameters()})
    
    optimizer = torch.optim.Adam(optimizer_params)

    for epoch in range(num_epochs):
        train_Y_latent = autoencoder.encoder(train_Y).detach() 
        # train_Y_recons = autoencoder.decoder(train_Y_latent)
        train_Y_recons = autoencoder(train_Y)

        loss = 0 

        if train_ae:
            ae_loss = ((train_Y - train_Y_recons) ** 2).sum() / train_Y.shape[-1]
            loss += ae_loss

            if epoch % 100 == 0:
                logger.info(
                    f"Joint training epoch {epoch}: autoencoder loss func = {ae_loss}"
                )

        if train_outcome_model:    
            
            outcome_model.set_train_data(
                inputs=train_X, 
                targets=torch.transpose(train_Y_latent, -2, -1), # this is transpose(train_Y_latent)
                strict=False
            )
            logger.debug(f"train_Y_latent shape: {train_Y_latent.shape}")
            logger.debug(f"train_X shape: {train_X.shape}")
            outcome_pred = outcome_model(train_X)
            logger.debug(f"outcome_pred: {outcome_pred}")
            outcome_loss = -mll_outcome(
                outcome_pred, 
                outcome_model.train_targets # this is transpose(train_Y_latent)
            )
            loss += torch.sum(outcome_loss)

            if epoch % 100 == 0:
                logger.info(
                    f"Joint training epoch {epoch}: outcome model loss func = {outcome_loss}, sum = {torch.sum(outcome_loss)}"
                )

        if train_util_model:
            
            train_pref_outcomes_latent = autoencoder.encoder(train_pref_outcomes).detach() 

            util_model.set_train_data(
                comparisons=train_comps, datapoints=train_pref_outcomes_latent, update_model=True
            )

            util_pred = util_model(train_pref_outcomes_latent)
            util_loss = -mll_util(util_pred, train_comps)
            loss += util_loss

            if epoch % 100 == 0:
                logger.info(
                    f"Joint training epoch {epoch}: util model loss func = {util_loss}"
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if train_ae:
        autoencoder.eval()
    if outcome_model is not None:
        outcome_model.eval()
        # outcome_model.set_autoencoder(autoencoder=autoencoder)
    if util_model is not None:
        util_model.eval()
        util_model.set_autoencoder(autoencoder=autoencoder)
    
    return (outcome_model if train_outcome_model else None,
        util_model if train_util_model else None,
        autoencoder if train_ae else None)


##############################################################################################################
##############################################################################################################
# Other helper funcs

def _get_unlabeled_outcomes(
    outcome_model: GPyTorchModel, bounds: Tensor, nsample: int
) -> Tensor:
    # let's add fake data
    X = (
        draw_sobol_samples(
            bounds=bounds,
            n=1,
            q=2 * nsample,  # fake outcomes be nsample * K outcomes
        )
        .squeeze(0)
        .to(torch.double)
        .detach()
    )
    # sampled from outcomes
    unlabeled_outcomes = outcome_model.posterior(X).rsample().squeeze(0).detach()
    return unlabeled_outcomes

