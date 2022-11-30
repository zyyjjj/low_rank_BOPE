# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from typing import Any, Dict, List, Optional, Union

import torch
from botorch.exceptions import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, HalfCauchyPrior, NormalPrior
from torch import Tensor
from torch.distributions.half_cauchy import HalfCauchy
from torch.nn import Parameter


class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: Optional[float] = None):
        self._tau = tau

    def tau(self, m):
        return (
            self._tau
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscale_prior_param_or_closure(self, m):
        return self.tau(m) / (m.lengthscale**2)

    def inv_lengthscale_prior_setting_closure(self, m, value):
        lb = m.raw_lengthscale_constraint.lower_bound
        ub = m.raw_lengthscale_constraint.upper_bound
        m._set_lengthscale((self.tau(m) / value).sqrt().clamp(lb, ub))

    def tau_prior_param_or_closure(self, m):
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m, value):
        lb = m.raw_tau_constraint.lower_bound
        ub = m.raw_tau_constraint.upper_bound
        m.raw_tau.data.fill_(
            m.raw_tau_constraint.inverse_transform(value.clamp(lb, ub)).item()
        )


def add_saas_prior(
    base_kernel: Kernel, tau: Optional[float] = None, **tkwargs
) -> Kernel:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is given by tau / lengthscale^2 ~ HC(1.0). If tau is None,
    we place an additional HC(0.1) prior on tau similar to the original SAAS prior
    that relies on inference with NUTS.

    Args:
        base_kernel: Base kernel that has a lengthscale and uses ARD.
            Note that this function modifies the kernel object in place.
        tau: Value of the global shrinkage. If `None`, infer the global
            shrinkage parameter.

    Returns:
        Base kernel with SAAS priors added.

    Example:
        >>> matern_kernel = MaternKernel(...)
        >>> add_saas_prior(matern_kernel, tau=None)  # Add a SAAS prior
    """
    if not base_kernel.has_lengthscale:
        raise UnsupportedError("base_kernel must have lengthscale(s)")
    if hasattr(base_kernel, "lengthscale_prior"):
        raise UnsupportedError("base_kernel must not specify a lengthscale prior")
    # TODO: Appears sensitive to the initial value, worth thinking about why.
    base_kernel.register_constraint(
        param_name="raw_lengthscale",
        constraint=Interval(0.01, 1e4, initial_value=1.0),
        replace=True,
    )
    prior_helper = SaasPriorHelper(tau=tau)
    if tau is None:  # Place a HC(0.1) prior on tau
        base_kernel.register_parameter(
            name="raw_tau", parameter=Parameter(torch.tensor(0.1, **tkwargs))
        )
        base_kernel.register_constraint(
            param_name="raw_tau",
            constraint=Interval(1e-3, 10, initial_value=0.1),
            replace=True,
        )
        base_kernel.register_prior(
            name="tau_prior",
            prior=HalfCauchyPrior(torch.tensor(0.1, **tkwargs)),
            param_or_closure=prior_helper.tau_prior_param_or_closure,
            setting_closure=prior_helper.tau_prior_setting_closure,
        )
    # Place a HC(1) prior on tau / lengthscale^2
    base_kernel.register_prior(
        name="inv_lengthscale_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscale_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscale_prior_setting_closure,
    )
    return base_kernel


def _get_map_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    tau: Optional[float] = None,
) -> Union[FixedNoiseGP, SingleTaskGP]:
    """Helper method for creating an unfitted MAP SAAS model."""
    # TODO: Shape checks
    tkwargs = {"device": train_X.device, "dtype": train_X.dtype}
    mean_module = ConstantMean(constant_prior=NormalPrior(loc=0.0, scale=1.0))
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])
    add_saas_prior(base_kernel=base_kernel, tau=tau, **tkwargs)
    covar_module = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_prior=GammaPrior(
            torch.tensor(2.0, **tkwargs), torch.tensor(0.15, **tkwargs)
        ),
    )
    if train_Yvar is not None:
        return FixedNoiseGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(
            torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)
        ),
        noise_constraint=GreaterThan(1e-6),
    )
    return SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        mean_module=mean_module,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )


def get_fitted_map_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    tau: Optional[float] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[FixedNoiseGP, SingleTaskGP]:
    """Get a fitted MAP SAAS model with a Matern kernel.

    Args:
        train_X: Tensor of shape `n x d` with training inputs.
        train_Y: Tensor of shape `n x 1` with training targets.
        train_Yvar: Optional tensor of shape `n x 1` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transforms.
        tau: Fixed value of the global shrinkage tau. If None, the model
            places a HC(0.1) prior on tau.
        optimizer_kwargs: A dict of options for the optimizer passed
            to fit_gpytorch_mll.

    Returns:
        A fitted SingleTaskGP with a Matern kernel.
    """

    # make sure optimizer_kwargs is a Dict
    optimizer_kwargs = optimizer_kwargs or {}

    model = _get_map_saas_model(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        input_transform=input_transform.train()
        if input_transform is not None
        else None,
        outcome_transform=outcome_transform,
        tau=tau,
    )
    mll = ExactMarginalLogLikelihood(model=model, likelihood=model.likelihood)
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model


def get_fitted_map_saas_ensemble(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Optional[Tensor] = None,
    input_transform: Optional[InputTransform] = None,
    outcome_transform: Optional[OutcomeTransform] = None,
    taus: Optional[Union[Tensor, List[float]]] = None,
    num_taus: int = 4,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> SaasFullyBayesianSingleTaskGP:
    """Get a fitted SAAS ensemble using several different tau values.

    Args:
        train_X: Tensor of shape `n x d` with training inputs.
        train_Y: Tensor of shape `n x 1` with training targets.
        train_Yvar: Optional tensor of shape `n x 1` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transforms.
        taus: Global shrinkage values to use. If None, we sample `num_taus` values
            from an HC(0.1) distrbution.
        num_taus: Optional argument for how many taus to sample.
        optimizer_kwargs: A dict of options for the optimizer passed
            to fit_gpytorch_mll.

    Returns:
        A fitted SaasFullyBayesianSingleTaskGP with a Matern kernel.
    """
    tkwargs = {"device": train_X.device, "dtype": train_X.dtype}
    if taus is None:
        taus = HalfCauchy(0.1).sample([num_taus]).to(**tkwargs)
    num_samples = len(taus)
    if num_samples == 1:
        raise ValueError(
            "Use `get_fitted_map_saas_model` if you only specify one value of tau"
        )

    mean = torch.zeros(num_samples, **tkwargs)
    outputscale = torch.zeros(num_samples, **tkwargs)
    lengthscale = torch.zeros(num_samples, train_X.shape[-1], **tkwargs)
    noise = torch.zeros(num_samples, **tkwargs)

    # Fit a model for each tau and save the hyperparameters
    for i, tau in enumerate(taus):
        model = get_fitted_map_saas_model(
            train_X,
            train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            tau=tau,
            optimizer_kwargs=optimizer_kwargs,
        )
        mean[i] = model.mean_module.constant.detach().clone()
        outputscale[i] = model.covar_module.outputscale.detach().clone()
        lengthscale[i, :] = model.covar_module.base_kernel.lengthscale.detach().clone()
        if train_Yvar is None:
            noise[i] = model.likelihood.noise.detach().clone()

    # Load the samples into a fully Bayesian SAAS model
    ensemble_model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        input_transform=input_transform.train()
        if input_transform is not None
        else None,
        outcome_transform=outcome_transform,
    )
    mcmc_samples = {
        "mean": mean,
        "outputscale": outputscale,
        "lengthscale": lengthscale,
    }
    if train_Yvar is None:
        mcmc_samples["noise"] = noise
    ensemble_model.train()
    ensemble_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
    ensemble_model.eval()
    return ensemble_model


def get_and_fit_map_saas_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    num_taus: int = 4,
    taus: Optional[List[float]] = None,
    **kwargs: Any,
) -> ModelListGP:
    if taus is None and num_taus == 1:
        taus = HalfCauchy(0.1).sample([1]).tolist()
    if not refit_model:  # Load from state dict
        if taus is not None and len(taus) == 1:
            models = [
                _get_map_saas_model(
                    train_X=X,
                    train_Y=Y,
                    train_Yvar=None if Yvar.isnan().any() else Yvar,
                    tau=taus[0],
                )
                for X, Y, Yvar in zip(Xs, Ys, Yvars)
            ]
        else:
            # TODO: Remove when `SaasFullyBayesianSingleTaskGP` has added support
            # for `load_state_dict`.
            models = [
                _get_fully_bayesian_model(
                    X=X,
                    Y=Y,
                    Yvar=None if Yvar.isnan().any() else Yvar,
                    num_mcmc_samples=len(taus) if taus is not None else num_taus,
                )
                for X, Y, Yvar in zip(Xs, Ys, Yvars)
            ]
        model = ModelListGP(*models)
        model.to(Xs[0])
        model.load_state_dict(state_dict)  # pyre-ignore
        return model

    # Fit from scratch
    if taus is not None and len(taus) == 1:
        models = [
            get_fitted_map_saas_model(
                train_X=X,
                train_Y=Y,
                train_Yvar=None if Yvar.isnan().any() else Yvar,
                tau=taus[0],
            )
            for X, Y, Yvar in zip(Xs, Ys, Yvars)
        ]
    else:
        models = [
            get_fitted_map_saas_ensemble(
                train_X=X,
                train_Y=Y,
                train_Yvar=None if Yvar.isnan().any() else Yvar,
                taus=taus,
                num_taus=num_taus,
            )
            for X, Y, Yvar in zip(Xs, Ys, Yvars)
        ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model
