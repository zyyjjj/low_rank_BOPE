import sys

import gpytorch
import torch
from botorch.models.model import Model
from low_rank_BOPE.src.pref_learning_helpers import (gen_initial_real_data,
                                                     generate_random_inputs)
from torch import Tensor
from typing import Optional

sys.path.append('..')


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
        torch.eye(outcome_dim) -
        torch.transpose(axes_learned, -2, -1) @ axes_learned
    )

    # (I-VV^T)A
    true_subspace_lost = orth_proj @ torch.transpose(
        ground_truth_principal_axes, -2, -1
    )
    true_subspace_lost_frobenius_norm = torch.linalg.norm(true_subspace_lost)

    # ||A||^2 = latent_dim
    frac_squared_norm = torch.square(
        true_subspace_lost_frobenius_norm) / latent_dim

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
        torch.eye(outcome_dim) -
        torch.transpose(axes_learned, -2, -1) @ axes_learned
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
        torch.eye(outcome_dim) -
        torch.transpose(axes_learned, -2, -1) @ axes_learned
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
    Compute the diagnostic $\mathbb{E}[max_x \|g(f(x)) - g(VV^T f(x))\|_2]$,
    through Monte Carlo sampling. V = axes_learned transposed, where
    each column of V is a learned principal axis. 
    f is the true outcome function and g is the true utility function.

    This quantity is a Monte Carlo estimate of ``expected worst case
    magnitude of utility recovery error", which reflects
    how well we can learn the utility function with the input space being
    outcome space projected onto the subspace V.

    Args:
        problem: a TestProblem, maps inputs to outcomes
        axes_learned: num_axes x outcome_dim tensor,
            each row a learned principal axis
        util_func: ground truth utility function (outcome -> utility)
        n_test: number of test points to estimate the expectation
    Returns:
        expected maximum difference, among the sampled data points, of the
            true utility value and the utility value of the projection
            of sampled outcome data onto the subpace V.
    """

    test_X = generate_random_inputs(problem, n_test).detach()
    test_Y = problem.evaluate_true(test_X).detach()

    # VV^T, projection onto subspace spanned by V, `outcome_dim x outcome_dim`
    proj = torch.transpose(axes_learned, -2, -1) @ axes_learned

    # compute util(Y) - util(VV^T Y)
    test_util_difference = torch.abs(
        util_func(test_Y) - util_func(test_Y @ proj))

    return torch.max(test_util_difference).item()


def compute_variance_explained_per_axis(data, axes, **tkwargs) -> torch.Tensor:
    r"""
    Compute the fraction of variance explained with each axis supplied in `axes`

    Args:
        data: `num_datapoints x output_dim` tensor
        axes: `num_axes x output_dim` tensor where each row is a principal axis

    Returns:
        var_explained: `1 x num_axes` tensor with i-th entry being the fraction 
            of variance explained by the i-th supplied axis
    """

    total_var = sum(torch.var(data, dim=0)).item()

    # check if each row of axes is normalized; if not, divide by L2 norm
    axes = torch.div(axes, torch.linalg.norm(axes, dim=1).unsqueeze(1))

    var_explained = torch.var(
        torch.matmul(data, axes.transpose(0, 1).to(**tkwargs)), dim=0
    ).detach()
    var_explained = var_explained / total_var

    return var_explained


def check_outcome_model_fit(
    outcome_model: Model, 
    problem: torch.nn.Module, 
    n_test: int, 
    batch_eval: bool = True
) -> float:
    r"""
    Evaluate the goodness of fit of the outcome model.
    Args:
        outcome_model: GP model mapping input to outcome
        problem: TestProblem
        n_test: size of test set
    Returns:
        mse: mean squared error between posterior mean and true value
            of the test set observations
    """

    torch.manual_seed(n_test)

    # generate test set
    test_X = generate_random_inputs(problem, n_test).detach()
    if not batch_eval:
        Y_list = []
        for idx in range(len(test_X)):
            y = problem(test_X[idx]).detach()
            Y_list.append(y)
        test_Y = torch.stack(Y_list).squeeze(1)
    else:
        test_Y = problem.evaluate_true(test_X).detach()

    # run outcome model posterior prediction on test data
    test_posterior_mean = outcome_model.posterior(test_X).mean

    # compute relative mean squared error
    mse = ((test_posterior_mean - test_Y)**2 / test_Y**2).mean(axis=0).detach().sum().item()

    se_rel = torch.sum((test_posterior_mean - test_Y) ** 2, dim=1) / torch.sum(test_Y**2, dim=1)
    print(se_rel.shape)
    mse_rel = se_rel.mean(axis=0).item()

    return mse_rel


def check_util_model_fit(
    pref_model: Model, 
    problem: torch.nn.Module, 
    util_func: torch.nn.Module, 
    n_test: int, 
    batch_eval: bool,
    return_util_vals: bool = False,
    projection: Optional[Tensor] = None
) -> float:
    r"""
    Evaluate the goodness of fit of the utility model.
    Args:
        pref_model: GP mapping outcome to utility
        problem: TestProblem
        util_func: ground truth utility function (outcome -> utility)
        n_test: number of outcomes in test set; this gives rise to
            `n_test/2` pairwise comparisons
        projection: optional `latent_dim x outcome_dim` tensor of projection to 
            latent space; if not None, the pref model is fit on the latent space
    Returns:
        pref_prediction_accuracy: fraction of the `n_test/2` pairwise
            preference that the model correctly predicts
    """

    # generate test set
    test_X, test_Y, test_util_vals, test_comps = gen_initial_real_data(
        n=n_test, 
        problem=problem, 
        util_func=util_func, 
        comp_noise=0, 
        batch_eval=batch_eval
    )

    # run pref_model on test data, get predictions
    if projection is not None:
        test_L = torch.matmul(test_Y, torch.transpose(projection, -2, -1))
        posterior_util_mean = pref_model.posterior(test_L).mean
    else:
        posterior_util_mean = pref_model.posterior(test_Y).mean
    posterior_util_mean_ = posterior_util_mean.reshape((n_test // 2, 2))

    # compute pref prediction accuracy
    # the prediction for pair (i, i+1) is correct if
    # item i is preferred to item i+1, so the row in test_comps is [i, i+1]
    # and predicted utility of item i is higher than that of i+1
    # vice versa: [i+1, i] and posterior_util(i) < posterior_util(i+1)
    correct_test_rankings = (posterior_util_mean_[:,0] - posterior_util_mean_[:,1]) * (
        test_comps[:, 0] - test_comps[:, 1]
    )
    pref_prediction_accuracy = sum(correct_test_rankings < 0) / len(
        correct_test_rankings
    )
    print('util model accuracy', pref_prediction_accuracy.item())

    if return_util_vals:
        return test_util_vals, posterior_util_mean, pref_prediction_accuracy.item()
    else:
        return pref_prediction_accuracy.item()


def check_util_model_fit_wrapper(problem, util_func, models_dict, seed = 0, n_test = 1000):
    """ 
    Check the accuracy of preference prediction of the models in `models_dict` 
    on a separate test set. Return the accuracy in a dictionary. 
    """
    torch.manual_seed(seed)
    acc_dict = {}
    for model_key, model in models_dict.items():
        print(f'checking fit of {model_key}')
        acc = check_util_model_fit(
            pref_model = model,
            problem = problem,
            util_func = util_func,
            n_test = n_test,
            batch_eval = True,
            return_util_vals = False
        )
        acc_dict[model_key] = acc
    
    return acc_dict