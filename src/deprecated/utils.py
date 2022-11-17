from typing import Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from ae.pca.models import (
    get_posterior_predictions_PCA,
    get_posterior_predictions_ST,
    initialize_model_PCA,
    initialize_model_ST,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood


def scatter_plot_w_diag_line(data1: torch.Tensor, data2: torch.Tensor, metric_idx: int):
    r"""Create a scatter plot with a diagonal line
    comparing the `metric_idx`- th component of data1 and data2.

    Args:
        data1: `num_samples x data_dims` tensor
        data2: `num_samples x data_dims` tensor
        metric_idx: integer specifying the index of the metric to compare

    Returns:
        a scatter plot of data1[:,metric_idx] against data2[:,metric_idx]
        also prints mean squared error
    """

    plt.scatter(data1[:, metric_idx], data2[:, metric_idx])

    axis_lb = min(min(data1[:, metric_idx]), min(data2[:, metric_idx]))
    axis_ub = max(max(data2[:, metric_idx]), max(data2[:, metric_idx]))

    mse = ((data1[:, metric_idx] - data2[:, metric_idx]) ** 2).mean(axis=0)
    print("mean squared error = {}".format(mse))

    plt.plot([axis_lb, axis_ub], [axis_lb, axis_ub], "k--", linewidth=1)


def compute_cosine_sim(
    vec1: torch.Tensor, vec2: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    r"""Compute the cosine similarity between two vectors along `dim` dimension.
    (Typical use case is to compare `num_axes x metric_dim`-shaped ground truth and learned principal axes.)

    Args:
        vec1: `m x n` tensor
        vec2: `m x n` tensor
        dim: integer specifying the dimension along which to compute the cosine similarity

    Returns:
        cosine similarity along `dim` components of vec1 and vec2
        shape is 1 x m if dim = 1; 1 x n if dim = 0.
    """

    cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)

    return cos(vec1, vec2)


def plot_metric_posterior(
    test_x: torch.Tensor,
    metric_dim: int,
    method: str,
    model: Type[Model],
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    axes: torch.Tensor = None,
    **tkwargs,
):
    r"""
    Plot posterior distribution for a specified metric. Assumes input dimension is 1.

    Args:
        test_x: `num_test_points`-dimensional tensor
        metric_dim: integer specifying the index of the metric we want to plot
        method: one of {"PCA", "ST"}
        model: fitted GP model
        train_X: `num_samples x input_dim` tensor of sampled input points
        train_Y: `num_samples x output_dim` tensor where each row is an observed metric datapoint
        axes: `num_axes x output_dim` tensor where rows are fitted principal axes

    Returns:
        Plot of posterior mean and 95% credible interval of the `metric_dim`-th metric
    """

    if method == "PCA":
        (
            PC_posterior_mean,
            PC_posterior_variance,
            metric_posterior_mean,
            metric_posterior_variance,
        ) = get_posterior_predictions_PCA(test_x.unsqueeze(1), model, axes, **tkwargs)
    elif method == "ST":
        metric_posterior_mean, metric_posterior_variance = get_posterior_predictions_ST(
            test_x.unsqueeze(1), model, **tkwargs
        )

    lower = metric_posterior_mean - 1.96 * torch.sqrt(metric_posterior_variance)
    upper = metric_posterior_mean + 1.96 * torch.sqrt(metric_posterior_variance)

    plt.plot(train_X.detach().numpy(), train_Y.detach().numpy()[:, metric_dim], "k*")
    plt.plot(
        test_x.detach().numpy(),
        metric_posterior_mean.detach().numpy()[:, metric_dim],
        "b",
        linewidth=1,
    )
    plt.fill_between(
        test_x.detach().numpy(),
        lower.detach().numpy()[:, metric_dim],
        upper.detach().numpy()[:, metric_dim],
        alpha=0.5,
    )
    plt.legend(["Observed Data", "Mean", "95% Credible Interval"])
    plt.tight_layout()
    plt.title("Metric {} posterior distribution ({})".format(metric_dim, method))


def loocv(
    X_all: torch.Tensor,
    Y_all: torch.Tensor,
    method: str,
    infer_noise: bool = True,
    Y_var_all: torch.Tensor or float = None,
    variance_explained_threshold: float = None,
    **tkwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Perform leave-one-out cross validation for a given dataset of inputs and observations and a specified method.
    For each sample, hold it out and fit a GP model on the rest of the data.
    Then, use the fitted model to predict the posterior mean and variance of the held-out sample.

    Args:
        X_all: `num_samples x input_dim` tensor
        Y_all: `num_samples x output_dim` tensor
        method: one of {"PCA", "ST"}
        infer_noise: matters for 'ST' method; whether to infer noise in SingleTaskGP() or use supplied Y_var_all to fit a FixedNoiseGP()
        Y_var_all: either `num_samples x output_dim` tensor or float for observation variance # TODO: only been using float now
        variance_explained_threshold: number between 0 and 1 specifying the amount of variance in the data
            that we want the principal axes to explain

    Returns:
        predicted_Y_means: `num_samples x output_dim` tensor where the i-th row is the predicted posterior mean for the i-th sample.
        predicted_Y_vars: `num_samples x output_dim` tensor where the i-th row is the predicted posterior variance for the i-th sample.
    """

    print(
        "Full input data shape {}, output data shape {}".format(
            X_all.shape, Y_all.shape
        )
    )

    predicted_Y_means = torch.Tensor()
    predicted_Y_vars = torch.Tensor()

    for i in range(X_all.shape[0]):
        if i % 5 == 0:
            print("loocv round {}".format(i))

        X_train = torch.cat((X_all[:i], X_all[i + 1 :]))
        Y_train = torch.cat((Y_all[:i], Y_all[i + 1 :]))

        Y_train_raw_mean = torch.mean(Y_train, dim=0)
        Y_train -= Y_train_raw_mean

        if Y_var_all is not None:
            if torch.is_tensor(Y_var_all):
                Y_var_train = torch.cat((Y_var_all[:i], Y_var_all[i + 1 :]))
            else:
                Y_var_train = Y_var_all
        else:
            Y_var_train = None

        X_test = X_all[i].unsqueeze(0)

        if method == "PCA":
            model, axes_learned = initialize_model_PCA(
                X_train, Y_train, variance_explained_threshold
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            (
                _,
                _,
                Y_posterior_mean,
                Y_posterior_variance,
            ) = get_posterior_predictions_PCA(X_test, model, axes_learned, **tkwargs)

        elif method == "ST":
            model = initialize_model_ST(
                X_train, Y_train, Y_var_train, infer_noise, **tkwargs
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            Y_posterior_mean, Y_posterior_variance = get_posterior_predictions_ST(
                X_test, model
            )

        # shift by the original mean of Y_train
        Y_posterior_mean += Y_train_raw_mean

        # use loo-fitted model to compute the posterior mean on held-out datapoint
        predicted_Y_means = torch.cat((predicted_Y_means, Y_posterior_mean))
        predicted_Y_vars = torch.cat((predicted_Y_vars, Y_posterior_variance))

    return predicted_Y_means, predicted_Y_vars


def plot_metric_loocv_results(
    observed_Y: torch.Tensor,
    posterior_Y_mean: torch.Tensor,
    posterior_Y_var: torch.Tensor,
    metric_idx: int,
    Y_var_all: torch.Tensor or float = None,
):
    r"""
    Plot leave-one-out cross validation results for a metric given a set of observations and posterior predictions

    Args:
        observed_Y: `num_samples x output_dim` tensor of observations
        posterior_Y_mean: `num_samples x output_dim` tensor where the i-th row is the predicted posterior mean for the i-th sample
        posterior_Y_var: `num_samples x output_dim` tensor where the i-th row is the predicted posterior variance for the i-th sample
        metric_idx: index of the metric for which we want to plot loocv results
        Y_var_all: either `num_samples x output_dim` tensor or float of observation variance (not SD) # TODO: only been using float now
    Returns:
        plot LOOCV results for the `metric_idx`-th metric
    """

    fig, ax_ = plt.subplots(nrows=1, ncols=1)

    ax_.scatter(
        observed_Y.detach().numpy()[:, metric_idx],
        posterior_Y_mean.detach().numpy()[:, metric_idx],
    )
    axis_lb = min(
        min(observed_Y.detach().numpy()[:, metric_idx]),
        min(posterior_Y_mean.detach().numpy()[:, metric_idx]),
    )
    axis_ub = max(
        max(observed_Y.detach().numpy()[:, metric_idx]),
        max(posterior_Y_mean.detach().numpy()[:, metric_idx]),
    )

    ax_.plot([axis_lb, axis_ub], [axis_lb, axis_ub], "k--", linewidth=0.5)

    ax_.set_xlabel("Observed metric {}".format(metric_idx))
    ax_.set_ylabel("Predicted metric {}".format(metric_idx))
    ax_.set_title(f"Metric {metric_idx}")

    # error bars along the y-axis, i.e., posterior standard deviation
    ax_.errorbar(
        observed_Y.detach().numpy()[:, metric_idx],
        posterior_Y_mean.detach().numpy()[:, metric_idx],
        yerr=2 * torch.sqrt(posterior_Y_var)[:, metric_idx],
        ls="none",
        linewidth=0.5,
    )

    # error bars along the x-axis, i.e., observation noise
    if Y_var_all is not None:
        if torch.is_tensor(Y_var_all):
            ax_.errorbar(
                observed_Y.detach().numpy()[:, metric_idx],
                posterior_Y_mean.detach().numpy()[:, metric_idx],
                xerr=2 * torch.sqrt(Y_var_all).detach().numpy()[:, metric_idx],
                ls="none",
                linewidth=0.5,
            )
        else:
            ax_.errorbar(
                observed_Y.detach().numpy()[:, metric_idx],
                posterior_Y_mean.detach().numpy()[:, metric_idx],
                xerr=2 * np.sqrt(Y_var_all),
                ls="none",
                linewidth=0.5,
            )

    fig.tight_layout()


def plot_BO_performance(
    all_method_names: List[str],
    all_result_dicts: List[Dict],
    time_on_x_axis: bool = True,
    CI_width: float = 1,
):
    r"""
    Plot BO performance of supplied methods and trials

    Args:
        all_method_names: list of strings specifying method name
        all_result_dicts: list of dictionaries logging optimization performance over iterations for the methods in all_method_names
        time_on_x_axis: whether to plot performance progression over cumulative iteration time; if False, plot over BO iterations
        CI_width: number of standard errors we want the confidence interval to be

    Returns:
        Plot of BO performance of the supplied methods and trials
    """

    assert len(all_result_dicts) == len(
        all_method_names
    ), "list of method names should have the same length as the list of result dicts"

    for method_idx in range(len(all_method_names)):

        method_name = all_method_names[method_idx]
        result_dicts = all_result_dicts[method_idx]
        num_trials = len(result_dicts)

        # compute incremental average of time spent over iterations
        # compute incremental average and sample variance of best value achieved over iterations
        # reference: Welford's algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
        iter_times_inc_mean = np.zeros(len(result_dicts[0]["iteration_times"]))
        objective_inc_mean = np.zeros(len(result_dicts[0]["best_vals"]))
        objective_inc_var = np.zeros(len(result_dicts[0]["best_vals"]))
        objective_inc_mse = np.zeros(len(result_dicts[0]["best_vals"]))

        for i in range(num_trials):
            objective_inc_mean_old = objective_inc_mean
            iter_times_inc_mean = iter_times_inc_mean + (
                np.array(np.cumsum(result_dicts[i]["iteration_times"]))
                - iter_times_inc_mean
            ) / (i + 1)
            objective_inc_mean = objective_inc_mean + (
                np.array(result_dicts[i]["best_vals"]) - objective_inc_mean
            ) / (i + 1)
            if i > 0:
                objective_inc_mse = objective_inc_mse + (
                    np.array(result_dicts[i]["best_vals"]) - objective_inc_mean_old
                ) * (np.array(result_dicts[i]["best_vals"]) - objective_inc_mean)
                objective_inc_var = objective_inc_mse / i

        print(
            f"method {method_name} over {num_trials} trials: final mean objective {objective_inc_mean[-1]}, time spent {iter_times_inc_mean[-1]} sec"
        )

        iter_times_inc_mean = np.insert(iter_times_inc_mean, 0, 0)

        if time_on_x_axis:
            plt.plot(iter_times_inc_mean, objective_inc_mean, label=method_name)
            plt.fill_between(
                iter_times_inc_mean,
                objective_inc_mean - CI_width * np.sqrt(objective_inc_var),
                objective_inc_mean + CI_width * np.sqrt(objective_inc_var),
                alpha=0.3,
            )
        else:
            plt.plot(objective_inc_mean, label=method_name)
            plt.fill_between(
                np.arange(len(objective_inc_mean)),
                objective_inc_mean - CI_width * np.sqrt(objective_inc_var),
                objective_inc_mean + CI_width * np.sqrt(objective_inc_var),
                alpha=0.3,
            )

    plt.title("Optimization performance")
    if time_on_x_axis:
        plt.xlabel("Cumulative time spent (seconds)")
    else:
        plt.xlabel("Number of function evaluations")
    plt.ylabel("Best objective value achieved")
    plt.legend()
