import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
import math

# ===== Plotting settings =====

colors_dict = {
    "pca": "tab:red", 
    "pca_rt": "tab:pink",
    "pca_norefit_rt": "tab:brown",
    "pca_all_rt": "tab:pink",
    "pca_eubo_rt": "tab:cyan",
    "pca_postmax_rt": "tab:orange",
    "wpca_true_rt": "tab:purple",
    "wpca_est_rt": "tab:orange",
    # "pcr": "tab:cyan", 
    "st": "tab:blue", 
    "true_proj": "tab:pink",
    "random_linear_proj": "tab:green", 
    "random_subset": "tab:orange", 
    "random_search": "tab:cyan",
    "mtgp": "tab:purple", 
    "lmc1": "tab:pink",    
    "lmc2": "tab:brown"
}

linestyle_dict = {"Random-f": "--", "EUBO-zeta": "-"}

marker_dict = {"$EUBO-\zeta$": "o", "True Utility": "s", "Random-f": "^"}

labels_dict = {
    "st": "Indep", 
    "pca": "PCA", 
    "pca_rt": "PCA retraining",
    "pca_norefit_rt": "PCA retraining w/o refitting other model",
    "pca_all_rt": "PCA retraining using all",
    "pca_eubo_rt": "PCA retraining with EUBO winners",
    "pca_postmax_rt": "PCA retraining with posterior max",
    "wpca_true_rt": "weighted PCA using true util values",
    "wpca_est_rt": "weighted PCA using utility posterior mean",
    "pcr": "PCR", 
    "random_linear_proj": "Rand-linear-proj", 
    "random_subset": "Rand-subset", 
    "random_search": "Random search",
    "mtgp": "MTGP", 
    "lmc1": "LMC1", "lmc2": "LMC2",
    "true_proj": "True-proj"
}

# horizontal jitter to add to different methods   
x_jitter_dict = {
    "pca": 0.1, 
    "st": 0, 
    "random_linear_proj": 0.2, 
    "random_subset": 0.3, 
    "mtgp": 0.75, 
    "lmc1": 0.4, 
    "lmc2": 0.5, 
    "pcr": 0.05,
    "uwpca": 0.1, 
    "uwpca_rt": 0.2,
}

#################################################################################
#################################################################################

# ===== Plotting results in PE stage =====

# "within_session_results" -- results logged during PE stage

performance_over_comps_labels_dict = {
    "util": "True utility of estimated \n utility-maximizing design",
    "util_model_acc": "Rank accuracy of util model",
    "util_model_acc_top_half": "Rank accuracy of util model on top 50% test points",
    "util_model_acc_top_quarter": "Rank accuracy of util model on top 25% test points",
    "overall_model_acc": "Overall accuracy of outcome and utility models",
}
    
def plot_performance_over_comps_single(
    outputs: dict,
    problem: str,
    methods: List[str],
    pe_strategy: str, 
    problem_type: Optional[str] = None,
    metric: str = "util",
    shade: bool = True,
    num_plot_datapoints: Optional[int] = None,
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    include_legend: bool = True,
    **kwargs
):
    r"""
    Create a plot of performance evolution over PE comparisons for one problem,
    with one pe strategy and multiple methods.

    Args:
        outputs: big nested dictionary storing loaded experiment outputs
        problem: problem name
        methods: list of methods to show
        pe_strategy: single PE strategy to plot
        problem_type: Optional, one of {"synthetic", "shapes", "cars",}
            this helps the function decide how to parse input/outcome dims from the problem name,
            though one could pass in `input_dim` and `outcome_dim` directly
        metric: the quantity to plot, one of {"util", "util_model_acc"}
        shade: whether to plot error bars as shaded regions or not
        num_plot_datapoints: number of checkpoints to show in the plot
        save_path: directory to save the figure, if saving
        save_file_name: file name under save_path to save the figure, if saving
        include_legend: whether to include a legend in the plot
    """
    f, axs = plt.subplots(1, 1, 
                          figsize=kwargs.get("figsize",(6, 4))
                          )
    

    input_dim = kwargs.get("input_dim", None)
    outcome_dim = kwargs.get("outcome_dim", None)
    if input_dim is None or outcome_dim is None:
        if problem_type == "synthetic":
            _, rank, _, input_dim, outcome_dim, alpha, noise = problem.split('_')
        elif problem_type == "shapes":
            input_dim = 4
            num_pixels, _ = problem.split("by")
            outcome_dim = int(num_pixels) ** 2
        elif problem_type == "music":
            input_dim = 12
            outcome_dim = 441 # TODO: don't hardcode
        else:
            raise RuntimeError("Input and outcome dims not specified!")
    
    within_session_results = [res 
                              for i in outputs[problem]['within_session_results'].keys() 
                              for res in outputs[problem]["within_session_results"][i]]

    within_df = pd.DataFrame(within_session_results)

    within_df["pe_strategy"] = within_df["pe_strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")

    within_df = (
        within_df.groupby(["n_comps", "method", "pe_strategy"])
        .agg({metric: ["mean", "sem"]})
        .droplevel(level=0, axis=1)
        .reset_index()
    )

    for name, group in within_df.groupby(["method", "pe_strategy"]):
        if name[1] == pe_strategy:
            if name[0] in methods:

                if num_plot_datapoints is None:
                    num_plot_datapoints = len(group["n_comps"].values)

                
                # print(name[0], name[1], 'n_comps: ', group["n_comps"].values)

                if shade: 
                    axs.plot(
                        # x_jittered[:num_plot_datapoints],
                        group["n_comps"].values,
                        group["mean"].values[:num_plot_datapoints],
                        label=labels_dict[name[0]],
                        color=colors_dict[name[0]],
                    )
                    axs.fill_between(
                        # x=x_jittered[:num_plot_datapoints],
                        group["n_comps"].values,
                        y1=group["mean"].values[:num_plot_datapoints] \
                            - group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                        y2=group["mean"].values[:num_plot_datapoints] \
                            + group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                        alpha=kwargs.get("alpha", 0.2),
                        color=colors_dict[name[0]],
                    )
                else:
                    jitter = x_jitter_dict[group["method"].values[0]]
                    x_jittered = [x_ + jitter for x_ in group["n_comps"].values]
                    
                    axs.errorbar(
                        x=x_jittered[:num_plot_datapoints],
                        y=group["mean"].values[:num_plot_datapoints],
                        yerr=group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                        label=labels_dict[name[0]],
                        linewidth=1.5,
                        capsize=3,
                        alpha=0.6,
                        color=colors_dict[name[0]],
                    )

    axs.set_xlabel("Number of comparisons")
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    title = kwargs.get("title", f"{problem}\n d={input_dim}, k={outcome_dim}")
    axs.set_title(
        title, 
        fontsize=kwargs.get("title_fontsize", 12.5)
    )
    
    ylabel = kwargs.get("ylabel", performance_over_comps_labels_dict[metric])
    axs.set_ylabel(ylabel, fontsize=kwargs.get("ylabel_fontsize", 12))

    if include_legend:
        axs.legend(
            bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (-0.05, -0.3)), 
            loc=kwargs.get("legend_loc", "lower left"), 
            ncol=kwargs.get("legend_ncols", 5), 
            fontsize=kwargs.get("legend_fontsize", 12)
        )

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        f.savefig(save_path + save_file_name, bbox_inches = "tight")


def plot_performance_over_comps_multiple(
    outputs: dict,
    problem_type: str, # TODO: enable passing in a list of input_dim and outcome_dim
    problem_l: List[str], 
    methods: List[str],
    pe_strategy: str, 
    metric: str = "util",
    shade: bool = True,
    num_plot_datapoints: Optional[int] = None,
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    include_legend: bool = True,
    **kwargs
    ):

    r"""
    Create multiple side-by-side plots of candidate quality over PE comparisons.
    Each plot is for a different problem (specified in problem_l),
    with one pe strategy and multiple methods.

    Args:
        problem_l: list of problem names
        All the others are the same as plot_performance_over_comps_single()
    """

    f, axs = plt.subplots(
        1, len(problem_l), 
        figsize=kwargs.get("figsize", (10,3))
    )

    for j in range(len(problem_l)):
        problem = problem_l[j]

        if problem_type == "synthetic":
            _, rank, _, input_dim, outcome_dim, alpha, noise = problem.split('_')
            problem_name = problem # TODO: come back to this
        elif problem_type == "shapes":
            input_dim = 4
            num_pixels, _ = problem.split("by")
            outcome_dim = int(num_pixels) ** 2
            problem_name = problem # TODO: come back to this
        elif problem_type == "cars":
            problem_name, iodim, util_type, outcome_dim, _ = problem.split("_")
            input_dim = int(iodim[0])
            outcome_dim = int(outcome_dim)
            problem_name = problem_name + "_" + util_type
        
        within_session_results = [res 
                                for i in outputs[problem]['within_session_results'].keys() 
                                for res in outputs[problem]["within_session_results"][i]]

        within_df = pd.DataFrame(within_session_results)

        within_df["pe_strategy"] = within_df["pe_strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")
        within_df = (
            within_df.groupby(["n_comps", "method", "pe_strategy"])
            .agg({metric: ["mean", "sem"]}) 
            .droplevel(level=0, axis=1)
            .reset_index()
        )

        for name, group in within_df.groupby(["method", "pe_strategy"]):
            if name[1] == pe_strategy:
                if name[0] in methods:

                    if num_plot_datapoints is None:
                        num_plot_datapoints = len(group["n_comps"].values)

                    
                    if shade:
                        axs[j].plot(
                            # x_jittered[:num_plot_datapoints],
                            group["n_comps"].values,
                            group["mean"].values[:num_plot_datapoints],
                            label=labels_dict[name[0]],
                            color=colors_dict[name[0]],
                        )
                        axs[j].fill_between(
                            # x=x_jittered[:num_plot_datapoints],
                            group["n_comps"].values,
                            y1=group["mean"].values[:num_plot_datapoints]-group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                            y2=group["mean"].values[:num_plot_datapoints]+group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                            alpha=kwargs.get("alpha", 0.2),
                            color=colors_dict[name[0]],
                        )

                    else:
                        jitter = x_jitter_dict[group["method"].values[0]]
                        x_jittered = [x_ + jitter for x_ in group["n_comps"].values]
                        axs[j].errorbar(
                            x=x_jittered[:num_plot_datapoints],
                            y=group["mean"].values[:num_plot_datapoints],
                            yerr = group["sem"][:num_plot_datapoints] * kwargs.get("yerr_sems", 1.96),
                            label=labels_dict[name[0]],
                            linewidth=1.5,
                            capsize=3,
                            alpha=0.6,
                            color=colors_dict[name[0]],
                        )

                    axs[j].set_xlabel(
                        "Number of comparisons", 
                        fontsize=kwargs.get("xlabel_fontsize", 12)
                    )
                    axs[j].set_title(
                        f"{problem_name}\n d={input_dim}, k={outcome_dim}", 
                        fontsize=kwargs.get("title_fontsize", 12.5)
                    )
    ylabel = kwargs.get("ylabel", performance_over_comps_labels_dict[metric])
    axs[0].set_ylabel(ylabel, fontsize=kwargs.get("ylabel_fontsize", 12))

    if include_legend:
        axs[0].legend(
            bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (-0.05, -0.4)), 
            loc=kwargs.get("legend_loc", "lower left"), 
            ncol=kwargs.get("legend_ncols", 5), 
            fontsize=kwargs.get("legend_fontsize", 12)
        )

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        f.savefig(save_path + save_file_name, bbox_inches = "tight")

#################################################################################
#################################################################################
# ===== "subspace_diagnostics" -- diagnostics of subspace quality logged during PE stage

def tensor_list_to_float_list(tensor_list):
    res = []
    for item in tensor_list:
        res.append(item.item())
    
    return res

subspace_diagnostics_labels_dict = {
    "best_util": "Best utility in subspace",
    "avg_util": "Average utility in subspace",
    "model_fitting_time": "Time to fit outcome model",
    "rel_mse": "Frac variance unexplained of outcome model",
    "max_util_error": "Max util error in subspace",
    "max_outcome_error": "Max outcome error in subspace",
    "latent_dim": "subspace dimension",
    "grassmannian": "Grassmannian distance",
}


def plot_subspace_diagnostics_single(
    outputs: dict,
    problem: str,
    methods: List[str] = ["uwpca", "uwpca_rt"],
    pe_strategy: str = "EUBO-zeta",
    metric: str = "best_util",
    shade: bool = True,
    plot_vline: Optional[bool] = False,
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    include_legend: bool = True,
    **kwargs
):
    r"""
    Plot the evolution of `metric` in iterative retraining of subspace methods

    Args:
        outputs: big nested dictionary storing loaded experiment outputs
        problem: problem name
        methods: list of methods to show
        pe_strategy: single PE strategy to plot
        metric: the quantity of interest, 
            one of {"best_util", "model_fitting_time", "rel_mse", "max_util_error", "max_outcome_error"}
        shade: whether to plot error bars as shaded regions or not
        plot_vline: whether to plot a vertical line,
            this is used to separate the PE and BO stages, if metrics are logged together for both
        save_path: directory to save the figure, if saving
        save_file_name: file name under save_path to save the figure, if saving
        include_legend: whether to include legend in the plot
    """

    available_trials = list(outputs[problem]["subspace_diagnostics"].keys())

    found_valid_trial = False
    try_trial = 0
    while not found_valid_trial:
        try:
            num_retrain_and_boiters = len(outputs[problem]["subspace_diagnostics"][available_trials[try_trial]][("pca_rt", pe_strategy)][metric])
            found_valid_trial = True
        except:
            try_trial += 1

    print("num_retrain_and_boiters: ", num_retrain_and_boiters)

    # num_retrain_and_boiters = len(outputs[problem]["subspace_diagnostics"][available_trials[0]][("pca_all_rt", pe_strategy)][metric])

    print("available trials: ", available_trials)

    num_retrain = None

    for method in methods:

        print(f"Plotting for method {method}")

        # a nested list where each sub-list is a record of metric over retraining for one trial     
        data = []

        for trial_idx in available_trials:
            try:
                tmp = outputs[problem]["subspace_diagnostics"][trial_idx][(method, pe_strategy)][metric]
                if torch.is_tensor(tmp[0]): # in case I accidentally saved a 1-element tensor instead of a float
                    data.append(tensor_list_to_float_list(tmp))
                else:
                    data.append(tmp)
                longest = max(longest, len(tmp))
            except:
                continue

        # for the methods whose record length is shorter, repeat the first element to match the length of the longest one
        # TODO: handle the case where data is empty (why?)
        if len(data[0]) < num_retrain_and_boiters:
            data_np = np.array(data)
            data_np_head = np.repeat(data_np[:, 0:1], num_retrain_and_boiters-len(data[0]), axis=1)
            data_np = np.concatenate((data_np_head, data_np), axis=1)
            if num_retrain is None:
                num_retrain = num_retrain_and_boiters-len(data[0]) 
        else:
            data_np = np.array(data)
        print(data_np.shape)
        mean = np.array(data_np).mean(axis=0)
        sem = np.std(data_np, axis=0, ddof=1) / np.sqrt(len(available_trials))
        
        if shade:
            plt.plot(mean, color=colors_dict[method], label=labels_dict[method])
            plt.fill_between(
                x=range(len(mean)), 
                y1=mean-sem*kwargs.get("yerr_sems", 1), 
                y2=mean+sem*kwargs.get("yerr_sems", 1), 
                alpha=0.4, 
                color=colors_dict[method]
            )
        else:
            plt.errorbar(
                x=range(len(mean)),
                y=mean,
                yerr=sem*kwargs.get("yerr_sems", 1),
                color=colors_dict[method],
                label=labels_dict[method],
                linewidth=1.5,
                capsize=3,
                alpha=0.6,
            )

    if plot_vline:
        plt.vlines(
            x = num_retrain, 
            ymin = kwargs.get("vline_ymin", 0), 
            ymax = kwargs.get("vline_ymax", 1),
            color = "black", 
            alpha = 0.5,
            linestyles = "dashed", 
            linewidth = 1.5
        )

    if include_legend:
        plt.legend(
            bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (-0.05, -0.3)), 
            loc=kwargs.get("legend_loc", "lower left"), 
            ncol=kwargs.get("legend_ncols", 5), 
            fontsize=kwargs.get("legend_fontsize", 12)
        )

    plt.xlabel("Number of times retrained")

    y_label = kwargs.get("y_label", subspace_diagnostics_labels_dict[metric])
    plt.ylabel(y_label)

    title = kwargs.get("title", f"{problem},\n {y_label} over retraining")
    plt.title(title)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(save_path + save_file_name, bbox_inches = "tight")


# TODO: add plot_subspace_diagnostics_multiple()

#################################################################################
#################################################################################

# ===== Plotting final results =====
# "exp_candidate_results" -- results logged in the 2nd experimentation stage

BO_results_labels_dict = {
    "candidate_util": "True utility of BO candidate",
    "acqf_val": "qNEIUU acquisition function value",
    "util_model_acc": "Utility model accuracy",
    "best_util_so_far": "Best utility so far",
}

def plot_BO_results_single(
    outputs: dict,
    problem: str,
    methods: List[str],
    pe_strategy: str, 
    problem_type: Optional[str] = None,
    metric: str = "candidate_util",
    shade: bool = True,
    num_plot_datapoints: Optional[int] = None,
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    include_legend: bool = True,
    **kwargs
):
    r"""
    Create a plot of BO performance in the second experimentation stage 
    for one problem, one pe strategy and multiple methods.

    Args:
        outputs: big nested dictionary storing loaded experiment outputs
        problem: problem name
        methods: list of methods to show
        pe_strategy: single PE strategy to plot
        problem_type: Optional, one of {"synthetic", "shapes", "cars",}
            this helps the function decide how to parse input/outcome dims from the problem name,
            though one could pass in `input_dim` and `outcome_dim` directly
        metric: the quantity to plot, one of {"candidate_util", "acqf_val", "util_model_acc"}
        shade: whether to plot error bars as shaded regions or not
        num_plot_datapoints: number of BO iterations to plot
        save_path: directory to save the figure, if saving
        save_file_name: file name under save_path to save the figure, if saving
        include_legend: whether to include a legend in the plot
    """

    f, axs = plt.subplots(1, 1, figsize=kwargs.get("figsize",(6, 4)))

    input_dim = kwargs.get("input_dim", None)
    outcome_dim = kwargs.get("outcome_dim", None)
    if input_dim is None or outcome_dim is None:
        if problem_type == "synthetic":
            _, rank, _, input_dim, outcome_dim, alpha, noise = problem.split('_')
        elif problem_type == "shapes":
            input_dim = 4
            num_pixels, _ = problem.split("by")
            outcome_dim = int(num_pixels) ** 2
        elif problem_type == "music":
            input_dim = 12
            outcome_dim = 441 # TODO: don't hardcode
        else:
            raise RuntimeError("Input and outcome dims not specified!")
    
    BO_results = [res 
                    for i in outputs[problem]['exp_candidate_results'].keys() 
                    for res in outputs[problem]["exp_candidate_results"][i]]
    
    print(f"{len(outputs[problem]['exp_candidate_results'].keys())} valid trials: ",  outputs[problem]['exp_candidate_results'].keys())

    BO_results_df = pd.DataFrame(BO_results)

    BO_results_df["strategy"] = BO_results_df["strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")

    BO_results_df = (
        BO_results_df.groupby(["BO_iter", "method", "strategy"])
        .agg({metric: ["mean", "sem"]})
        .droplevel(level=0, axis=1)
        .reset_index()
    )

    for name, group in BO_results_df.groupby(["method", "strategy"]):
        if name[1] == pe_strategy:
            if name[0] in methods:

                if num_plot_datapoints is None:
                    num_plot_datapoints = len(group["mean"].values)

                if shade: 
                    axs.plot(
                        # x_jittered[:num_plot_datapoints],
                        group["BO_iter"].values[:num_plot_datapoints],
                        group["mean"].values[:num_plot_datapoints],
                        label=labels_dict[name[0]],
                        color=colors_dict[name[0]],
                    )
                    axs.fill_between(
                        # x=x_jittered[:num_plot_datapoints],
                        group["BO_iter"].values[:num_plot_datapoints],
                        y1=group["mean"].values[:num_plot_datapoints] \
                            - group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                        y2=group["mean"].values[:num_plot_datapoints] \
                            + group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                        alpha=kwargs.get("alpha", 0.2),
                        color=colors_dict[name[0]],
                    )
                else:
                    jitter = x_jitter_dict[group["method"].values[0]]
                    x_jittered = [x_ + jitter for x_ in group["BO_iters"].values]
                    
                    axs.errorbar(
                        x=x_jittered[:num_plot_datapoints],
                        y=group["mean"].values[:num_plot_datapoints],
                        yerr=group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                        label=labels_dict[name[0]],
                        linewidth=1.5,
                        capsize=3,
                        alpha=0.6,
                        color=colors_dict[name[0]],
                    )
    
    title = kwargs.get("title", f"{problem}\n d={input_dim}, k={outcome_dim}")
    axs.set_title(
        title,
        fontsize=kwargs.get("title_fontsize", 12.5)
    )
   
    axs.set_xlabel("Number of BO iterations")
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    ylabel = kwargs.get("ylabel", BO_results_labels_dict[metric])
    axs.set_ylabel(ylabel)

    if include_legend:
        axs.legend(
            bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (-0.05, -0.3)), 
            loc=kwargs.get("legend_loc", "lower left"), 
            ncol=kwargs.get("legend_ncols", 5), 
            fontsize=kwargs.get("legend_fontsize", 12)
        )

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        f.savefig(save_path + save_file_name, bbox_inches = "tight")


def plot_BO_results_multiple(
    outputs: dict,
    problem_type: str,  # TODO: enable passing in a list of input_dim and outcome_dim
    problem_l: List[str],
    methods: List[str],
    pe_strategy: str, 
    metric: str = "candidate_util",
    input_outcome_dims: Optional[List[tuple[int]]] = None,
    shade: bool = True,
    num_plot_datapoints: Optional[int] = None,
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    include_legend: bool = True,
    **kwargs
):
    r"""
    Create multiple side-by-side plots of BO performance in the second experimentation stage.
    Each plot is for a different problem (specified in problem_l),
    with one pe strategy and multiple methods.

    Args:
        problem_l: list of problem names
        All the others are the same as plot_BO_results_single()
    """

    f, axs = plt.subplots(1, len(problem_l), figsize=kwargs.get("figsize",(10,3)))

    input_outcome_dims = kwargs.get("input_outcome_dims", None)

    for j in range(len(problem_l)):

        problem = problem_l[j]

        if input_outcome_dims is None:
            if problem_type == "synthetic":
                _, rank, _, input_dim, outcome_dim, alpha, noise = problem.split('_')
            elif problem_type == "shapes":
                input_dim = 4
                num_pixels, _ = problem.split("by")
                outcome_dim = int(num_pixels) ** 2
            elif problem_type == "music":
                input_dim = 12
                outcome_dim = 441 # TODO: don't hardcode
            elif problem_type == "cars":
                problem_name, iodim, util_type, outcome_dim, _ = problem.split("_")
                input_dim = int(iodim[0])
                outcome_dim = int(outcome_dim)
            else:
                raise RuntimeError("Input and outcome dims not specified!")
        else:
            input_dim, outcome_dim = input_outcome_dims[j]
        
        BO_results = [res 
                        for i in outputs[problem]['exp_candidate_results'].keys() 
                        for res in outputs[problem]["exp_candidate_results"][i]]

        BO_results_df = pd.DataFrame(BO_results)

        BO_results_df["strategy"] = BO_results_df["strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")

        BO_results_df = (
            BO_results_df.groupby(["BO_iter", "method", "strategy"])
            .agg({metric: ["mean", "sem"]})
            .droplevel(level=0, axis=1)
            .reset_index()
        )

        for name, group in BO_results_df.groupby(["method", "strategy"]):
            if name[1] == pe_strategy:
                if name[0] in methods:

                    if num_plot_datapoints is None:
                        num_plot_datapoints = len(group["BO_iter"].values)

                    if shade: 
                        axs[j].plot(
                            # x_jittered[:num_plot_datapoints],
                            group["BO_iter"].values,
                            group["mean"].values[:num_plot_datapoints],
                            label=labels_dict[name[0]],
                            color=colors_dict[name[0]],
                        )
                        axs[j].fill_between(
                            # x=x_jittered[:num_plot_datapoints],
                            group["BO_iter"].values,
                            y1=group["mean"].values[:num_plot_datapoints] \
                                - group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                            y2=group["mean"].values[:num_plot_datapoints] \
                                + group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                            alpha=kwargs.get("alpha", 0.2),
                            color=colors_dict[name[0]],
                        )
                    else:
                        jitter = x_jitter_dict[group["method"].values[0]]
                        x_jittered = [x_ + jitter for x_ in group["BO_iters"].values]
                        
                        axs[j].errorbar(
                            x=x_jittered[:num_plot_datapoints],
                            y=group["mean"].values[:num_plot_datapoints],
                            yerr=group["sem"][:num_plot_datapoints]*kwargs.get("yerr_sems", 1.96),
                            label=labels_dict[name[0]],
                            linewidth=1.5,
                            capsize=3,
                            alpha=0.6,
                            color=colors_dict[name[0]],
                        )
        
                    axs[j].set_title(
                        f"{problem}\n d={input_dim}, k={outcome_dim}",
                        fontsize=kwargs.get("title_fontsize", 12.5)
                    )
                
                    axs[j].set_xlabel("Number of BO iterations")
                    axs[j].xaxis.set_major_locator(MaxNLocator(integer=True))

    ylabel = kwargs.get("ylabel", BO_results_labels_dict[metric])
    axs[0].set_ylabel(ylabel)

    if include_legend:
        axs[0].legend(
            bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (-0.05, -0.3)), 
            loc=kwargs.get("legend_loc", "lower left"), 
            ncol=kwargs.get("legend_ncols", 5), 
            fontsize=kwargs.get("legend_fontsize", 12)
        )

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        f.savefig(save_path + save_file_name, bbox_inches = "tight")
