import matplotlib.pyplot as plt
from typing import Optional, List
import pandas as pd
import os

# ===== Plotting settings =====
# TODO: these need to be updated once we add new methods

colors_dict = {
    "pca": "tab:red", 
    "uwpca": "tab:red",
    "uwpca_rt": "tab:pink",
    "pcr": "tab:cyan", 
    "st": "tab:blue", 
    "true_proj": "tab:pink",
    "random_linear_proj": "tab:green", 
    "random_subset": "tab:orange", 
    "mtgp": "tab:purple", 
    "lmc1": "tab:pink",    
    "lmc2": "tab:brown"
}

linestyle_dict = {"Random-f": "--", "EUBO-zeta": "-"}

marker_dict = {"$EUBO-\zeta$": "o", "True Utility": "s", "Random-f": "^"}

labels_dict = {
    "st": "Indep", 
    "pca": "PCA", 
    "uwpca": "unweighted PCA",
    "uwpca_rt": "unweighted PCA retraining",
    "pcr": "PCR", 
    "random_linear_proj": "Rand-linear-proj", 
    "random_subset": "Rand-subset", 
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


# ===== Functions for plotting =====

def plot_candidate_over_comps_multiple(
    outputs: dict,
    problem_type: str,
    problem_l: List[str], 
    methods: List[str],
    pe_strategy: str, 
    num_plot_datapoints: Optional[int] = None,
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    **kwargs
    ):

    r"""
    Create multiple side-by-side plots of candidate quality over PE comparisons.
    Each plot is for a different problem (specified in problem_l),
    with one pe strategy and multiple methods.

    Args:
        outputs: big nested dictionary storing loaded experiment outputs
        problem_type: one of {"synthetic", "shapes", "cars",}
            this helps the function decide how to parse input/outcome dims from the problem name
        problem_l: list of problem names
        methods: list of methods to show
        pe_strategy: single PE strategy to plot
        num_plot_datapoints: number of checkpoints to show in the plot
        save_path: directory to save the figure, if saving
        save_file_name: file name under save_path to save the figure, if saving
    """

    f, axs = plt.subplots(1, len(problem_l), figsize=(10, 3))

    for j in range(len(problem_l)):
        problem = problem_l[j]

        if problem_type == "synthetic":
            _, rank, _, input_dim, outcome_dim, alpha, noise = problem.split('_')
        elif problem_type == "shapes":
            input_dim = 4
            num_pixels, _ = problem.split("by")
            outcome_dim = int(num_pixels) ** 2
        

        within_session_results = [res 
                                for i in outputs[problem]['within_session_results'].keys() 
                                for res in outputs[problem]["within_session_results"][i]]

        within_df = pd.DataFrame(within_session_results)

        within_df["pe_strategy"] = within_df["pe_strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")
        within_df = (
            within_df.groupby(["n_comps", "method", "pe_strategy"])
            .agg({"util": ["mean", "sem"]}) 
            # TODO: can we generalize the function to allow aggregating over a different quantity 
            .droplevel(level=0, axis=1)
            .reset_index()
        )

        for name, group in within_df.groupby(["method", "pe_strategy"]):
            if name[1] == pe_strategy:
                if name[0] in methods:

                    if num_plot_datapoints is None:
                        num_plot_datapoints = len(group["n_comps"].values)

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
                        f"{problem}\n d={input_dim}, k={outcome_dim}", 
                        fontsize=kwargs.get("title_fontsize", 12.5)
                    )

    axs[0].set_ylabel(
        "True utility of estimated \n utility-maximizing design", 
        fontsize=kwargs.get("ylabel_fontsize", 12)
    )
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


def plot_result_metric_multiple(
    outputs: dict,
    problem_type: str,
    problem_l: List[str], 
    metric: str, 
    ylabel_text: str,
    pe_strategies = ["$EUBO-\zeta$"],
    save_path: Optional[str] = None,
    save_file_name: Optional[str] = None,
    **kwargs
):

    r"""
    Create multiple side-by-side plots of result metrics.
    Each plot is for a different problem (specified in problem_l),
    with one pe strategy and multiple methods.

    Args:
        outputs: big nested dictionary storing loaded experiment outputs
        problem_type: one of {"synthetic", "shapes", "cars",}
            this helps the function decide how to parse input/outcome dims from the problem name
        problem_l: list of problem names
        metric: string specifying the metric to plot
            one of {"candidate_util", "PE_time", "util_model_acc"}
        ylabel_text: English text to put on the y label
            "candidate_util": "Utility of final candidate"
            "PE_time": "Time consumed in preference exploration"
            "util_model_acc": "Final utility model accuracy"
            # TODO: with retraining, some of these are logged multiple times in one trial
        pe_strategies: list of PE strategies (usually I'd just put one)
        save_path: directory to save the figure, if saving
        save_file_name: file name under save_path to save the figure, if saving
    """

    f, axs = plt.subplots(
        1, 
        len(problem_l), 
        figsize=kwargs.get("figsize", (10,3))
    )

    for i in range(len(problem_l)):
        problem = problem_l[i]

        if problem_type == "synthetic":
            _, rank, _, input_dim, outcome_dim, alpha, noise_std = problem.split('_')
        elif problem_type == "shapes":
            input_dim = 4
            num_pixels, _ = problem.split("by")
            outcome_dim = int(num_pixels) ** 2

        available_trials = outputs[problem]["exp_candidate_results"].keys()

        exp_candidate_results = [res for i in available_trials for res in outputs[problem]["exp_candidate_results"][i]]

        exp_candidate_results_random = []
        exp_candidate_results_nonrandom = []

        for res in exp_candidate_results:
            if res["strategy"] == "Random Experiment":
                exp_candidate_results_random.append(res)
            else:
                exp_candidate_results_nonrandom.append(res)

        # Prepare the 2nd experimentation batch data for plot
        exp_df = pd.DataFrame(exp_candidate_results_nonrandom)
        exp_df["strategy"] = exp_df["strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")
        exp_df["strategy"] = pd.Categorical(
            exp_df["strategy"],
            ["True Utility", "$EUBO-\zeta$", "Random-f"],
        )

        exp_df = (
            exp_df.groupby(["method", "strategy"], sort=False)
            .agg({metric: ["mean", "sem"]}) # the quantity we plot is specified in `metric`
            .droplevel(level=0, axis=1)
            .reset_index()
        )

        for name, group in exp_df.groupby(["method", "strategy"]):

            if group["method"].values[0] == 'mtgp':
                continue

            if group["strategy"].values[0] in pe_strategies:

                if not group["mean"].isna().all():
                    axs[i].errorbar(
                        x=[group["method"].values[0] + "_" + group["strategy"].values[0]],
                        y=group["mean"],
                        yerr=kwargs.get("yerr_sems", 1.96) * group["sem"],
                        fmt=marker_dict[name[1]],
                        markersize=8,
                        label=labels_dict[group["method"].values[0]] + "_" + group["strategy"].values[0],
                        linewidth=1.5,
                        capsize=3,
                        color=colors_dict[name[0]],
                    )

        axs[i].set_title(
            f"{problem}\n d={input_dim}, k={outcome_dim}", 
            fontsize=kwargs.get("title_fontsize", 12.5)
        )
        axs[i].set_xticks([])

    axs[0].legend(
        bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (-0.1, -0.2)), 
        loc=kwargs.get("legend_loc", "lower left"), 
        ncol=kwargs.get("legend_ncols", 5), 
        fontsize=kwargs.get("legend_fontsize", 12)
    )
    axs[0].set_ylabel(ylabel_text)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        f.savefig(save_path + save_file_name, bbox_inches = "tight")




# TODO: after the changes for retraining, I probably need to have another function 
# plotting these metrics over addition of datapoints? Need to think this through