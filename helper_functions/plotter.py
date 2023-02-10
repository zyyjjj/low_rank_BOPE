"""
Generate plots for completed experiments
"""

import os, sys
sys.path.append("/home/yz685/low_rank_BOPE/")
import itertools
import torch
import numpy as np
import pandas as pd
from matplotlib.pyplot import plt  
from matplotlib.ticker import MaxNLocator

from collections import defaultdict
from typing import Dict, List


def extract_data(problem_names, trials_range):

    outputs = defaultdict(lambda: defaultdict(dict))

    for problem in problem_names:

        results_folder = f'/home/yz685/low_rank_BOPE/experiments/{problem}/'

        for trial in trials_range:

            try:

                outputs[problem]['exp_candidate_results'][trial] = \
                    list(vv for v in torch.load(
                        results_folder + f'final_candidate_results_trial={trial}.th'
                        ).values() for vv in v.values())
                    
                outputs[problem]['within_session_results'][trial] = \
                    list(itertools.chain.from_iterable(
                        vv for v in torch.load(
                            results_folder + f'PE_session_results_trial={trial}.th'
                            ).values() for vv in v.values()))
            
            except FileNotFoundError:
                continue
        
    return outputs

colors_dict = {
    "pca": "tab:red", 
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
    "st": "Indep", "pca": "PCA", "pcr": "PCR", "random_linear_proj": "Rand-linear-proj", 
    "random_subset": "Rand-subset", "mtgp": "MTGP", "lmc1": "LMC1", "lmc2": "LMC2",
    "true_proj": "True-proj"
}


def plot_candidate_over_comps(
    problem: str, 
    problems_dict: str, 
    outputs: Dict, 
    pe_strategy: str, 
    methods: List[str],
    save_fig_path: str,
):
    r"""
    Plot true utility of posterior mean utility maximizing design over PE comps.
    Args:
        problem: 
        problems_dict:
        outputs:
        pe_strategy:
        methods:
        save_fig_path:
    """

    # TODO: improve the naming of variables
    # TODO: add docs

    f, axs = plt.subplots(1, 1, figsize=(8, 6))

    x_jitter_dict = {
        "pca": 0.1, 
        "st": 0, 
        "random_linear_proj": 0.2, "random_subset": 0.3, 
        "mtgp": 0.75, "lmc1": 0.4, "lmc2": 0.5, 
        "pcr": 0.05, "true_proj": 0.15}
    
    input_dim, outcome_dim = problems_dict[problem]

    within_session_results = [res 
                              for i in outputs[problem]['within_session_results'].keys() 
                              for res in outputs[problem]["within_session_results"][i]]

    within_df = pd.DataFrame(within_session_results)

    within_df["pe_strategy"] = within_df["pe_strategy"].str.replace("EUBO-zeta", r"$EUBO-\\zeta$")
    within_df = (
        within_df.groupby(["n_comps", "method", "pe_strategy"])
        .agg({"util": ["mean", "sem"]})
        .droplevel(level=0, axis=1)
        .reset_index()
    )

    for name, group in within_df.groupby(["method", "pe_strategy"]):
        if name[1] == pe_strategy:
            if name[0] in methods:

                jitter = x_jitter_dict[group["method"].values[0]]
                x_jittered = [x_ + jitter for x_ in group["n_comps"].values]
                print(name[0], name[1], 'n_comps: ', group["n_comps"].values)

                axs.errorbar(
                    x=x_jittered,
                    y=group["mean"].values,
                    yerr=1.96 * group["sem"],
                    # label="_".join(name),
                    label=labels_dict[name[0]],
                    linewidth=1.5,
                    capsize=3,
                    alpha=0.6,
                    color=colors_dict[name[0]],
                )

                axs.set_xlabel("Number of comparisons")
                axs.set_title(
                    f"{problem}\n d={input_dim}, k={outcome_dim}", fontsize=16
                )
                axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    axs.set_ylabel("True utility of estimated \n utility-maximizing design")
    axs.legend(loc="lower left", ncol=5, fontsize=15)

    # TODO: save under /plots/
    plt.savefig(save_fig_path + problem + ".pdf")



if __name__ == "__main__":


    # TODO: update this
    # then move to a separate file "plotter_synthetic.py"
    problems = {
        "rank_1_linear_1_20_0.2_0.1": (1, 20),
        "rank_2_linear_1_20_0.2_0.1": (1, 20),
        "rank_4_linear_1_20_0.2_0.1": (1, 20),
    }

    outputs = extract_data(problems, range(21))

    save_fig_path = "/home/yz685/low_rank_BOPE/plots/synthetic/"

    plot_candidate_over_comps(
        problem = "rank_1_linear_1_20_0.2_0.1", 
        problems_dict = problems, 
        outputs = outputs, 
        pe_strategy = "$EUBO-\zeta$", 
        methods = ["st", "pca", "pcr", "true_proj"],
        save_fig_path = save_fig_path
    )