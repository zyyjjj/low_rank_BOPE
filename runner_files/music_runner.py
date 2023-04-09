import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import torch
import yaml

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.bope_class_retraining import RetrainingBopeExperiment
from low_rank_BOPE.test_problems.music.music import HarmonyOneKey, Consonance, get_model_spectra, DISSONANCE

def run_pipeline(
    trial_idx, n_pixels, 
    # outcome_func_name,
    # util_func_name,
    retrain,
    methods = ["st", "pca", "pcr"],
    pe_strategies = ["EUBO-zeta", "Random-f"],
    spectrum_truncation_length: int = 20,
    **kwargs):

    torch.manual_seed(trial_idx)
    kwargs.update({"standardize": False}) # don't standardize for music due to numerical issues

    print("methods to plug into BopeExperiment: ", methods)

    problem = HarmonyOneKey()

    try:
        model_spectra = torch.load('/home/yz685/low_rank_BOPE/low_rank_BOPE/test_problems/music/model_spectra_{trunc_length}.pt')
    except FileNotFoundError:
        model_spectra = get_model_spectra(trunc_length=spectrum_truncation_length)

    util_func = Consonance(
        model_spectra = model_spectra,
        dissonance_vals = list(DISSONANCE.values())
    )

    if retrain:
        experiment_class = RetrainingBopeExperiment
        output_path = "/home/yz685/low_rank_BOPE/experiments/music/" + \
            f"onekey_{spectrum_truncation_length}/"
    else:
        experiment_class = BopeExperiment
        output_path = "/home/yz685/low_rank_BOPE/experiments/music/" + \
            f"onekey_{spectrum_truncation_length}/"

    experiment = experiment_class(
        problem, 
        util_func, 
        methods = methods,
        pe_strategies = pe_strategies,
        trial_idx = trial_idx,
        output_path = output_path,
        **kwargs
    )
    experiment.run_BOPE_loop()


if __name__ == "__main__":

    # read trial_idx from command line input
    trial_idx = int(sys.argv[1])
    # read experiment config from yaml file
    args = yaml.load(open(sys.argv[2]), Loader = yaml.FullLoader)

    print("Experiment args: ", args)

    run_pipeline(
        trial_idx = trial_idx,
        n_pixels=args["n_pixels"],
        # util_func_name = args["util_func_name"],
        # outcome_func_name = args["outcome_func_name"],
        retrain = args["retrain"],
        n_check_post_mean = args["n_check_post_mean"], 
        methods = args["methods"], 
        pe_strategies = args["pe_strategies"],
        spectrum_truncation_length = args["spectrum_truncation_length"],
        pca_var_threshold = args["pca_var_threshold"],
        initial_experimentation_batch = args["init_exp_batch"],
    )