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
from low_rank_BOPE.test_problems.shapes import (AreaUtil, Bars,
                                                GradientAwareAreaUtil, Image,
                                                LargestRectangleUtil)


def run_pipeline(
    trial_idx, n_pixels, 
    outcome_func_name,
    util_func_name,
    retrain,
    methods = ["st", "pca", "pcr"],
    pe_strategies = ["EUBO-zeta", "Random-f"],
    **kwargs):

    torch.manual_seed(trial_idx)
    kwargs.update({"standardize": False}) # don't standardize for shapes due to numerical issues

    if outcome_func_name == "rectangle":
        problem = Image(num_pixels=n_pixels)
    elif outcome_func_name == "bars":
        problem = Bars(num_pixels=n_pixels)

    if util_func_name == "area":   
        util_func = AreaUtil(binarize=binarize_area)
    elif util_func_name == "gradient_aware_area":
        penalty_param = kwargs.get('penalty_param', 0.5)
        util_func = GradientAwareAreaUtil(
            penalty_param=penalty_param, 
            image_shape=(n_pixels, n_pixels)
        )
    elif util_func_name == "max_rectangle":
        util_func = LargestRectangleUtil(image_shape=(n_pixels, n_pixels))

    print("methods to plug into BopeExperiment: ", methods)

    if retrain:
        experiment_class = RetrainingBopeExperiment
        output_path = "/home/yz685/low_rank_BOPE/experiments/shapes_rt/" + \
            f"{n_pixels}by{n_pixels}_{outcome_func_name}_{util_func_name}/"
    else:
        experiment_class = BopeExperiment
        output_path = "/home/yz685/low_rank_BOPE/experiments/shapes/" + \
            f"{n_pixels}by{n_pixels}_{outcome_func_name}_{util_func_name}/"

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
        util_func_name = args["util_func_name"],
        outcome_func_name = args["outcome_func_name"],
        retrain = args["retrain"],
        n_check_post_mean = args["n_check_post_mean"], 
        methods = args["methods"], 
        pe_strategies = args["pe_strategies"],
        pca_var_threshold = args["pca_var_threshold"],
        initial_experimentation_batch = args["init_exp_batch"],
        penalty_param = args["penalty_param"],
        binarize_area = args.get("binarize", True)
    )