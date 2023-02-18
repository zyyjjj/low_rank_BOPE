import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import torch
import yaml

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.shapes import AreaUtil, Image


def run_pipeline(
    trial_idx, n_pixels, 
    methods = ["st", "pca", "pcr"],
    pe_strategies = ["EUBO-zeta", "Random-f"],
    **kwargs):

    torch.manual_seed(trial_idx)

    problem = Image(num_pixels=n_pixels)
    util_func = AreaUtil()

    output_path = "/home/yz685/low_rank_BOPE/experiments/shapes/" + \
        f"{n_pixels}by{n_pixels}/"

    print("methods to plug into BopeExperiment: ", methods)

    experiment = BopeExperiment(
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
        n_check_post_mean = args["n_check_post_mean"], 
        methods=args["methods"], 
        pe_strategies=args["pe_strategies"],
        pca_var_threshold = args["pca_var_threshold"],
        initial_experimentation_batch = args["init_exp_batch"],
    )