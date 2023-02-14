import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import argparse

import torch
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


def parse():
    # experiment-running params -- read from command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_idx", type = int, default = 0)
    parser.add_argument("--n_pixels", type = int, default = 16)
    parser.add_argument("--n_check_post_mean", type = int, default = 13)
    parser.add_argument("--methods", type = str, nargs = "+", 
        default = ["st", "pca", "pcr"])
    parser.add_argument("--pe_strategies", type = str, nargs = "+", 
        default = ["EUBO-zeta"])

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    print("Parsed args: ", args)

    run_pipeline(
        trial_idx = args.trial_idx,
        n_pixels=args.n_pixels,
        n_check_post_mean = args.n_check_post_mean, 
        methods=args.methods, 
        pe_strategies=args.pe_strategies
    )

    # TODO: can I replace absolute path with script directory, like
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # what's the difference between this and 
    # file_dir = os.path.dirname(__file__) ??
    # if output_path is None:
        # output_path = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp_output"
        # )
