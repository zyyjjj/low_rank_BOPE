import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append("/home/yz685/low_rank_BOPE")
sys.path.append(["..", "../..", "../../.."])

import argparse

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.portfolio_opt_surrogate.portfolio_surrogate import (
    DistributionalPortfolioSurrogate, RiskMeasureUtil)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trial_idx", type = int, default = 0)
    parser.add_argument("--n_w_samples", type = int, default = 50)
    parser.add_argument("--lambdaa", type = float, default = 0.5)
    parser.add_argument("--n_check_post_mean", type = int, default = 13)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    problem = DistributionalPortfolioSurrogate(
        negate=True, 
        n_w_samples=args.n_w_samples
    )
    util_func = RiskMeasureUtil(
        util_func_key="mean_plus_sd", 
        lambdaa=args.lambdaa)
    problem_name = f"portfolio_w=uniform_{args.n_w_samples}_util=mean_plus_sd_{args.lambdaa}"

    output_path = "/home/yz685/low_rank_BOPE/experiments/portfolio/" + problem_name + "/"

    experiment = BopeExperiment(
        problem,
        util_func,
        methods=["st", "pca", "pcr"],
        pe_strategies=["EUBO-zeta", "Random-f"],
        trial_idx=args.trial_idx,
        output_path=output_path,
        n_check_post_mean=args.n_check_post_mean
    )
    experiment.run_BOPE_loop()
    