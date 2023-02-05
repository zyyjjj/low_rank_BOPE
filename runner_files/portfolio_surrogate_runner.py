import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append("/home/yz685/low_rank_BOPE")
sys.path.append(["..", "../..", "../../.."])

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.portfolio_opt_surrogate.portfolio_surrogate import (
    DistributionalPortfolioSurrogate, RiskMeasureUtil)

if __name__ == "__main__":

    trial_idx = int(sys.argv[1])
    n_w_samples = int(sys.argv[2])

    problem = DistributionalPortfolioSurrogate(negate=True, n_w_samples=n_w_samples)
    util_func = RiskMeasureUtil(util_func_key="mean_plus_sd", lambdaa=0.8)
    problem_name = f"portfolio_w=uniform_{n_w_samples}_util=mean_plus_sd_0.8"

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "experiments",
        problem_name,
        "/"
    )

    experiment = BopeExperiment(
        problem,
        util_func,
        methods=["st", "pca", "pcr", "true_proj"],
        pe_strategies=["EUBO-zeta", "Random-f"],
        trial_idx=trial_idx,
        output_path=output_path
    )
    experiment.run_BOPE_loop()
    