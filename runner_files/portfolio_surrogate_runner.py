import os, sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])
from low_rank_BOPE.test_problems.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.portfolio_opt_surrogate import PortfolioSurrogate


if __name__ == "__main__":

    # experiment-running params -- read from command line input
    trial_idx = int(sys.argv[1])

    # # TODO: modify below to run portfolio problem

    # problem_setup_names = [
    #     "vehiclesafety_5d3d_piecewiselinear_3c",
    #     "carcabdesign_7d9d_piecewiselinear_3c",
    #     "carcabdesign_7d9d_linear_3c",
    # ]

    # for problem_setup_name in problem_setup_names:

    #     input_dim, outcome_dim, problem, _, util_func, _, _ = problem_setup_augmented(
    #         problem_setup_name, augmented_dims_noise=0.01
    #         # TODO: maybe need seed
    #     )

    problem = PortfolioSurrogate(negate=True)
    # TODO; util function for portfolio problems!
    # utility should be a function of the return distribution over env variables
    # can start from: mean + lambda * standard deviation 
    # next TODO: figure out how to sample these env RV's -- or does BoRisk repo have it already?
    # first three dims are designs, last two are environmental variables



    experiment = BopeExperiment(
        problem, 
        util_func, 
        methods = ["st", "pca", "pcr"],
        pe_strategies = [
            "EUBO-zeta", 
            "Random-f"
        ],
        trial_idx = trial_idx,
        output_path = "/home/yz685/low_rank_BOPE/experiments/"+problem_setup_name+"/"
    )
    experiment.run_BOPE_loop()

    # TODO: can I replace absolute path with script directory, like
    # script_dir = os.path.dirname(os.path.abspath(__file__))
