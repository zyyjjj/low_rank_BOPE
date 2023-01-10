from low_rank_BOPE.experiments.bope_class import BopeExperiment
from low_rank_BOPE.src.real_problems import problem_setup_augmented

problem_setup_names = [
    "vehiclesafety_5d3d_piecewiselinear_3c",
    "carcabdesign_7d9d_piecewiselinear_3c",
    "carcabdesign_7d9d_linear_3c",
    # "osy_6d8d_piecewiselinear_3c",
]

for problem_setup_name in problem_setup_names:

    input_dim, outcome_dim, problem, _, util_func, _, _ = problem_setup_augmented(
        problem_setup_name, augmented_dims_noise=0.01
        # TODO: maybe need seed
    )

    BopeExperiment(
        problem, 
        util_func, 
        methods = ["st", "pca"],
        pe_strategies = ["EUBO-zeta", "Random-f"],
        trial_idx = 1,
        save_dir = "/home/yz685/low_rank_BOPE/experiments/reults/car_problems/"
    )


# data storage?