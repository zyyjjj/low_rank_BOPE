import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])
import argparse

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.car_problems import problem_setup_augmented


def parse():
    # experiment-running params -- read from command line input
    parser = argparse.ArgumentParser()

    parser.add_argument("--trial_idx", type = int, default = 0)
    parser.add_argument("--noise_std", type = float, default = 0.01)
    parser.add_argument("--n_check_post_mean", type = int, default = 13)

    return parser.parse_args()

if __name__ == "__main__":

    # experiment-running params -- read from command line input
    args = parse()

    print("Experiment args: ", args)

    problem_setup_names = [
        "vehiclesafety_5d3d_piecewiselinear_5c",
        "carcabdesign_7d9d_piecewiselinear_5c",
        "carcabdesign_7d9d_linear_5c",
    ]

    for problem_setup_name in problem_setup_names:

        input_dim, outcome_dim, problem, _, util_func, _, _ = problem_setup_augmented(
            problem_setup_name, augmented_dims_noise=args.noise_std, noisy=True
            # TODO: maybe need seed
        )

        output_path = f"/home/yz685/low_rank_BOPE/experiments/cars/{problem_setup_name}_{args.noise_std}/"

        experiment = BopeExperiment(
            problem, 
            util_func, 
            methods = ["st", "pca", "pcr"],
            pe_strategies = [
                "EUBO-zeta", 
                "Random-f"
            ],
            trial_idx = args.trial_idx,
            n_check_post_mean = args.n_check_post_mean,
            output_path = output_path
        )
        experiment.run_BOPE_loop()
