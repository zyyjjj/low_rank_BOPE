import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('/home/yz685/low_rank_BOPE')
sys.path.append(['..', '../..', '../../..'])

import yaml

from low_rank_BOPE.bope_class import BopeExperiment
from low_rank_BOPE.test_problems.car_problems import problem_setup_augmented

if __name__ == "__main__":

    # read trial_idx from command line input
    trial_idx = int(sys.argv[1])
    # read experiment config from yaml file
    args = yaml.load(open(sys.argv[2]), Loader = yaml.FullLoader)

    print("Experiment args: ", args)

    for problem_setup_name in args["problem_setup_names"]:

        input_dim, outcome_dim, problem, _, util_func, _, _ = problem_setup_augmented(
            problem_setup_name, augmented_dims_noise=args["noise_std"], noisy=True
            # TODO: maybe need seed
        )

        output_path = f"/home/yz685/low_rank_BOPE/experiments/cars/{problem_setup_name}_"\
                        + str(args["noise_std"])+"/"

        experiment = BopeExperiment(
            problem, 
            util_func, 
            methods = args["methods"],
            pe_strategies = args["pe_strategies"],
            trial_idx = trial_idx,
            n_check_post_mean = args["n_check_post_mean"],
            output_path = output_path,
            pca_var_threshold = args["pca_var_threshold"]
        )
        experiment.run_BOPE_loop()
