import copy
import random
import time
from typing import Any, Dict, List, NamedTuple

import fblearner.flow.api as flow
import gpytorch
import numpy as np
import torch
from make_problem import make_problem_and_util_func
from problem_setups import PROBLEM_SETUPS


class OneRun(NamedTuple):
    exp_candidate_results: List[Dict[str, Any]]
    within_session_results: List[Dict[str, Any]]
    # TODO: more results?
    # TODO: confirm return type, key could be Tuple[str]



"""
List of problem names:

[
    "vehiclesafety_5d3d_piecewiselinear_24", 
    "carcabdesign_7d9d_piecewiselinear_72", 
    "carcabdesign_7d9d_linear_72",
    "8by8_rectangle_gradientAwareArea",
    "16by16_rectangle_gradientAwareArea",
    "robot_3_100_1",
    "robot_3_500_5",
    "inventory_100",
    "PTS=6_input=1_outcome=45_latent=3_PCls=0.5_pwlinear_seed=1234_alpha=0.0",
    "PTS=6_input=1_outcome=45_latent=3_PCls=0.5_pwlinear_seed=1234_alpha=0.5",
    "PTS=6_input=1_outcome=45_latent=3_PCls=0.5_pwlinear_seed=1234_alpha=1.0",
]

"""



# workflow OPTION 1
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    problem_list: List[str],
    baselines:  List[str],
    trial_range: List[int],
) -> Dict[str, List[OneRun]]:
    
    all_results = {}
    
    # for problem_name in problem_list:
    #    problem, util_func = make_problem_and_util_func(problem_name)
    #    for baseline in baselines:
    #       all_results[f"{problem_name}_{baseline}_{trial_idx}"] = \
    #            [run_one_trial(problem, util_func, [baseline], trial_idx) \
    #               for for trial_idx in trial_range]

    return all_results


# workflow OPTION 2
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    problem_name: str,
    baselines:  List[str],
    trial_range: List[int],
) -> Dict[str, List[OneRun]]:
    
    all_results = {}

    # problem, util_func = make_problem_and_util_func(problem_name, options = PROBLEM_SETUPS[problem_name])
    # for baseline in baselines:
    #    all_results[f"{problem_name}_{baseline}_{trial_idx}"] = \
    #            [run_one_trial(problem, util_func, [baseline], trial_idx) \
    #               for for trial_idx in trial_range]
    
    return all_results

# TODO: how do I allow different trials indices for different problems? 

@flow.flow_async()
@flow.typed()
def run_one_trial(
    problem: torch.nn.Module,
    util_func: torch.nn.Module,
    methods: List[str],
    trial_idx: int,
    experiment_config: Dict[str, Any],
) -> OneRun:

    # create experiment class with specified args
    # call exp.run_BOPE_loop()
    # return exp results

    return OneRun(
        exp_candidate_results=exp_candidate_results,
        within_session_results=within_session_results,
    )