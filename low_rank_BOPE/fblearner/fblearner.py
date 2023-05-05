import copy
import random
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import fblearner.flow.api as flow
import gpytorch
import numpy as np
import torch
from make_problem import make_problem_and_util_func
from problem_setups import PROBLEM_SETUPS, EXPERIMENT_SETUPS

from low_rank_BOPE.bope_class_retraining import RetrainingBopeExperiment

class OneRun(NamedTuple):
    exp_candidate_results: Dict[str, Dict[str, Any]]
    PE_session_results: Dict[str, Dict[str, Any]]
    pref_data_dict: Dict[Tuple[str], Any]
    subspace_diagnostics: Dict[Tuple[str], Any]
    BO_data_dict: Dict[Tuple[str], Any]


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

# As an example, running "8by8_rectangle_gradientAwareArea"
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    problem_name: str = "8by8_rectangle_gradientAwareArea", 
    baselines:  List[str] = ["pca"],
    trial_range: List[int] = list(range(10)),
) -> Dict[str, List[OneRun]]:
    
    problem, util_func = make_problem_and_util_func(
        problem_name, options = PROBLEM_SETUPS[problem_name])
    
    for baseline in baselines:
        for trial_idx in trial_range:
            run_one_trial(
                problem=problem, 
                util_func=util_func, 
                methods=[baseline], 
                trial_idx=trial_idx,
                experiment_options=EXPERIMENT_SETUPS[problem_name]
            )
    

@flow.flow_async()
@flow.typed()
def run_one_trial(
    problem: torch.nn.Module,
    util_func: torch.nn.Module,
    methods: List[str],
    trial_idx: int,
    experiment_options: Dict[str, Any],
) -> OneRun:

    # create experiment class with specified setup
    exp = RetrainingBopeExperiment(
        problem = problem,
        util_func=util_func,
        methods=methods,
        pe_strategies = ["EUBO-zeta"],
        trial_idx=trial_idx,
        save_results = False,
        **experiment_options
    )
    
    exp.run_BOPE_loop()
    
    return OneRun(
        exp_candidate_results=exp.final_candidate_results,
        PE_session_results=exp.PE_session_results,
        pref_data_dict=exp.pref_data_dict,
        subspace_diagnostics=exp.subspace_diagnostics,
        BO_data_dict=exp.BO_data_dict,
    )