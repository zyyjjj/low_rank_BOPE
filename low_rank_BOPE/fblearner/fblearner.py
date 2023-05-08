import copy
import random
import time
from typing import Any, Dict, List, NamedTuple, Tuple
from collections import defaultdict

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
    pref_data_dict: Dict[str, Dict[str, Any]]
    subspace_diagnostics: Dict[str, Dict[str, Any]]
    BO_data_dict: Dict[str, Dict[str, Any]]


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
        trial_idx=trial_idx,
        save_results = False,
        **experiment_options
    )
    
    exp.run_BOPE_loop()

    # process data

    # key: "method__pestrategy", value: List[{"some metric": float or List[float or List[float]]}]
    exp_candidate_results_ = {}
    for k, v in exp.final_candidate_results.items():
        if isinstance(v, defaultdict):
            exp_candidate_results_[k] = dict(v)
    exp_candidate_results__ = {}
    for key, v in exp_candidate_results_.items():
        for v_key in v.keys():
            exp_candidate_results__['__'.join([k, v_key])] = exp_candidate_results_[key][v_key]

    # key: "method__pestrategy", value: List[{"some metric": float or List[float or List[float]]}]
    PE_session_results_ = {}
    for k, v in exp.final_candidate_results.items():
        if isinstance(v, defaultdict):
            PE_session_results_[k] = dict(v)
    PE_session_results__ = {}
    for key, v in PE_session_results_.items():
        for v_key in v.keys():
            PE_session_results__['__'.join([k, v_key])] = PE_session_results_[key][v_key]

    # key: "method__pestrategy", value: {"Y": List, "util_vals": List, "comps": List}
    pref_data_dict_ = {}
    for key in exp.pref_data_dict.keys():
        pref_data_dict_['__'.join([key[0], key[1]])] = exp.pref_data_dict[key]
        for key_, val_ in pref_data_dict_['__'.join([key[0], key[1]])].items():
            pref_data_dict_['__'.join([key[0], key[1]])][key_] = val_.tolist() # convert tensor to list

    # key: "method__pestrategy", value: {"diagnostic": List[float]}
    subspace_diagnostics_ = {}
    for key in exp.subspace_diagnostics.keys():
        subspace_diagnostics_['__'.join([key[0], key[1]])] = exp.subspace_diagnostics[key]
    subspace_diagnostics__ = {}
    for k, v in subspace_diagnostics_.items():
        if isinstance(v, defaultdict):
            subspace_diagnostics__[k] = dict(v)

    # key: "method__pestrategy", value: {"X": List, "Y": List}
    BO_data_dict_ = {}
    for key in exp.BO_data_dict.keys():
        BO_data_dict_['__'.join([key[0], key[1]])] = exp.BO_data_dict[key]
        for key_, val_ in BO_data_dict_['__'.join([key[0], key[1]])].items():
            BO_data_dict_['__'.join([key[0], key[1]])][key_] = val_.tolist() # convert tensor to list

    # these are all defaultdict; if an error occurs, can try casting them to dict 
    return OneRun(
        exp_candidate_results=exp_candidate_results__,
        PE_session_results=PE_session_results__,
        pref_data_dict=pref_data_dict_,
        subspace_diagnostics=subspace_diagnostics__,
        BO_data_dict=BO_data_dict_,
    )