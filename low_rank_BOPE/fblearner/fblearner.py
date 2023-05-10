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
from problem_setups import PROBLEM_SETUPS, EXPERIMENT_SETUPS, METHOD_SETUPS

from low_rank_BOPE.bope_class_retraining import RetrainingBopeExperiment

class OneRun(NamedTuple):
    exp_candidate_results: Dict[str, Dict[str, Any]]
    PE_session_results: Dict[str, Dict[str, Any]]
    pref_data_dict: Dict[str, Dict[str, Any]]
    subspace_diagnostics: Dict[str, Dict[str, Any]]
    BO_data_dict: Dict[str, Dict[str, Any]]
    time_consumption: Dict[str, Dict[str, Any]]


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
    "PTS=6_input=3_outcome=45_latent=3_PCls=0.5_pwlinear_seed=1234_alpha=0.5",
    "PTS=6_input=3_outcome=45_latent=3_PCls=0.5_pwlinear_seed=1234_alpha=1.0",
]

"""

# As an example, running "8by8_rectangle_gradientAwareArea"
@flow.registered(owners=["oncall+ae_experiments"])
@flow.typed()
def main(
    problem_name: str = "8by8_rectangle_gradientAwareArea", 
    baselines:  List[str] = ["pca"],
    trial_idx_start: int = 1,
    trial_idx_end: int = 10,
) -> Dict[str, List[OneRun]]:
    
    problem, util_func = make_problem_and_util_func(
        problem_name, options = PROBLEM_SETUPS[problem_name])
    
    for baseline in baselines:
        for trial_idx in range(trial_idx_start, trial_idx_end+1):
            options = copy.deepcopy(EXPERIMENT_SETUPS[problem_name])
            if baseline in METHOD_SETUPS:
                method_options = METHOD_SETUPS[baseline]
                options.update(method_options)
            run_one_trial(
                problem=problem, 
                util_func=util_func, 
                methods=[baseline], 
                trial_idx=trial_idx,
                experiment_options=options
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
    # first cast items in the nested dict from defaultdict to dict
    exp_candidate_results_ = {}
    for k, v in exp.final_candidate_results.items():
        if isinstance(v, defaultdict):
            exp_candidate_results_[k] = dict(v)
    # then turn nested dict[m][p] to unnested dict[m__p]
    exp_candidate_results__ = {}
    for k_, v_ in exp_candidate_results_.items():
        for v_key in v_.keys():
            exp_candidate_results__['__'.join([k_, v_key])] = exp_candidate_results_[k_][v_key]

    # key: "method__pestrategy", value: List[{"some metric": float or List[float or List[float]]}]
    # first cast items in the nested dict from defaultdict to dict
    PE_session_results_ = {}
    for k, v in exp.PE_session_results.items():
        if isinstance(v, defaultdict):
            PE_session_results_[k] = dict(v)
    # then turn nested dict[m][p] to unnested dict[m__p]
    PE_session_results__ = {}
    for k_, v_ in PE_session_results_.items():
        for v_key in v_.keys():
            PE_session_results__['__'.join([k_, v_key])] = PE_session_results_[k_][v_key]

    # key: "method__pestrategy", value: {"Y": List, "util_vals": List, "comps": List}
    # turn dict key from tuple (m, p) to str "m__p" 
    pref_data_dict_ = {}
    for key in exp.pref_data_dict.keys():
        pref_data_dict_['__'.join([key[0], key[1]])] = exp.pref_data_dict[key]
        # convert tensors under "Y", "util_vals", "comps" to list
        for key_, val_ in pref_data_dict_['__'.join([key[0], key[1]])].items():
            pref_data_dict_['__'.join([key[0], key[1]])][key_] = val_.tolist()

    # key: "method__pestrategy", value: {"diagnostic": List[float]}
    # turn dict key from tuple (m, p) to str "m__p" 
    subspace_diagnostics_ = {}
    for key in exp.subspace_diagnostics.keys():
        subspace_diagnostics_['__'.join([key[0], key[1]])] = exp.subspace_diagnostics[key]
    # cast items from defaultdict to dict
    subspace_diagnostics__ = {}
    for k_, v_ in subspace_diagnostics_.items():
        if isinstance(v_, defaultdict):
            subspace_diagnostics__[k_] = dict(v_)

    # key: "method__pestrategy", value: {"X": List, "Y": List}
    # turn dict key from tuple (m, p) to str "m__p" 
    BO_data_dict_ = {}
    for key in exp.BO_data_dict.keys():
        BO_data_dict_['__'.join([key[0], key[1]])] = exp.BO_data_dict[key]
        # convert tensors under "X" and "Y" to list
        for key_, val_ in BO_data_dict_['__'.join([key[0], key[1]])].items():
            BO_data_dict_['__'.join([key[0], key[1]])][key_] = val_.tolist()

    # key: "method__pestrategy", value: {"time for doing something": List[float]}
    # turn dict key from tuple (m, p) to str "m__p" 
    time_consumption_ = {}
    for key in exp.time_consumption.keys():
        time_consumption_['__'.join([key[0], key[1]])] = exp.time_consumption[key]
    # cast items from defaultdict to dict
    time_consumption__ = {}
    for k_, v_ in time_consumption_.items():
        if isinstance(v_, defaultdict):
            time_consumption__[k_] = dict(v_)

    return OneRun(
        exp_candidate_results=exp_candidate_results__,
        PE_session_results=PE_session_results__,
        pref_data_dict=pref_data_dict_,
        subspace_diagnostics=subspace_diagnostics__,
        BO_data_dict=BO_data_dict_,
        time_consumption=time_consumption__,
    )