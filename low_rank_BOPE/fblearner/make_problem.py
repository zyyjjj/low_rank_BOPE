from typing import Dict, Any
import os
import torch
import numpy as np

from low_rank_BOPE.test_problems.car_problems import problem_setup_augmented
from low_rank_BOPE.test_problems.shapes import GradientAwareAreaUtil, Image
from low_rank_BOPE.test_problems.robot.robot import SpotMiniMiniProblem, RobotUtil
from low_rank_BOPE.test_problems.inventory_control import Inventory, \
    InventoryUtil
from low_rank_BOPE.test_problems.synthetic_problem import (
    LinearUtil, PiecewiseLinear,\
    generate_principal_axes, make_controlled_coeffs, make_problem)


def make_problem_and_util_func(
    problem_name: str,
    options: Dict[str, Any],
):
    if problem_name in ["vehiclesafety_5d3d_piecewiselinear_24", 
                        "carcabdesign_7d9d_piecewiselinear_72", 
                        "carcabdesign_7d9d_linear_72"]:
        input_dim, outcome_dim, problem, _, util_func, _, _ = problem_setup_augmented(
            problem_name, 
            augmented_dims_noise=options.get("noise_std", 0.01), 
            noisy=options.get("noisy", True),
            problem_seed = options.get("problem_seed", 1234)
        )
    
    elif problem_name == "8by8_rectangle_gradientAwareArea":
        problem = Image(num_pixels=8)
        util_func = GradientAwareAreaUtil(
            penalty_param=options.get('penalty_param', 0.5), 
            image_shape=(8,8)
        )
    
    elif problem_name == "16by16_rectangle_gradientAwareArea":
        problem = Image(num_pixels=16)
        util_func = GradientAwareAreaUtil(
            penalty_param=options.get('penalty_param', 0.5),
            image_shape=(16,16)
        )
    
    elif problem_name.startswith("robot"):
        # "robot_3_100_1", or "robot_3_500_5",

        _, input_dim, max_timesteps, record_pos_every_n = problem_name.split("_")

        problem = SpotMiniMiniProblem(
            dim = int(input_dim),
            max_timesteps = int(max_timesteps),
            record_pos_every_n=int(record_pos_every_n),
        )

        util_func = RobotUtil(
            y_drift_penalty=options["y_drift_penalty"],
            y_var_penalty=options["y_var_penalty"],
            final_z_reward=options["final_z_reward"],
            z_var_penalty=options["z_var_penalty"],
        )

    elif problem_name == "inventory_100":

        problem = Inventory(
            duration = 100, 
            init_inventory = options.get("init_inventory", 50),
            x_baseline = options.get("x_baseline", 50),
            x_scaling = options.get("x_scaling", 50),
            params = options["simulation_params"]) 
        util_func = InventoryUtil(
            stockout_penalty_per_unit=options.get("stockout_penalty_per_unit", 0.1),
            holding_cost_per_unit=options.get("holding_cost_per_unit", 0.1),
            order_cost_one_time=options.get("order_cost_one_time", 1.0),
            order_cost_per_unit=options.get("order_cost_per_unit", 0.1),
        ) 
    
    elif problem_name.startswith("PTS"):

        script_dir = os.path.dirname(os.path.abspath(__file__)) # TODO: test this
        
        CSVData = open(
            os.path.join(script_dir, "data", "PTS", 
                         f"metric_corr_exp_{options['matrix_id']}.csv")
        )
        metric_corr = torch.tensor(
            np.loadtxt(CSVData, delimiter=","), dtype=torch.double)
        outcome_dim = len(metric_corr)
        U, S, V = torch.linalg.svd(
            metric_corr + torch.diag(torch.ones(len(metric_corr))) * 1e-10)
        true_axes = V[:options["latent_dim"]]
        scaling = torch.sqrt(S[:options["latent_dim"]])

        problem = make_problem(
            input_dim = options["input_dim"],
            outcome_dim = outcome_dim,
            noise_std = options["noise_std"],
            true_axes = true_axes,
            PC_lengthscales = options["PC_lengthscales"],
            PC_scaling_factors = scaling,
            problem_seed = options["problem_seed"],
            state_dict_str = None
        )

        beta = make_controlled_coeffs(
            full_axes=V,
            latent_dim=options["latent_dim"],
            alpha=options["alpha"],
            n_reps = 1,
            dtype=torch.double,
            seed = options["problem_seed"]
        ) # shape is 1 x outcome_dim

        if options["util_type"] == "linear":
            util_func = LinearUtil(beta=beta.transpose(-2, -1))
        elif options["util_type"] == "piecewiselinear":
            util_func = PiecewiseLinear(
                beta1=beta * options.get("util_coeff_multiplier", 5.0),
                beta2=beta,
                thresholds=torch.tensor([0.]*outcome_dim, dtype=torch.double)
            )

    return problem, util_func