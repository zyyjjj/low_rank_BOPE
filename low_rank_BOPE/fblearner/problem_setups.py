




PROBLEM_SETUPS = {
    "vehiclesafety_5d3d_piecewiselinear_24": {
        "noise_std": 0.01,
        "problem_seed": 1234,
        "noisy": True
    },
    "carcabdesign_7d9d_piecewiselinear_72": {
        "noise_std": 0.01,
        "problem_seed": 1234,
        "noisy": True
    },
    "carcabdesign_7d9d_linear_72": {
        "noise_std": 0.01,
        "problem_seed": 1234,
        "noisy": True
    },
    "8by8_rectangle_gradientAwareArea": {
        "penalty_param": 0.5
    },
    "16by16_rectangle_gradientAwareArea": {
        "penalty_param": 0.5
    },
    "robot_3_100_1": {
        "y_drift_penalty": 0.1,
        "y_var_penalty": 0.1,
        "final_z_reward": 0.1,
        "z_var_penalty": 0.1
    },
    "inventory_100": {
        "init_inventory": 50,
        "x_baseline": 50,
        "x_scaling": 50,
        "stockout_penalty_per_unit": 0.1,
        "holding_cost_per_unit": 0.1,
        "order_cost_one_time": 0.0,
        "order_cost_per_unit": 0.1
    }
}

# TODO: double check, complete 