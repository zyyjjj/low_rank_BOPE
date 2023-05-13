# Configuration for the outcome and utility functions

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
    "robot_3_500_5": {
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
        "order_cost_per_unit": 0.1,
        "simulation_params": {
            'demand_mean' : 5, 
            'demand_std' : 2,
            'lead_time' : 10, # tau in the slides
            'stockout_penalty' : 0.1, # doesn't matter for us
            'holding_cost' : 0.01, # doesn't matter for us
            'K' : 1, # doesn't matter for us 
            'c' : 0.1 # doesn't matter for us
        }
    },
    "PTS=6_input=3_outcome=45_latent=3_alpha=0.5_pwlinear": {
        "matrix_id": 6,
        "input_dim": 3,
        "outcome_dim": 45,
        "latent_dim": 3,
        "alpha": 0.5,
        "noise_std": 0.01,
        "util_type": "piecewiselinear",
        "PC_lengthscales": [0,1, 0.1, 0.1],
        "problem_seed": 1234
    },
    "PTS=6_input=3_outcome=45_latent=3_alpha=1.0_pwlinear": {
        "matrix_id": 6,
        "input_dim": 3,
        "outcome_dim": 45,
        "latent_dim": 3,
        "alpha": 1.0,
        "noise_std": 0.01,
        "util_type": "piecewiselinear",
        "util_coeff_multiplier": 5.0,
        "PC_lengthscales": [0,1, 0.1, 0.1],
        "problem_seed": 1234
    }
}

# Configuration for BOPE experiment pipeline

EXPERIMENT_SETUPS = {
    "vehiclesafety_5d3d_piecewiselinear_24": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 4,
        "initial_experimentation_batch": 32,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 8,
        "n_meta_iters": 10,
    },
    "carcabdesign_7d9d_piecewiselinear_72": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 4,
        "initial_experimentation_batch": 32,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 8,
        "n_meta_iters": 10,
    },
    "carcabdesign_7d9d_linear_72": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 4,
        "initial_experimentation_batch": 32,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 8,
        "n_meta_iters": 10,
    },
    "8by8_rectangle_gradientAwareArea": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 4,
        "initial_experimentation_batch": 32,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 8,
        "n_meta_iters": 10,
        "standardize": False,
    },
    "16by16_rectangle_gradientAwareArea": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 8,
        "initial_experimentation_batch": 64,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 16,
        "n_meta_iters": 5,
        "standardize": False,
    },
    "robot_3_100_1": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 8,
        "initial_experimentation_batch": 64,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 16,
        "n_meta_iters": 5,
        "standardize": False, 
    },
    "inventory_100": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 8,
        "initial_experimentation_batch": 64,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 16,
        "n_meta_iters": 5,
        "standardize": False
    },
    "PTS=6_input=3_outcome=45_latent=3_alpha=0.5_pwlinear": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 4,
        "initial_experimentation_batch": 32,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 8,
        "n_meta_iters": 10,
    },
    "PTS=6_input=3_outcome=45_latent=3_alpha=1.0_pwlinear": {
        "pe_strategies": ["EUBO-zeta"],
        "every_n_comps": 2,
        "n_check_post_mean": 4,
        "initial_experimentation_batch": 32,
        "pca_var_threshold": 0.9,
        "n_BO_iters": 1,
        "BO_batch_size": 8,
        "n_meta_iters": 10,
    }
}

METHOD_SETUPS = {
    "wpca_est_1_rt": {
        "wpca_type": "rank_cts",
        "wpca_options": {"k": 1}
    },
    "wpca_est_1e-3_rt": {
        "wpca_type": "rank_cts",
        "wpca_options": {"k": 0.001}
    },
    "wpca_est_1e-6_rt": {
        "wpca_type": "rank_cts",
        "wpca_options": {"k": 0.000001}
    },
}