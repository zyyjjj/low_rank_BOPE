test_configs_low_latent_dim = {
    "config_1": {
        "input_dim": 6,
        "outcome_dim": 15,
        "latent_dim": 1,
        "PC_lengthscales": [0.1],
        "PC_scaling_factors": [1.0],
        "noise_std": 0.001,
        "num_initial_samples": 10,
        # "lin_proj_latent_dim": 1,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_2": {
        "input_dim": 6,
        "outcome_dim": 30,
        "latent_dim": 2,
        "PC_lengthscales": [0.1, 0.5],
        "PC_scaling_factors": [3.0, 1.0],
        "noise_std": 0.001,
        "num_initial_samples": 20,
        # "lin_proj_latent_dim": 2,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_3": {
        "input_dim": 6,
        "outcome_dim": 50,
        "latent_dim": 3,
        # "PC_lengthscales": [0.1, 0.5, 0.5],
        # "PC_scaling_factors": [3.0, 1.0, 1.0],
        "PC_lengthscales": [0.1, 0.2, 0.4],
        "PC_scaling_factors": [5.0, 2.0, 1.0],
        "noise_std": 0.001,
        "num_initial_samples": 30,
        # "lin_proj_latent_dim": 3,
        "initial_experimentation_batch": 32,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
}


test_configs_moderate_latent_dim = {
    "config_1": {
        "input_dim": 6,
        "outcome_dim": 15,
        "latent_dim": 3,  # increased from 1
        "PC_lengthscales": [0.1, 0.5, 0.5],
        "PC_scaling_factors": [3.0, 1.0, 1.0],
        "noise_std": 0.001,
        "num_initial_samples": 10,
        # "lin_proj_latent_dim": 1,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_2": {
        "input_dim": 6,
        "outcome_dim": 30,
        "latent_dim": 6,  # increased from 2
        "PC_lengthscales": [0.1, 0.5, 0.5, 0.5, 0.8, 0.8],
        "PC_scaling_factors": [3.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        "noise_std": 0.001,
        "num_initial_samples": 20,
        # "lin_proj_latent_dim": 2,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_3": {
        "input_dim": 6,
        "outcome_dim": 50,
        "latent_dim": 10,
        "PC_lengthscales": [
            0.1,
            0.5,
            0.5,
            0.5,
            0.8,
            0.8,
            1.0,
            1.0,
            1.0,
            1.0,
        ],  # [0.1, 0.5, 0.5],
        "PC_scaling_factors": [
            3.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.2,
            0.2,
            0.2,
            0.2,
        ],
        "noise_std": 0.01,
        "num_initial_samples": 30,
        # "lin_proj_latent_dim": 3,
        "initial_experimentation_batch": 32,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
}


test_configs_outcome_model_fit = {
    "config_1": {
        "input_dim": 6,
        "outcome_dim": 10,
        "latent_dim": 1,
        "PC_lengthscales": [0.1],
        "PC_scaling_factors": [1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 10,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_2": {
        "input_dim": 6,
        "outcome_dim": 20,
        "latent_dim": 2,
        "PC_lengthscales": [0.1, 0.2],
        "PC_scaling_factors": [2.0, 1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 20,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_3": {
        "input_dim": 6,
        "outcome_dim": 30,
        "latent_dim": 3,
        "PC_lengthscales": [0.1, 0.2, 0.3],
        "PC_scaling_factors": [3.0, 2.0, 1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 30,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_4": {
        "input_dim": 6,
        "outcome_dim": 40,
        "latent_dim": 4,
        "PC_lengthscales": [0.1, 0.2, 0.3, 0.4],
        "PC_scaling_factors": [4.0, 3.0, 2.0, 1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 40,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_5": {
        "input_dim": 6,
        "outcome_dim": 50,
        "latent_dim": 5,
        "PC_lengthscales": [0.1, 0.2, 0.3, 0.4, 0.5],
        "PC_scaling_factors": [5.0, 4.0, 3.0, 2.0, 1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 50,
        "initial_experimentation_batch": 32,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
}


test_configs_new_scaling = {
    "config_1": {
        "input_dim": 6,
        "outcome_dim": 10,
        "latent_dim": 1,
        "PC_lengthscales": [0.1],
        "PC_scaling_factors": [1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 10,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_3": {
        "input_dim": 6,
        "outcome_dim": 30,
        "latent_dim": 3,
        "PC_lengthscales": [0.1, 0.1, 0.1],
        "PC_scaling_factors": [10.0, 5.0, 1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 30,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_5": {
        "input_dim": 6,
        "outcome_dim": 50,
        "latent_dim": 5,
        "PC_lengthscales": [0.1, 0.1, 0.1, 0.1, 0.1],
        "PC_scaling_factors": [10.0, 10.0, 5.0, 5.0, 1.0],
        "noise_std": 0.001,
        # "noise_std": 0.0,
        "num_initial_samples": 50,
        "initial_experimentation_batch": 32,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
}


test_configs_new_moderate_scaling = {
    "config_1": {
        "input_dim": 6,
        "outcome_dim": 10,
        "latent_dim": 1,
        "PC_lengthscales": [0.1],
        "PC_scaling_factors": [1.0],
        "noise_std": 0.01,
        # "noise_std": 0.0,
        "num_initial_samples": 100,
        "initial_experimentation_batch": 16,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_3": {
        "input_dim": 6,
        "outcome_dim": 30,
        "latent_dim": 3,
        "PC_lengthscales": [0.5, 0.5, 0.5],
        "PC_scaling_factors": [2.0, 1.5, 1.0],
        "noise_std": 0.01,
        # "noise_std": 0.0,
        "num_initial_samples": 300,
        "initial_experimentation_batch": 24,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
    "config_5": {
        "input_dim": 6,
        "outcome_dim": 50,
        "latent_dim": 5,
        "PC_lengthscales": [0.5, 0.5, 0.5, 0.5, 0.5],
        "PC_scaling_factors": [3.0, 2.5, 2.0, 1.5, 1.0],
        "noise_std": 0.01,
        # "noise_std": 0.0,
        "num_initial_samples": 500,
        "initial_experimentation_batch": 32,
        "n_check_post_mean": 13,
        "every_n_comps": 3,
    },
}
