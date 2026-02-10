from ray import tune

NEURAL_CONFIG_POOL = {
    'NHITS': {
        "input_size_multiplier": tune.choice([1, 2]),
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                [3, 2, 1],  #
                [6, 2, 1],  #
                [8, 4, 1],
                3 * [1],
                3 * [2],
                3 * [4],
                [24, 8, 2],  #
                [12, 6, 3],  #
                [16, 8, 1]
            ]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [6, 2, 1],  #
                [24, 8, 2],  #
                [1, 1, 1],
            ]
        ),
        "mlp_units": tune.choice(
            [
                3 * [[64, 64]],
                3 * [[64, 64, 64]],
                3 * [[128, 128]],
                3 * [[128, 128, 128]],
                3 * [[256, 256]],
                3 * [[256, 256, 256]],
                3 * [[512, 512]],
                3 * [[512, 512, 512]],
            ]
        ),

        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None,
                                    "robust",
                                    "revin",
                                    "standard"]),
        "max_steps": tune.quniform(lower=500,
                                   # upper=1500,
                                   upper=2000,
                                   q=100),
        "pooling_mode": tune.choice(['MaxPool1d', 'AvgPool1d']),
        "interpolation_mode": tune.choice(['linear', 'nearest', 'cubic']),
        "start_padding_enabled": tune.choice([True, False]),
        "dropout_prob_theta": tune.choice([0.0, 0.1, 0.2]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        # "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    },

    'PatchTST': {
        "input_size_multiplier": [1, 2, 3],
        "hidden_size": tune.choice([16, 128, 256]),
        "linear_hidden_size": tune.choice([64, 128, 256]),
        "n_heads": tune.choice([2, 4, 8, 16, 24]),
        "encoder_layers": tune.choice([1, 2, 3]),
        "patch_len": tune.choice([16, 24]),
        "stride": tune.choice([2, 4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "revin": tune.choice([False, True]),
        "max_steps": tune.choice([500, 1000, 2000, 5000]),
        "activation": tune.choice(["relu", "gelu"]),
        "res_attention": tune.choice([True, False]),
        "batch_normalization": tune.choice([True, False]),
        "learn_pos_embed": tune.choice([True, False]),
        "start_padding_enabled": tune.choice([True, False]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
    }
}
