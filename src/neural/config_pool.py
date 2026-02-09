from ray import tune

# todo acrescentar params

NEURAL_CONFIG_POOL = {
    'NHITS': {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        # "h": None,
        "n_pool_kernel_size": tune.choice(
            [
                [2, 2, 1],
                3 * [1],
                3 * [2],
                3 * [4],
                [8, 4, 1],
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
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        # "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    },

    'PatchTST': {
        "input_size_multiplier": [1, 2, 3],
        # "h": None,
        "hidden_size": tune.choice([16, 128, 256]),
        "n_heads": tune.choice([4, 16]),
        "patch_len": tune.choice([16, 24]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "revin": tune.choice([False, True]),
        "max_steps": tune.choice([500, 1000, 5000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        # "loss": None,
        "random_seed": tune.randint(1, 20),
    }
}
