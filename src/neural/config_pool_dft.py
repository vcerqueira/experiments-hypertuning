from ray import tune

NEURAL_CONFIG_POOL = {
    'NHITS': {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
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
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    },

    'PatchTST': {
        "input_size_multiplier": [1, 2, 3],
        "h": None,
        "hidden_size": tune.choice([16, 128, 256]),
        "n_heads": tune.choice([4, 16]),
        "patch_len": tune.choice([16, 24]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "revin": tune.choice([False, True]),
        "max_steps": tune.choice([500, 1000, 5000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    },

    'MLP': {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([256, 512, 1024]),
        "num_layers": tune.randint(2, 6),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    },

    'KAN': {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "grid_size": tune.choice([5, 10, 15]),
        "spline_order": tune.choice([2, 3, 4]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    },

    'TFT': {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    },

    'GRU': {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "encoder_hidden_size": tune.choice([16, 32, 64, 128]),
        "encoder_n_layers": tune.randint(1, 4),
        "context_size": tune.choice([5, 10, 50]),
        "decoder_hidden_size": tune.choice([16, 32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    },

    'Autoformer': {
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.choice([500, 1000, 2000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }
}
