import random
import hashlib
import json
from typing import Dict

import numpy as np
import pandas as pd

from src.config import SEED, N_SAMPLES


class ConfigSampler:
    # from ray import tune
    #
    # config_space_ = {
    #     "n_pool_kernel_size": tune.choice([2, 3, 5]),
    #     "learning_rate": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([16, 32, 64]),
    #     "activation": "relu"
    # }
    #
    # sample_list_ = ConfigSampler.generate_samples(config_space_, num_samples=4)
    #
    # df = pd.DataFrame(sample_list_).set_index('config_id')

    BAD_CONFIGS = ['fd1d8da7684e79a15d7d',
                   '2b034fe8fd6d9ee361de',
                   'aefb0e3b650af6c4f8ea',
                   '856f1172fc2f3108b2eb',
                   '6c6aa783172ef07d645d',
                   'e23b2735de790f35e3b4',
                   '0ae460bbd17da6e08ef9',
                   '09316b60a187415cd7dc',
                   'fd9e39298706aa752095']

    @classmethod
    def generate_samples(cls,
                         config_pool: Dict,
                         num_samples: int = N_SAMPLES,
                         random_state: int = SEED,
                         return_df: bool = False):

        """
        Uninformed Random Sampling
        """

        cls.set_seeds(random_state)

        sample_list = []
        for i in range(num_samples):
            sample = {
                k: (v.sample() if hasattr(v, 'sample') else v)
                for k, v in config_pool.items()
            }

            sample['config_id'] = cls.get_config_id(sample)

            # if sample['batch_size'] > 32:
            #     continue

            sample_list.append(sample)

        sample_list = [sample for sample in sample_list if sample['config_id'] not in cls.BAD_CONFIGS]

        if return_df:
            df = pd.DataFrame(sample_list).set_index('config_id')
            return df

        return sample_list

    @staticmethod
    def set_seeds(seed: int = SEED):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_config_id(config):
        hash_len = 20

        config_str = json.dumps(config, sort_keys=True)
        config_id = hashlib.md5(config_str.encode()).hexdigest()[:hash_len]

        return config_id
