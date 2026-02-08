from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler

model = 'NHITS'

config_pool = NEURAL_CONFIG_POOL[model]


config_df = ConfigSampler.generate_samples(config_pool=config_pool, num_samples=4, return_df=True)

