import os
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.utils.reading_data import read_cv_results
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.config import N_SAMPLES, SEED
from src.neural.param_samples import ConfigSampler

model = 'Autoformer'
# target = 'monash_tourism_quarterly'
# target = 'monash_tourism_monthly'
target = 'monash_m1_monthly'
partition = 'outer'


RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'
# RESULTS_DIR = Path('assets/results_hpo')
print(RESULTS_DIR)

df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
in_set, _ = ChronosDataset.time_wise_split(df, horizon)
in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

# results_dir = Path('../assets/results')

mase_func = partial(mase, seasonality=seas_len)

# 'GRU,ECL,0aac2baac421864d17e8,inner.csv'


cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')
cv_outer, _ = read_cv_results(RESULTS_DIR, model, target, 'outer')

radar_inner = ModelRadar(
    cv_df=cv_inner,
    metrics=[mase_func],
    model_names=config_ids,
    train_df=in_set_train,
    hardness_reference=config_ids[0],
    ratios_reference=config_ids[0],
)

radar_outer = ModelRadar(
    cv_df=cv_outer,
    metrics=[mase_func],
    model_names=config_ids,
    train_df=in_set,
    hardness_reference=config_ids[0],
    ratios_reference=config_ids[0],
)

err_inner = radar_inner.evaluate(keep_uids=False)
print((err_inner == 0).sum())
print(err_inner.describe())
err_inner.sort_values()

err_outer = radar_outer.evaluate(keep_uids=False)
print((err_outer == 0).sum())
print(err_outer.describe())

config_pool = NEURAL_CONFIG_POOL[model]
config_list = ConfigSampler.generate_samples(config_pool=config_pool, num_samples=N_SAMPLES, random_state=SEED)

list_bad_config_ids = err_inner[err_inner == 0].index.tolist()
list_bad_config_ids = err_outer[err_outer == 0].index.tolist()

bad_configs = [c for c in config_list if c['config_id'] in list_bad_config_ids]

print(pd.DataFrame(bad_configs).T)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
