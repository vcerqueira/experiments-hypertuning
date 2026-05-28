import math
from pprint import pprint
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase, mae

from src.loaders import ChronosDataset
from src.utils.reading_data import read_cv_results
from src.config import N_SAMPLES, SEED
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler
from src.fanova import run_fanova

model = 'PatchTST'
target = 'monash_m1_monthly'
partition = 'inner'
RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'

df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
in_set, _ = ChronosDataset.time_wise_split(df, horizon)

mase_func = partial(mase, seasonality=seas_len)

cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')

radar_outer = ModelRadar(
    cv_df=cv_inner,
    metrics=[mase_func],
    model_names=config_ids,
    train_df=in_set,
    hardness_reference=config_ids[0],
    ratios_reference=config_ids[0],
)

err = radar_outer.evaluate(keep_uids=False)
err_df = radar_outer.evaluate(keep_uids=True)

config_pool = NEURAL_CONFIG_POOL[model]
config_list = ConfigSampler.generate_samples(config_pool=config_pool, num_samples=N_SAMPLES, random_state=SEED)
pprint(config_list[0])

config_df = pd.DataFrame(config_list)
err_long = (
    err_df.rename_axis('unique_id')
    .reset_index()
    .melt(id_vars='unique_id', var_name='config_id', value_name='error_score')
)

err_with_config = err_long.merge(config_df, on='config_id', how='left')

lr_log_centers = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
lr_labels = [f'1e{exp:g}' for exp in lr_log_centers]
err_with_config['learning_rate_bin'] = err_with_config['learning_rate'].map(
    lambda lr: lr_labels[min(range(len(lr_log_centers)), key=lambda i: abs(math.log10(lr) - lr_log_centers[i]))]
)

err_with_config['learning_rate_bin'].value_counts()

pprint(err_with_config.head().to_dict(orient='records')[0])

err_with_config.drop(columns=['unique_id', 'config_id', 'learning_rate'], inplace=True)
input_columns = err_with_config.drop(columns=['error_score']).columns.tolist()
err_with_config[input_columns] = err_with_config[input_columns].astype(str)

importance = run_fanova(df=err_with_config,
                        target_col='error_score',
                        max_samples=600,
                        n_trees=16)

print(importance)
