"""
i can use this later for AT based on correlation
or aggregating redundant configs, let's see

"""
from pathlib import Path
from functools import partial

import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase, mae

from src.loaders import ChronosDataset

# NHITS,monash_m1_monthly,0a37cbc4d85c6d5e13d7,outer
model = 'NHITS'
target = 'monash_m1_monthly'
partition = 'outer'
RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'

df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
in_set, _ = ChronosDataset.time_wise_split(df, horizon)


pattern = f"{model},{target},*,{partition}.csv"
config_files = list(RESULTS_DIR.glob(pattern))
config_ids = [f.stem.split(',')[2] for f in config_files]

mase_func = partial(mase, seasonality=seas_len)

cv_score_file0 = config_files[0]
config_id0 = cv_score_file0.stem.split(',')[2]
cv_outer0 = pd.read_csv(cv_score_file0)
cv_outer0.rename(columns={model: config_id0}, inplace=True)

for file in config_files[1:]:
    config_id = file.stem.split(',')[2]
    cv_outer = pd.read_csv(file)
    cv_outer.rename(columns={model: config_id}, inplace=True)

    cv_outer0 = cv_outer0.merge(cv_outer.drop(columns=['y']), on=['unique_id', 'ds', 'cutoff'], how='inner')

radar_outer = ModelRadar(
    cv_df=cv_outer0,
    metrics=[mase_func],
    model_names=config_ids,
    train_df=in_set,
    hardness_reference=config_ids[0],
    ratios_reference=config_ids[0],
)
err = radar_outer.evaluate(keep_uids=False)
err_df = radar_outer.evaluate(keep_uids=True)

err_df.corr()
err_df.corr()['49c9ec496fbf64c28e52'].describe()
