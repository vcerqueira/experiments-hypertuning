import os
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import plotnine as p9
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset
from src.utils.reading_data import read_cv_results

# NHITS,monash_m1_monthly,0a37cbc4d85c6d5e13d7,outer
# model = 'NHITS'
# model = 'PatchTST'
# model = 'MLP'
# target = 'monash_m1_monthly'
# target = 'monash_m3_quarterly'
target = 'monash_m3_monthly'
partition = 'outer'

MODEL_LIST = ['KAN',
              'MLP',
              'NHITS',
              'TFT',
              'PatchTST',
              'GRU']
LEARNING_CURVE = [1, 2, 5, 10, 15, 25, 50, 75, 100, 200, 300, 400]
N_REPS = 50

df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
in_set, _ = ChronosDataset.time_wise_split(df, horizon)
in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

mase_func = partial(mase, seasonality=seas_len)

results_dir = Path() / 'assets' / 'results'

model_scores = {}
for model in MODEL_LIST:
    print(model)

    cv_inner, config_ids = read_cv_results(results_dir, model, target, 'inner')
    cv_outer, _ = read_cv_results(results_dir, model, target, 'outer')

    if cv_inner is None:
        continue

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
    err_outer = radar_outer.evaluate(keep_uids=False)

    scores = {}
    for s in LEARNING_CURVE:
        s_scores = []
        for _ in range(N_REPS):
            err_sample = err_inner.sample(s)

            selected_config = err_sample.idxmin()

            s_scores.append(err_outer[selected_config])

        scores[s] = np.mean(s_scores)

    model_scores[model] = scores

scores_df = pd.DataFrame(model_scores)
print(scores_df)

scores_long = (
    scores_df.tail(-2)
    .reset_index(names='n_samples')
    .melt(id_vars='n_samples', var_name='model', value_name='score')
)

x_breaks = sorted(scores_long['n_samples'].unique().tolist())
p = (
    p9.ggplot(scores_long, p9.aes(x='n_samples', y='score', color='model', group='model'))
    + p9.geom_line(size=1.0)
    + p9.geom_point(size=1.8)
    + p9.scale_x_log10(breaks=x_breaks)
    + p9.labs(x='Number of sampled configs (log10 scale)', y='MASE', color='Model')
    + p9.theme_538()
)

p.save(f'convergence,{target}.pdf', width=8, height=6)

