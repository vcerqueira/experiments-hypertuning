from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import plotnine as p9
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.utils.reading_data import read_cv_results
from src.utils.plotting import THEME

DATASETS = [
    'monash_tourism_quarterly',
    'monash_tourism_monthly',
    'monash_m3_quarterly',
    'monash_m3_monthly',
    'monash_m1_monthly',
    'monash_m1_quarterly',
    'TrafficL',
    'ECL',
]

MODEL_LIST = [
    'KAN',
    'MLP',
    'NHITS',
    'TFT',
    'PatchTST',
    'GRU',
    'Informer'
]

TRAJECTORY_SIZE = 500
N_REPS = 100
SHOW_UNCERTAINTY_BANDS = False
RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'
OUTPUT_DIR = Path('assets/outputs')


def load_dataset_splits(target):
    if target in ChronosDataset.FREQUENCY_MAP_DATASETS:
        df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
        in_set, _ = ChronosDataset.time_wise_split(df, horizon)
        in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)
    else:
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target, resample_to='D'
        )
        in_set, _ = ChronosDataset.time_wise_split(df, horizon)
        in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

    return in_set, in_set_train, seas_len


def build_overtuning_long(model_rel_ot):
    rep_rows = []
    for model, rel_or_list in model_rel_ot.items():
        for rel_ot in rel_or_list:
            for step, value in rel_ot.items():
                rep_rows.append({
                    'model': model,
                    'step': step + 1,
                    'rel_overtuning': value,
                })

    overtuning_long = (
        pd.DataFrame(rep_rows)
        .groupby(['model', 'step'], as_index=False)['rel_overtuning']
        .agg(mean='mean', std='std')
        .rename(columns={'mean': 'rel_overtuning'})
    )

    if SHOW_UNCERTAINTY_BANDS:
        overtuning_long['ymin'] = overtuning_long['rel_overtuning'] - overtuning_long['std']
        overtuning_long['ymax'] = overtuning_long['rel_overtuning'] + overtuning_long['std']

    return overtuning_long


def plot_overtuning(overtuning_long, target):
    p = p9.ggplot(
        overtuning_long,
        p9.aes(x='step', y='rel_overtuning', color='model', group='model'),
    )

    if SHOW_UNCERTAINTY_BANDS:
        p = p + p9.geom_ribbon(
            p9.aes(ymin='ymin', ymax='ymax', fill='model', group='model'),
            alpha=0.2,
            color=None,
        )

    p = (
            p
            + p9.geom_line(size=1.0)
            + p9.labs(
        x='Hyperparameter search step',
        y='Relative overtuning',
        color='Model',
        fill='Model',
    )
            + THEME
            + p9.theme(legend_position='top')
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p.save(OUTPUT_DIR / f'overtuning,{target}.pdf', width=8, height=6)


for target in DATASETS:
    print(f'\n=== {target} ===')
    in_set, in_set_train, seas_len = load_dataset_splits(target)
    mase_func = partial(mase, seasonality=seas_len)

    model_rel_ot = {}
    for model in MODEL_LIST:
        print(model)

        cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')
        cv_outer, _ = read_cv_results(RESULTS_DIR, model, target, 'outer')

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

        rel_or_list = []
        for _ in range(N_REPS):
            err_sample = err_inner.sample(TRAJECTORY_SIZE)
            err_outer_sample = err_outer[err_sample.index].values

            val_incumbent_errors = np.minimum.accumulate(err_sample).values

            incumbent_mask = np.concatenate(
                ([True], val_incumbent_errors[1:] < val_incumbent_errors[:-1])
            )

            test_incumbents = np.where(incumbent_mask, err_outer_sample, np.nan)

            mask = np.isnan(test_incumbents)
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, out=idx)
            test_incumbents = test_incumbents[idx]

            test_lambda_1 = test_incumbents[0]
            min_test_so_far = np.minimum.accumulate(test_incumbents)

            abs_overtuning = test_incumbents - min_test_so_far
            max_improvement = test_lambda_1 - min_test_so_far

            with np.errstate(divide='ignore', invalid='ignore'):
                rel_ot = np.where(
                    max_improvement > 0, abs_overtuning / max_improvement, 0.0
                )

            rel_or_list.append(pd.Series(rel_ot))

        model_rel_ot[model] = rel_or_list

    overtuning_long = build_overtuning_long(model_rel_ot)
    plot_overtuning(overtuning_long, target)
