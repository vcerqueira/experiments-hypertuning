from functools import partial
from pathlib import Path

import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.utils.reading_data import read_cv_results
from src.config import RESULTS_DIR

MODELS = [
    'GRU',
    # 'KAN',
    # 'MLP',
    # 'NHITS',
    # 'PatchTST',
    'TFT'
]
DATASETS = [
    # 'monash_tourism_quarterly',
    # 'monash_tourism_monthly',
    # 'monash_m3_quarterly',
    # 'monash_m3_monthly',
    # 'monash_m1_monthly',
    # 'monash_m1_quarterly',
    # 'Weather',
    # 'TrafficL',
    'ECL',
]

global_bad_ids = []


def find_bad_ids(model, target):
    if target in [*ChronosDataset.FREQUENCY_MAP_DATASETS]:
        df, horizon, _, _, seas_len = ChronosDataset.load_everything(target)
    else:
        df, horizon, _, _, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

    in_set, _ = ChronosDataset.time_wise_split(df, horizon)
    in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

    mase_func = partial(mase, seasonality=seas_len)

    cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')
    if not config_ids:
        return []

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
    err_outer = radar_outer.evaluate(keep_uids=False)

    inner_list_bad_ids = err_inner[err_inner == 0].index.tolist()
    outer_list_bad_ids = err_outer[err_outer == 0].index.tolist()

    list_bad_ids = list(set(inner_list_bad_ids + outer_list_bad_ids))

    return list_bad_ids


def remove_bad_config_files(model, target, bad_ids):
    for bad_id in bad_ids:
        for partition in ('inner', 'outer'):
            path = RESULTS_DIR / f'{model},{target},{bad_id},{partition}.csv'
            if path.exists():
                path.unlink()
                print(f'Removed {path}')


for model in MODELS:
    for target in DATASETS:
        print(f'Checking {model}, {target}...')
        bad_ids = find_bad_ids(model, target)
        if not bad_ids:
            print(f'  No bad configs found.')
            continue

        print(f'  Found {len(bad_ids)} bad config(s).')
        global_bad_ids.extend(bad_ids)
        remove_bad_config_files(model, target, bad_ids)

print(f'\nTotal bad config ids across all pairs: {len(global_bad_ids)}')
print(f'Unique bad config ids: {len(set(global_bad_ids))}')
