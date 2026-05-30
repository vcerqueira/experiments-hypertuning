from pathlib import Path
from functools import partial

import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.utils.reading_data import read_cv_results
from src.coseal.active_testing import active_testing_selection
from src.coseal.preference import bradley_terry_ranking

CORR_SELECTION = 0.9
N_TRIALS = 25

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
    'GRU'
]

model = 'NHITS'

RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'


def load_dataset(target):
    if target in ChronosDataset.FREQUENCY_MAP_DATASETS:
        df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
    else:
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target, resample_to='D'
        )

    in_set, _ = ChronosDataset.time_wise_split(df, horizon)
    in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

    return in_set, in_set_train, seas_len


scores = []
for i, target in enumerate(DATASETS):
    print(target)
    in_set, in_set_train, seas_len = load_dataset(target)
    mase_func = partial(mase, seasonality=seas_len)

    cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')

    radar_outer = ModelRadar(
        cv_df=cv_inner,
        metrics=[mase_func],
        model_names=config_ids,
        train_df=in_set_train,
        hardness_reference=config_ids[0],
        ratios_reference=config_ids[0],
    )

    err_df = radar_outer.evaluate(keep_uids=True).reset_index()
    err_df['dataset'] = target

    err_df['unique_id'] = err_df['unique_id'].apply(lambda x: f"{i}{x}")

    scores.append(err_df)

scores_df = pd.concat(scores).set_index('unique_id')

scores_f = []
for i, target in enumerate(DATASETS):
    print(target)

    in_set, in_set_train, seas_len = load_dataset(target)
    mase_func = partial(mase, seasonality=seas_len)

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
    err_outer = radar_outer.evaluate(keep_uids=False)

    scores_sub = scores_df.query(f'dataset!="{target}"').drop(columns='dataset')

    print('at1')
    at_list = active_testing_selection(scores_df=scores_sub,
                                       use_ranks=False,
                                       max_trials=N_TRIALS,
                                       corr_threshold=CORR_SELECTION,
                                       delta=0)

    print('at2')
    atr_list = active_testing_selection(scores_df=scores_sub,
                                        use_ranks=True,
                                        max_trials=N_TRIALS,
                                        corr_threshold=CORR_SELECTION,
                                        delta=0)

    print('at3')
    bt_list = bradley_terry_ranking(scores_df=scores_sub,
                                    max_trials=N_TRIALS,
                                    corr_threshold=CORR_SELECTION)

    err_sample = err_inner.sample(N_TRIALS).index.tolist()

    active_testing_cfg = err_inner[at_list].idxmin()
    active_testing_r_cfg = err_inner[atr_list].idxmin()
    pref_learning_cfg = err_inner[bt_list].idxmin()
    random_search_cfg = err_sample.idxmin()

    rs_err = err_outer[active_testing_cfg]
    active_testing_err = err_outer[active_testing_r_cfg]
    active_testing_r_err = err_outer[pref_learning_cfg]
    pref_learning_err = err_outer[random_search_cfg]

    scores_f.append({
        'Dataset': target,
        'RS': rs_err,
        'AT': active_testing_err,
        'ATR': active_testing_r_err,
        'PL': pref_learning_err,
    })

    print(pd.DataFrame(scores_f))

print(pd.DataFrame(scores_f))

a = pd.DataFrame(scores_f)
a.set_index('dataset').rank(axis=1).mean()
