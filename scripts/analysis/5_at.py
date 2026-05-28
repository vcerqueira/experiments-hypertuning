from pathlib import Path
from functools import partial

import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset
from src.utils.reading_data import read_cv_results
from src.coseal.active_testing import active_testing_selection
from src.coseal.preference import active_testing_bradley_terry

# NHITS,monash_m1_monthly,0a37cbc4d85c6d5e13d7,outer
model = 'NHITS'

DATASET_LIST = [
    'monash_m1_monthly',
    'monash_m1_quarterly',
    'monash_m3_monthly',
    'monash_m3_quarterly',
    'monash_tourism_monthly',
    'monash_tourism_quarterly',
]

results_dir = Path() / 'assets' / 'results'

# target = 'monash_m1_monthly'

scores = []
for i, target in enumerate(DATASET_LIST):
    print(target)
    df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
    # df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
    in_set, _ = ChronosDataset.time_wise_split(df, horizon)
    in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

    mase_func = partial(mase, seasonality=seas_len)

    cv_inner, config_ids = read_cv_results(results_dir, model, target, 'inner')

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

scores_df.mean()

# err_df.corr()

N_TRIALS = 25

scores_f = []
for i, target in enumerate(DATASET_LIST):
    print(target)

    df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
    # df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
    in_set, _ = ChronosDataset.time_wise_split(df, horizon)
    in_set_train, _ = ChronosDataset.time_wise_split(in_set, horizon)

    mase_func = partial(mase, seasonality=seas_len)

    cv_inner, config_ids = read_cv_results(results_dir, model, target, 'inner')
    cv_outer, _ = read_cv_results(results_dir, model, target, 'outer')

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

    config_list1 = active_testing_selection(scores_df=scores_sub,
                                            use_ranks=False,
                                            max_trials=N_TRIALS,
                                            corr_threshold=0.97,
                                            delta=0.01)

    config_list2 = active_testing_selection(scores_df=scores_sub,
                                            use_ranks=False,
                                            max_trials=N_TRIALS,
                                            corr_threshold=0.9,
                                            delta=0.01)

    config_list3 = active_testing_bradley_terry(scores_df=scores_sub,
                                                max_trials=N_TRIALS,
                                                corr_threshold=0.97)

    at1 = err_inner[config_list1].idxmin()
    at2 = err_inner[config_list2].idxmin()
    at3 = err_inner[config_list3].idxmin()

    err_sample = err_inner.sample(N_TRIALS)

    selected_config = err_sample.idxmin()

    rs_err = err_outer[selected_config]
    at1_err = err_outer[at1]
    at2_err = err_outer[at2]
    at3_err = err_outer[at3]

    scores_f.append({
        'dataset': target,
        'random-search': rs_err,
        'AT1': at1_err,
        'AT2': at2_err,
        'AT3': at3_err,
    })

    print(pd.DataFrame(scores_f))

print(pd.DataFrame(scores_f))
