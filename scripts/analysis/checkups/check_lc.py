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
    # 'monash_tourism_quarterly',
    # 'monash_tourism_monthly',
    # 'monash_m3_quarterly',
    # 'monash_m3_monthly',
    'monash_m1_monthly',
    # 'monash_m1_quarterly',
    # 'TrafficL',
    # 'ECL',
]

MODEL_LIST = [
    # 'KAN',
    # 'MLP',
    # 'NHITS',
    # 'TFT',
    # 'PatchTST',
    # 'GRU'
    # 'Autoformer',
    'Informer',
]

LEARNING_CURVE = [1, 2, 5, 10, 15, 25, 40, 60,95,100,150,175,225, 280]
N_REPS = 100
SHOW_UNCERTAINTY_BANDS = False
# RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'
RESULTS_DIR = Path('assets/results_hpo')
print(RESULTS_DIR.absolute())
OUTPUT_DIR = Path('assets/')


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


def build_scores_long(model_scores):
    records = []
    for model, scores in model_scores.items():
        for n_samples, rep_scores in scores.items():
            rep_scores = np.asarray(rep_scores)
            record = {
                'n_samples': n_samples,
                'model': model,
                'score': rep_scores.mean(),
                # 'score': np.median(rep_scores),
            }
            if SHOW_UNCERTAINTY_BANDS:
                record['ymin'] = rep_scores.mean() - rep_scores.std()
                record['ymax'] = rep_scores.mean() + rep_scores.std()
            records.append(record)

    scores_df = pd.DataFrame(
        {
            model: {n_samples: np.mean(rep_scores) for n_samples, rep_scores in scores.items()}
            for model, scores in model_scores.items()
        }
    )
    scores_long = pd.DataFrame(records)
    scores_long['n_samples'] = pd.to_numeric(scores_long['n_samples'])
    return scores_df, scores_long


def plot_learning_curve(scores_long, target):
    df = scores_long.copy()
    df['n_samples'] = pd.to_numeric(df['n_samples'], errors='coerce')
    df = df.sort_values(['model', 'n_samples'])

    p = p9.ggplot(
        df,
        p9.aes(x='n_samples', y='score', color='model', group='model'),
    )

    if SHOW_UNCERTAINTY_BANDS:
        p = p + p9.geom_ribbon(
            p9.aes(ymin='ymin', ymax='ymax', fill='model'),
            alpha=0.2,
            color=None,
        )

    p = (
            p
            + p9.geom_line(size=1.0)
            + p9.geom_point(size=1.8)
            + p9.scale_x_log10()
            + p9.labs(
        x='Number of sampled configs',
        y='MASE',
        color='Model',
        fill='Model',
    )
            + THEME
            + p9.theme(legend_position='top')
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p.save(OUTPUT_DIR / f'rs_lc,{target}.pdf', width=8, height=6)


for target in DATASETS:
    print(f'\n=== {target} ===')
    in_set, in_set_train, seas_len = load_dataset_splits(target)
    mase_func = partial(mase, seasonality=seas_len)

    model_scores = {}
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

        scores = {}
        for s in LEARNING_CURVE:
            s_scores = []
            for _ in range(N_REPS):
                err_sample = err_inner.sample(s)
                selected_config = err_sample.idxmin()
                s_scores.append(err_outer[selected_config])

            scores[s] = s_scores

        model_scores[model] = scores

    if not model_scores:
        print(f'No results found for {target}, skipping plot.')
        continue

    scores_df, scores_long = build_scores_long(model_scores)
    print(scores_df)

    scores_long = scores_long[scores_long['n_samples'].isin(LEARNING_CURVE[2:])]
    plot_learning_curve(scores_long, target)
