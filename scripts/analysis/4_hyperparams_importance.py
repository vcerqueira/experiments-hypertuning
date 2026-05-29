import math
from pathlib import Path
from functools import partial

import pandas as pd
import plotnine as p9
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.utils.reading_data import read_cv_results
from src.config import N_SAMPLES, SEED
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler
from src.fanova import run_fanova

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

LR_LOG_CENTERS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
LR_LABELS = [f'1e{exp:g}' for exp in LR_LOG_CENTERS]
RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'
OUTPUT_DIR = Path('assets/outputs')


def load_dataset(target):
    if target in ChronosDataset.FREQUENCY_MAP_DATASETS:
        df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target)
    else:
        df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(
            target, resample_to='D'
        )

    in_set, _ = ChronosDataset.time_wise_split(df, horizon)
    return in_set, seas_len


def bin_learning_rate(lr):
    center_idx = min(
        range(len(LR_LOG_CENTERS)),
        key=lambda i: abs(math.log10(lr) - LR_LOG_CENTERS[i]),
    )
    return LR_LABELS[center_idx]


def build_err_with_config(model, target, in_set, mase_func):
    cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')
    if cv_inner is None:
        return None

    radar = ModelRadar(
        cv_df=cv_inner,
        metrics=[mase_func],
        model_names=config_ids,
        train_df=in_set,
        hardness_reference=config_ids[0],
        ratios_reference=config_ids[0],
    )

    err_df = radar.evaluate(keep_uids=True)

    config_pool = NEURAL_CONFIG_POOL[model]
    config_list = ConfigSampler.generate_samples(
        config_pool=config_pool,
        num_samples=N_SAMPLES,
        random_state=SEED,
    )
    config_df = pd.DataFrame(config_list)

    err_long = (
        err_df.rename_axis('unique_id')
        .reset_index()
        .melt(id_vars='unique_id', var_name='config_id', value_name='error_score')
    )

    err_with_config = err_long.merge(config_df, on='config_id', how='left')
    err_with_config['learning_rate_bin'] = err_with_config['learning_rate'].map(bin_learning_rate)
    err_with_config.drop(columns=['unique_id', 'config_id', 'learning_rate'], inplace=True)

    input_columns = err_with_config.drop(columns=['error_score']).columns.tolist()
    err_with_config[input_columns] = err_with_config[input_columns].astype(str)

    return err_with_config


def plot_model_importance(model_df, model):
    model_df = model_df.sort_values('importance', ascending=True)
    model_df = model_df.assign(
        hyperparameter=pd.Categorical(
            model_df['hyperparameter'],
            categories=model_df['hyperparameter'].tolist(),
            ordered=True,
        ),
        ymin=model_df['importance'] - model_df['std'],
        ymax=model_df['importance'] + model_df['std'],
    )

    p = (
        p9.ggplot(model_df, p9.aes(x='hyperparameter', y='importance'))
        + p9.geom_col(fill='#008fd5', width=0.7)
        + p9.geom_errorbar(
            p9.aes(ymin='ymin', ymax='ymax'),
            width=0.2,
        )
        + p9.coord_flip()
        + p9.labs(
            x='Hyperparameter',
            y='Average importance',
            title=model,
        )
        + p9.theme_538()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p.save(OUTPUT_DIR / f'hp_importance,{model}.pdf', width=8, height=6)


task_records = []
for model in MODEL_LIST:
    print(f'\n=== {model} ===')

    for target in DATASETS:
        print(target)
        in_set, seas_len = load_dataset(target)
        mase_func = partial(mase, seasonality=seas_len)

        err_with_config = build_err_with_config(model, target, in_set, mase_func)
        if err_with_config is None:
            continue

        importance = run_fanova(
            df=err_with_config,
            target_col='error_score',
            method='ped_anova',
        )

        for hyperparameter, score in importance.items():
            task_records.append({
                'model': model,
                'target': target,
                'hyperparameter': hyperparameter,
                'importance': score,
            })

importance_by_task = pd.DataFrame(task_records)

importance_long = (
    importance_by_task
    .groupby(['model', 'hyperparameter'], as_index=False)
    .agg(importance=('importance', 'mean'), std=('importance', 'std'))
)
importance_long['std'] = importance_long['std'].fillna(0)

print(importance_long)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
importance_long.to_csv(OUTPUT_DIR / 'hp_importance_summary.csv', index=False)

for model in MODEL_LIST:
    model_df = importance_long[importance_long['model'] == model]
    if model_df.empty:
        print(f'No importance results for {model}, skipping plot.')
        continue

    plot_model_importance(model_df, model)
