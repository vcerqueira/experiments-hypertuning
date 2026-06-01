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
SAFE_N_TRIALS = 50 # to ensure we actually get 100 for each... (some configs are not in some datasets)
# final value is N_TRIALS like err_inner[at_configs].head(N_TRIALS).idxmin()

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
]

RESULTS_DIR = Path().resolve().parent / 'hypertuning-files' / 'results-all-compiled'
OUTPUT_DIR = Path('assets/results')


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


def filter_configs_in_errors(
    config_order: list,
    err: pd.Series | pd.DataFrame,
) -> list:
    """Keep configs from config_order that exist in err, preserving order."""
    available = err.index if isinstance(err, pd.Series) else err.columns
    return [c for c in config_order if c in available]


def best_config_after_trials(
    err_inner: pd.Series | pd.DataFrame,
    config_order: list,
) -> str:
    configs = filter_configs_in_errors(config_order, err_inner)[:N_TRIALS]
    return err_inner[configs].idxmin()


def build_meta_scores_df(model: str) -> pd.DataFrame | None:
    scores = []
    for i, target in enumerate(DATASETS):
        print(f'  meta scores: {target}')
        in_set, in_set_train, seas_len = load_dataset(target)
        mase_func = partial(mase, seasonality=seas_len)

        cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')
        if cv_inner is None or not config_ids:
            print(f'    skipping {target} (no inner CV results)')
            continue

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

    if not scores:
        return None

    return pd.concat(scores).set_index('unique_id')


def evaluate_model(model: str, scores_df: pd.DataFrame) -> pd.DataFrame:
    scores_f = []
    for target in DATASETS:
        print(f'  evaluate: {target}')
        in_set, in_set_train, seas_len = load_dataset(target)
        mase_func = partial(mase, seasonality=seas_len)

        cv_inner, config_ids = read_cv_results(RESULTS_DIR, model, target, 'inner')
        cv_outer, _ = read_cv_results(RESULTS_DIR, model, target, 'outer')
        if cv_inner is None or cv_outer is None or not config_ids:
            print(f'    skipping {target} (no CV results)')
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

        scores_sub = scores_df.query(f'dataset!="{target}"').drop(columns='dataset')

        at_list = active_testing_selection(
            scores_df=scores_sub,
            use_ranks=False,
            max_trials=SAFE_N_TRIALS,
            corr_threshold=CORR_SELECTION,
            delta=0.01,
        )

        atr_list = active_testing_selection(
            scores_df=scores_sub,
            use_ranks=True,
            max_trials=SAFE_N_TRIALS,
            corr_threshold=CORR_SELECTION,
            delta=0.01,
        )

        bt_list = bradley_terry_ranking(
            scores_df=scores_sub,
            max_trials=SAFE_N_TRIALS,
            corr_threshold=CORR_SELECTION,
        )

        rs_configs = filter_configs_in_errors(
            err_inner.sample(N_TRIALS).index.tolist(), err_inner
        )

        scores_f.append({
            'Dataset': target,
            'RS': err_outer[best_config_after_trials(err_inner, rs_configs)],
            'AT': err_outer[best_config_after_trials(err_inner, at_list)],
            'ATR': err_outer[best_config_after_trials(err_inner, atr_list)],
            'PL': err_outer[best_config_after_trials(err_inner, bt_list)],
        })

    return pd.DataFrame(scores_f)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for model in MODEL_LIST:
    print(f'\n=== {model} ===')

    scores_df = build_meta_scores_df(model)
    if scores_df is None:
        print(f'No meta scores for {model}, skipping.')
        continue

    model_final_scrs = evaluate_model(model, scores_df)
    if model_final_scrs.empty:
        print(f'No evaluation results for {model}, skipping save.')
        continue

    out_path = OUTPUT_DIR / f'search,{N_TRIALS},{model}.csv'
    model_final_scrs.to_csv(out_path, index=False)
    print(model_final_scrs)
    print(f'Saved {out_path}')
    print(model_final_scrs.set_index('Dataset').rank(axis=1).mean())
