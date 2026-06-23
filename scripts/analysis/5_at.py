from pathlib import Path
from functools import partial

import pandas as pd
from modelradar.evaluate.radar import ModelRadar
from utilsforecast.losses import mase

from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.utils.reading_data import read_cv_results
from src.coseal.active_testing import active_testing_selection

CORR_SELECTION = 0.9
N_TRIALS = 5
SAFE_N_TRIALS = 15  # to ensure we actually get N_TRIALS for each... (some configs are not in some datasets)
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
    'Informer'
]

SEARCH_METHOD = 'AT'

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


def evaluate_model(model: str, scores_df: pd.DataFrame):
    scores_f, ext_scores_df = [], []
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
        err_outer_ext = radar_outer.evaluate(keep_uids=True)

        scores_sub = scores_df.query(f'dataset!="{target}"').drop(columns='dataset')

        atr_list = active_testing_selection(
            scores_df=scores_sub,
            use_ranks=True,
            max_trials=SAFE_N_TRIALS,
            corr_threshold=CORR_SELECTION,
            delta=0.01,
        )

        rs_configs = filter_configs_in_errors(
            err_inner.sample(N_TRIALS).index.tolist(), err_inner
        )

        if SEARCH_METHOD == 'AT':
            scr = err_outer_ext[best_config_after_trials(err_inner, atr_list)]
        else:
            scr = err_outer_ext[best_config_after_trials(err_inner, rs_configs)]

        scr.name = None
        scr.index = scr.index.map(lambda idx: f'{target}_{idx}')
        scr = scr.reset_index(inplace=False)
        scr.columns = ['unique_id', 'score']

        condensed_scr = {
            'Dataset': target,
            'RS': err_outer[best_config_after_trials(err_inner, rs_configs)],
            'AT': err_outer[best_config_after_trials(err_inner, atr_list)],
        }

        scores_f.append(condensed_scr)
        ext_scores_df.append(scr)

    ext_scr_df = pd.concat(ext_scores_df, axis=0).reset_index(drop=True)
    scr_df = pd.DataFrame(scores_f)

    return scr_df, ext_scr_df


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for model in MODEL_LIST:
    print(f'\n=== {model} ===')

    scores_df = build_meta_scores_df(model)

    model_final_scrs, ext_model_final_scrs = evaluate_model(model, scores_df)

    out_path = OUTPUT_DIR / f'search,{N_TRIALS},{model}.csv'
    ext_out_path = OUTPUT_DIR / f'extended_search,{N_TRIALS},{model}.csv'
    model_final_scrs.to_csv(out_path, index=False)
    ext_model_final_scrs.to_csv(ext_out_path, index=False)

    print(f'Saved {out_path}')
