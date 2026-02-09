import os
import warnings
from pathlib import Path

from src.neural.nf_arch import ModelsConfig
from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.config import N_SAMPLES, SEED, LIMIT_EPOCHS, TRY_MPS

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'monash_m1_monthly'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')


# df['unique_id'].value_counts().value_counts().sort_index()
# from pprint import pprint
# dt = ChronosDataset.get_chronos_datasets_names()
# pprint(dt)

results_dir = Path('../assets/results')

# - split dataset by time
# -- estimation_train is used for hypertuning
# -- estimation_test is only used at the end to evaluate hypertuning process
in_set, out_set = ChronosDataset.time_wise_split(df, horizon)

if __name__ == '__main__':
    print(results_dir.absolute())

    models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                             try_mps=TRY_MPS,
                                             limit_epochs=LIMIT_EPOCHS,
                                             n_samples=N_SAMPLES)

    print(f"Running cross validation for method: Time-wise Holdout")
    tw_cv, tw_cv_inner = time_wise_holdout(in_set=in_set,
                                           out_set=out_set,
                                           freq=freq,
                                           freq_int=seas_len,
                                           horizon=horizon,
                                           models=models,
                                           out_set_multiplier=OUT_SET_MULTIPLIER)

    tw_cv.to_csv(results_dir / f'{target},TimeHoldout,outer.csv', index=False)
    tw_cv_inner.to_csv(results_dir / f'{target},TimeHoldout,inner.csv', index=False)

    for method_name in CV_METHODS:
        print(f"Running cross validation for method: {method_name}")
        cv_result, cv_inner_result = run_cross_validation(cv_method=method_name,
                                                          in_set=in_set,
                                                          out_set=out_set,
                                                          freq=freq,
                                                          freq_int=seas_len,
                                                          horizon=horizon,
                                                          nf_models=models,
                                                          random_state=SEED,
                                                          out_set_multiplier=OUT_SET_MULTIPLIER)

        cv_result.to_csv(results_dir / f'{target},{method_name},outer.csv', index=False)
        cv_inner_result.to_csv(results_dir / f'{target},{method_name},inner.csv', index=False)
