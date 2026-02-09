import os
import warnings
from pathlib import Path

from neuralforecast import NeuralForecast

from src.neural.nf_arch import ModelsConfig
from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.config import N_SAMPLES, SEED, TRY_MPS
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'monash_tourism_monthly'

_, horizon, n_lags, _, _ = ChronosDataset.load_everything(target)
df, horizon, n_lags, freq, seas_len = ChronosDataset.load_everything(target, min_n_instances=2*(n_lags+horizon))
# df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

results_dir = Path('../assets/results')

# - split dataset by time
# -- estimation_train is used for hypertuning
# -- estimation_test is only used at the end to evaluate hypertuning process
in_set, _ = ChronosDataset.time_wise_split(df, horizon)

if __name__ == '__main__':
    print(results_dir.absolute())

    for model_nm in ModelsConfig.model_names:
        # model = 'NHITS'

        config_pool = NEURAL_CONFIG_POOL[model_nm]
        config_list = ConfigSampler.generate_samples(config_pool=config_pool,
                                                     num_samples=N_SAMPLES,
                                                     random_state=SEED)

        for config_sample in config_list:
            # todo stop when no of configs reaches max_samples
            print(config_sample)

            cfg_id = config_sample.pop('config_id')

            outer_fp = results_dir / f'{model_nm},{target},{cfg_id},outer.csv'
            inner_fp = results_dir / f'{model_nm},{target},{cfg_id},inner.csv'

            if outer_fp.exists():
                continue

            model = ModelsConfig.create_model_instance(model_class=model_nm,
                                                       model_config=config_sample,
                                                       horizon=horizon,
                                                       input_size=n_lags,
                                                       try_mps=TRY_MPS)

            CV_SETUP = {
                'val_size': horizon,
                'test_size': horizon,
                'step_size': 1,
                'n_windows': None,
            }

            nf_inner = NeuralForecast(models=[model], freq=freq)
            cv_inner = nf_inner.cross_validation(df=in_set, **CV_SETUP)

            nf_outer = NeuralForecast(models=[model], freq=freq)
            cv_outer = nf_outer.cross_validation(df=df, **CV_SETUP)

            cv_inner.to_csv(inner_fp, index=False)
            cv_outer.to_csv(outer_fp, index=False)
