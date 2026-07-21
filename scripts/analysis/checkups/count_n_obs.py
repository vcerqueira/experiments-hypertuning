import os
import warnings

from src.loaders import ChronosDataset, LongHorizonDatasetR

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'TrafficL'

_, horizon, n_lags, _, _ = LongHorizonDatasetR.load_everything(target, resample_to='D')
df, horizon, n_lags, freq, seas_len = LongHorizonDatasetR.load_everything(target,
                                                                          min_n_instances=2 * (n_lags + horizon),
                                                                          resample_to='D')

print(df)
