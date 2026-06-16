import pandas as pd

from src.config import N_SAMPLES, SEED
from src.neural.config_pool import NEURAL_CONFIG_POOL
from src.neural.param_samples import ConfigSampler

model = 'Informer'

config_pool = NEURAL_CONFIG_POOL[model]
config_list = ConfigSampler.generate_samples(config_pool=config_pool,
                                             num_samples=N_SAMPLES,
                                             remove_bad_configs=False,
                                             random_state=SEED)

df = pd.DataFrame(config_list)
df_badids = df.loc[df['config_id'].isin(ConfigSampler.BAD_CONFIGS)]

print(df_badids)
print(df_badids.dtypes)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_badids['scaler_type'].value_counts()
df_badids['encoder_bias'].value_counts()
df_badids['recurrent'].value_counts()
df_badids['start_padding_enabled'].value_counts()
df_badids['rnn_type'].value_counts()
df_badids['grn_activation'].value_counts()

df_badids.describe()
# encoder_bias=True