from typing import Optional, List, Union
import hashlib
import json

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import (AutoGRU,
                                 AutoNBEATS,
                                 AutoTiDE,
                                 AutoNLinear,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoPatchTST,
                                 AutoTFT,
                                 AutoDeepNPTS,
                                 AutoDeepAR,
                                 AutoTCN,
                                 AutoDilatedRNN)

from neuralforecast.models import (GRU,
                                   KAN,
                                   NBEATS,
                                   TiDE,
                                   NLinear,
                                   MLP,
                                   LSTM,
                                   DLinear,
                                   NHITS, DeepAR,
                                   PatchTST,
                                   TFT,
                                   DeepNPTS,
                                   DeepAR,
                                   TCN,
                                   DilatedRNN)


class ModelsConfig:
    AUTO_MODEL_CLASSES = {
        'AutoKAN': AutoKAN,
        'AutoMLP': AutoMLP,
        'AutoDLinear': AutoDLinear,
        'AutoNHITS': AutoNHITS,
        'AutoDeepNPTS': AutoDeepNPTS,
        'AutoPatchTST': AutoPatchTST,
        'AutoNBEATS': AutoNBEATS,
        'AutoTiDE': AutoTiDE,
        'AutoNLinear': AutoNLinear,
        'AutoTFT': AutoTFT,
        # 'AutoGRU': AutoGRU,
        # 'AutoDeepAR': AutoDeepAR,
        # 'AutoLSTM': AutoLSTM,
        # 'AutoDilatedRNN': AutoDilatedRNN,
        # 'AutoTCN': AutoTCN,
    }

    MODEL_CLASSES = {
        'AutoKAN': KAN,
        'AutoMLP': MLP,
        'AutoDLinear': DLinear,
        'AutoNHITS': NHITS,
        'AutoDeepNPTS': DeepNPTS,
        'AutoNBEATS': NBEATS,
        'AutoTiDE': TiDE,
        'AutoNLinear': NLinear,
        'AutoTFT': TFT,
        'AutoPatchTST': PatchTST,
        'AutoGRU': GRU,
        'AutoDeepAR': DeepAR,
        'AutoLSTM': LSTM,
        'AutoDilatedRNN': DilatedRNN,
        'AutoTCN': TCN,
    }

    NEED_CPU = ['AutoGRU',
                'AutoDeepNPTS',
                'AutoPatchTST',
                'AutoDeepAR',
                'AutoLSTM',
                'AutoKAN',
                'AutoDilatedRNN',
                'AutoTCN']

    @classmethod
    def get_auto_nf_models(cls,
                           horizon: int,
                           n_samples: int,
                           try_mps: bool = True,
                           limit_epochs: bool = False,
                           limit_val_batches: Optional[int] = None):

        models = []
        for mod_name, mod in cls.AUTO_MODEL_CLASSES.items():
            if try_mps:
                if mod_name in cls.NEED_CPU:
                    mod.default_config['accelerator'] = 'cpu'
                else:
                    mod.default_config['accelerator'] = 'mps'
            else:
                mod.default_config['accelerator'] = 'cpu'

            if limit_epochs:
                mod.default_config['max_steps'] = 2

            if limit_val_batches is not None:
                mod.default_config['limit_val_batches'] = limit_val_batches

            model_instance = mod(
                h=horizon,
                num_samples=n_samples,
                alias=mod_name,
                valid_loss=MAE(),
                refit_with_val=True,
            )

            models.append(model_instance)

        return models

    @staticmethod
    def get_all_config_results(nf: NeuralForecast):

        scores = []
        for mod in nf.models:
            print(f"Model: {mod.alias}")
            for i, res in enumerate(mod.results):
                print(res)
                res.config['learning_rate'] = np.round(res.config['learning_rate'], 5)

                conf_str = {k: str(v) for k, v in res.config.items()}
                sorted_string = json.dumps(conf_str, sort_keys=True)
                hash_value = hashlib.md5(sorted_string.encode()).hexdigest()

                scores.append({
                    'model': mod.alias,
                    'config_idx': i,
                    'loss': res.metrics['loss'],
                    'config': res.config,
                    'hash_value': hash_value
                })

        return scores

    @classmethod
    def _get_best_configs_from_folds(cls, fold_scores: List) -> List:
        folds_fl = [item for sublist in fold_scores for item in sublist]

        folds_df = pd.DataFrame(folds_fl)

        folds_avg = folds_df.groupby(['model', 'hash_value']).mean(numeric_only=True)

        best_configs = folds_avg.loc[folds_avg.groupby('model')['loss'].idxmin()].reset_index()

        optim_models = []
        for idx, row in best_configs.iterrows():
            config_inst = folds_df.query("model == @row['model'] and hash_value == @row['hash_value']")
            config = config_inst.iloc[0]['config']

            opm_mod = cls.MODEL_CLASSES[row['model']](**config)

            optim_models.append(opm_mod)

        return optim_models

    @classmethod
    def get_best_configs(cls, nf: Union[NeuralForecast, List]) -> List:
        if isinstance(nf, List):
            return cls._get_best_configs_from_folds(nf)

        optim_models = []
        for mod in nf.models:
            opm_mod = cls.MODEL_CLASSES[mod.alias](**mod.results.get_best_result().config)

            optim_models.append(opm_mod)

        return optim_models
