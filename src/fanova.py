from typing import Literal, Optional

import pandas as pd
import optuna
from optuna.importance import FanovaImportanceEvaluator, PedAnovaImportanceEvaluator

Method = Literal['fanova', 'ped_anova']


def _build_study(
        df: pd.DataFrame,
        param_cols: list[str],
        target_col: str,
) -> optuna.Study:
    distributions = {
        col: optuna.distributions.CategoricalDistribution(df[col].unique().tolist())
        for col in param_cols
    }

    trials = [
        optuna.trial.create_trial(
            params={col: record[col] for col in param_cols},
            distributions=distributions,
            value=record[target_col],
        )
        for record in df.to_dict('records')
    ]

    study = optuna.create_study(direction='minimize')
    study.add_trials(trials)
    return study


def run_fanova(
        df: pd.DataFrame,
        target_col: str,
        max_samples: Optional[int] = None,
        aggregate: bool = True,
        method: Method = 'fanova',
        n_trees: int = 64,
        max_depth: int = 64,
        target_quantile=0.1,
        region_quantile=1.0,
) -> pd.Series:
    param_cols = [c for c in df.columns if c != target_col]

    if aggregate:
        df = (
            df.groupby(param_cols, as_index=False)[target_col]
            .mean()
        )

    if max_samples is not None:
        df = df.sample(min(max_samples, len(df)))

    study = _build_study(df, param_cols, target_col)

    if method == 'fanova':
        evaluator = FanovaImportanceEvaluator(n_trees=n_trees, max_depth=max_depth)
    elif method == 'ped_anova':
        evaluator = PedAnovaImportanceEvaluator()
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'fanova' or 'ped_anova'.")

    importance = optuna.importance.get_param_importances(study, evaluator=evaluator)

    return pd.Series(importance).sort_values(ascending=False)
