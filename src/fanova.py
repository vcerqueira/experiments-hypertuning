from typing import Optional

import pandas as pd
import optuna
from optuna.importance import FanovaImportanceEvaluator


def run_fanova(df: pd.DataFrame,
               target_col: str,
               max_samples: Optional[int] = None,
               n_trees: int = 64,
               max_depth: int = 64) -> pd.Series:
    study = optuna.create_study(direction="minimize")

    if max_samples is not None:
        df = df.sample(max_samples)

    param_cols = [c for c in df.columns if c != target_col]

    distributions = {
        col: optuna.distributions.CategoricalDistribution(df[col].unique().tolist())
        for col in param_cols
    }

    for i, row in df.iterrows():
        trial = optuna.trial.create_trial(
            params=row[param_cols].to_dict(),
            distributions=distributions,
            value=row["error_score"],
        )
        study.add_trial(trial)

    evaluator = FanovaImportanceEvaluator(n_trees=n_trees, max_depth=max_depth)
    importance = optuna.importance.get_param_importances(study, evaluator=evaluator)

    importance = pd.Series(importance).sort_values(ascending=False)

    return importance
