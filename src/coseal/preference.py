import numpy as np
import pandas as pd
import choix


def scores_to_comparisons(
    scores_df: pd.DataFrame,
    lower_is_better: bool = True,
) -> tuple[list[tuple[int, int]], list[str]]:
    """
    Convert a score matrix to pairwise comparisons for choix.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Rows are samples, columns are config IDs, values are scores.
    lower_is_better : bool, default True
        If True, lower scores win comparisons.

    Returns
    -------
    tuple[list[tuple[int, int]], list[str]]
        - List of (winner_idx, loser_idx) tuples
        - List of config names (index corresponds to idx in tuples)
    """
    configs = scores_df.columns.tolist()
    n = len(configs)
    comparisons = []

    values = scores_df.values
    for row in values:
        for i in range(n):
            for j in range(i + 1, n):
                if lower_is_better:
                    if row[i] < row[j]:
                        comparisons.append((i, j))
                    elif row[j] < row[i]:
                        comparisons.append((j, i))
                else:
                    if row[i] > row[j]:
                        comparisons.append((i, j))
                    elif row[j] > row[i]:
                        comparisons.append((j, i))

    return comparisons, configs


def fit_bradley_terry(
    scores_df: pd.DataFrame,
    lower_is_better: bool = True,
    alpha: float = 0.0,
) -> pd.Series:
    """
    Fit Bradley-Terry model using choix's ILSR algorithm.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Rows are samples, columns are config IDs, values are scores.
    lower_is_better : bool, default True
        If True, lower scores win comparisons.
    alpha : float, default 0.0
        Regularization parameter for ILSR (adds alpha to each comparison count).

    Returns
    -------
    pd.Series
        Log-strength parameters θ for each config.
    """
    comparisons, configs = scores_to_comparisons(scores_df, lower_is_better)
    n = len(configs)

    params = choix.ilsr_pairwise(n, comparisons, alpha=alpha)

    return pd.Series(params, index=configs)


def active_testing_bradley_terry(
    scores_df: pd.DataFrame,
    max_trials: int | None = None,
    corr_threshold: float | None = None,
    min_win_prob: float | None = None,
    lower_is_better: bool = True,
    alpha: float = 0.0,
) -> list:
    """
    Active testing selection using Bradley-Terry preference model.

    The Bradley-Terry model estimates latent "strength" parameters for each config
    based on pairwise comparisons. The probability that config i beats config j is:
        P(i > j) = π_i / (π_i + π_j)
    where π = exp(θ) and θ are the log-strength parameters.

    Uses choix's ILSR algorithm for efficient fitting.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Rows are samples, columns are config IDs, values are scores.
    max_trials : int, optional
        Maximum number of configs to select. If None, selects all.
    corr_threshold : float, optional
        If set, skip candidates whose absolute correlation with any already-selected
        config exceeds this threshold.
    min_win_prob : float, optional
        Early stopping threshold. If the best candidate's predicted win probability
        (from BT model) falls below this value, stop selection.
    lower_is_better : bool, default True
        If True, lower scores are better.
    alpha : float, default 0.0
        Regularization parameter for Bradley-Terry fitting.

    Returns
    -------
    list
        Ordered list of config IDs in the sequence they were selected.
    """
    theta = fit_bradley_terry(scores_df, lower_is_better, alpha=alpha)
    pi = np.exp(theta)

    remaining = set(scores_df.columns)
    selected = []
    selected_corr_cache: dict[str, pd.Series] = {}

    best_config = theta.idxmax()
    selected.append(best_config)
    remaining.remove(best_config)

    if corr_threshold is not None:
        selected_corr_cache[best_config] = scores_df[best_config]

    while remaining:
        if max_trials is not None and len(selected) >= max_trials:
            break

        best_pi = pi[best_config]
        candidate_list = list(remaining)

        if corr_threshold is not None:
            filtered = []
            for c in candidate_list:
                skip = False
                for sel in selected:
                    corr_val = scores_df[c].corr(selected_corr_cache[sel])
                    if abs(corr_val) > corr_threshold:
                        skip = True
                        break
                if not skip:
                    filtered.append(c)
            candidate_list = filtered

        if not candidate_list:
            break

        probs = {c: pi[c] / (pi[c] + best_pi) for c in candidate_list}
        best_candidate = max(probs, key=probs.get)
        best_prob = probs[best_candidate]

        if min_win_prob is not None and best_prob < min_win_prob:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

        if corr_threshold is not None:
            selected_corr_cache[best_candidate] = scores_df[best_candidate]

        if pi[best_candidate] > best_pi:
            best_config = best_candidate

    return selected
