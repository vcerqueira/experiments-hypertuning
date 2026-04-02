import pandas as pd


def active_testing_selection(
        scores_df: pd.DataFrame,
        use_ranks: bool = False,
        max_trials: int | None = None,
        corr_threshold: float | None = None,
        delta: float | None = None,
        min_win_prob: float | None = None,
        lower_is_better: bool = True,
) -> list:
    """
    Active testing selection: iteratively picks configs most likely to beat the current best.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Rows are samples (e.g. unique_id), columns are config IDs, values are error scores.
    use_ranks : bool, default False
        If True, convert scores to ranks per sample (non-parametric) before comparisons.
    max_trials : int, optional
        Maximum number of configs to select. If None, selects all.
    corr_threshold : float, optional
        If set, skip candidates whose absolute correlation with any already-selected
        config exceeds this threshold (e.g. 0.9).
    delta : float, optional
        If set, only accept a new best if its mean score improves by at least `delta`
        compared to the current best's mean.
    min_win_prob : float, optional
        Early stopping threshold. If the best candidate's win probability falls below
        this value, stop selection (no remaining config has a realistic chance).
    lower_is_better : bool, default True
        If True, lower scores are better (e.g. error metrics). If False, higher scores
        are better (e.g. accuracy).

    Returns
    -------
    list
        Ordered list of config IDs in the sequence they were selected.
    """
    df = scores_df.copy()

    if use_ranks:
        df = df.rank(axis=1, method='average', ascending=lower_is_better)

    remaining = set(df.columns)
    selected = []
    selected_corr_cache: dict[str, pd.Series] = {}

    means = df.mean()
    best_config = means.idxmin() if lower_is_better else means.idxmax()
    selected.append(best_config)
    remaining.remove(best_config)

    if corr_threshold is not None:
        selected_corr_cache[best_config] = df[best_config]

    while remaining:
        if max_trials is not None and len(selected) >= max_trials:
            break

        best_scores = df[best_config]
        candidate_cols = list(remaining)

        if corr_threshold is not None:
            filtered_cols = []
            for c in candidate_cols:
                dominated = False
                for sel in selected:
                    corr_val = df[c].corr(selected_corr_cache[sel])
                    if abs(corr_val) > corr_threshold:
                        dominated = True
                        break
                if not dominated:
                    filtered_cols.append(c)
            candidate_cols = filtered_cols

        if not candidate_cols:
            break

        candidates_df = df[candidate_cols]
        if lower_is_better:
            win_probs = (candidates_df.lt(best_scores, axis=0)).mean()
        else:
            win_probs = (candidates_df.gt(best_scores, axis=0)).mean()

        best_prob = win_probs.max()
        best_candidate = win_probs.idxmax()

        if min_win_prob is not None and best_prob < min_win_prob:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

        if corr_threshold is not None:
            selected_corr_cache[best_candidate] = df[best_candidate]

        candidate_mean = means[best_candidate]
        best_mean = means[best_config]
        if lower_is_better:
            improvement = best_mean - candidate_mean
        else:
            improvement = candidate_mean - best_mean

        threshold = delta if delta is not None else 0.0
        if improvement > threshold:
            best_config = best_candidate

    return selected
