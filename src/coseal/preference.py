import warnings

import numpy as np
import pandas as pd
import choix


def scores_to_comparison_matrix(
    scores_df: pd.DataFrame,
    lower_is_better: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate pairwise outcomes into a dense win-count matrix for choix.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Rows are samples, columns are config IDs, values are scores.
    lower_is_better : bool, default True
        If True, lower scores win comparisons.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        - Matrix where entry (i, j) is the number of times config i beat config j
        - List of config names (index corresponds to matrix rows/columns)
    """
    values = scores_df.to_numpy()
    n = values.shape[1]
    comp_mat = np.zeros((n, n), dtype=np.float64)

    for row in values:
        if lower_is_better:
            comp_mat += row[:, None] < row[None, :]
        else:
            comp_mat += row[:, None] > row[None, :]

    return comp_mat, scores_df.columns.tolist()


def _fit_bradley_terry_ilsr(
    comp_mat: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    return choix.ilsr_pairwise_dense(
        comp_mat, alpha=alpha, max_iter=max_iter, tol=tol
    )


def fit_bradley_terry(
    scores_df: pd.DataFrame,
    lower_is_better: bool = True,
    alpha: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pd.Series:
    """
    Fit Bradley-Terry model using choix's dense ILSR algorithm.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Rows are samples, columns are config IDs, values are scores.
    lower_is_better : bool, default True
        If True, lower scores win comparisons.
    alpha : float, default 0.0
        Regularization parameter for ILSR (adds alpha to each comparison count).
    max_iter : int, default 100
        Maximum ILSR iterations per attempt.
    tol : float, default 1e-8
        Convergence tolerance for ILSR.

    Returns
    -------
    pd.Series
        Log-strength parameters θ for each config.

    Raises
    ------
    RuntimeError
        If ILSR does not converge on any attempt.
    """
    comp_mat, configs = scores_to_comparison_matrix(scores_df, lower_is_better)

    attempts = [
        (alpha, max_iter),
        (max(alpha, 1e-2), max(max_iter, 500)),
        (0.1, max(max_iter, 1000)),
    ]
    seen: set[tuple[float, int]] = set()
    last_error: RuntimeError | None = None

    for alpha_try, max_iter_try in attempts:
        key = (alpha_try, max_iter_try)
        if key in seen:
            continue
        seen.add(key)
        try:
            params = _fit_bradley_terry_ilsr(
                comp_mat, alpha=alpha_try, max_iter=max_iter_try, tol=tol
            )
            return pd.Series(params, index=configs)
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        "Bradley-Terry ILSR did not converge"
    ) from last_error


def random_config_ranking(
    scores_df: pd.DataFrame,
    max_trials: int | None = None,
    corr_threshold: float | None = None,
    rng: np.random.Generator | None = None,
) -> list:
    """
    Random search order: shuffled configs, optionally skipping correlated picks.
    """
    rng = rng or np.random.default_rng()
    remaining = scores_df.columns.to_numpy().copy()
    rng.shuffle(remaining)

    selected: list = []
    selected_corr_cache: dict[str, pd.Series] = {}

    for config in remaining:
        if max_trials is not None and len(selected) >= max_trials:
            break

        if corr_threshold is not None and selected:
            if any(
                abs(scores_df[config].corr(selected_corr_cache[sel])) > corr_threshold
                for sel in selected
            ):
                continue

        selected.append(config)
        if corr_threshold is not None:
            selected_corr_cache[config] = scores_df[config]

    return selected


def bradley_terry_ranking(
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

    Uses choix's dense ILSR algorithm for efficient fitting.

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
        Falls back to :func:`random_config_ranking` if BT fitting does not converge.
    """
    try:
        theta = fit_bradley_terry(scores_df, lower_is_better, alpha=alpha)
    except RuntimeError:
        warnings.warn(
            "Bradley-Terry did not converge; falling back to random config order.",
            stacklevel=2,
        )
        return random_config_ranking(
            scores_df,
            max_trials=max_trials,
            corr_threshold=corr_threshold,
        )

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
