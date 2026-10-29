"""Statistics for the central experiment (Phases 6-8): correlations, the fixed-magnitude
alignment effect, held-out regression, and cluster bootstrap intervals.

Time points within one PINN run are strongly correlated, so every uncertainty
interval treats runs (clusters), not time points, as the independent units.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson and Spearman correlation of two samples."""
    return {"pearson": float(stats.pearsonr(x, y)[0]), "spearman": float(stats.spearmanr(x, y)[0])}


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman correlation of x and y after removing (linearly, on ranks) what the control explains."""
    rx, ry, rc = (stats.rankdata(v) for v in (x, y, control))
    design = np.column_stack([np.ones_like(rc), rc])
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def alignment_effect(magnitude: np.ndarray, alignment: np.ndarray, log_target: np.ndarray, n_bins: int = 5) -> float:
    """Effect of alignment at (approximately) fixed residual magnitude.

    The samples are split into n_bins quantile bins of `magnitude`; inside each
    bin, into terciles of `alignment`. The result is the mean over bins of
    (median log_target in the top tercile) - (median log_target in the bottom
    tercile). Positive: more alignment goes with larger target.
    """
    edges = np.quantile(magnitude, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, magnitude, side="right") - 1, 0, n_bins - 1)
    differences = []
    for b in range(n_bins):
        inside = bins == b
        if inside.sum() < 6:
            continue
        a, y = alignment[inside], log_target[inside]
        low, high = np.quantile(a, [1 / 3, 2 / 3])
        if np.any(a <= low) and np.any(a >= high):
            differences.append(np.median(y[a >= high]) - np.median(y[a <= low]))
    return float(np.mean(differences)) if differences else float("nan")


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients for y ~ intercept + X; returns [intercept, slopes...]."""
    design = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def ols_predict(coefficients: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coefficients[0] + np.atleast_2d(X.T).T @ coefficients[1:]


def leave_one_group_out(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions: for each group, fit on all other groups and predict it."""
    prediction = np.empty_like(y, dtype=float)
    for group in np.unique(groups):
        held_out = groups == group
        coefficients = ols_fit(X[~held_out], y[~held_out])
        prediction[held_out] = ols_predict(coefficients, X[held_out])
    return prediction


def r_squared(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def cluster_sums(y: np.ndarray, predictions: list[np.ndarray], clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sufficient statistics for pooled R^2: columns n, sum y, sum y^2, then SSE per model."""
    ids, index = np.unique(clusters, return_inverse=True)
    columns = [np.bincount(index), np.bincount(index, weights=y), np.bincount(index, weights=y**2)]
    columns += [np.bincount(index, weights=(y - p) ** 2) for p in predictions]
    return ids, np.column_stack(columns)


def pooled_r_squared(sums: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pooled R^2 of every model from weighted cluster sums (weights = bootstrap counts)."""
    total = weights @ sums
    n, sum_y, sum_y2 = total[0], total[1], total[2]
    sst = sum_y2 - sum_y**2 / n
    return 1.0 - total[3:] / sst


def bootstrap_delta_r_squared(
    y: np.ndarray, prediction_base: np.ndarray, prediction_new: np.ndarray, clusters: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, float]:
    """Pooled R^2(new) - R^2(base) with a 95 % cluster-bootstrap interval (clusters resampled with replacement)."""
    _, sums = cluster_sums(y, [prediction_base, prediction_new], clusters)
    k = len(sums)
    rng = np.random.default_rng(seed)
    point = pooled_r_squared(sums, np.ones(k))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(float)
        r2 = pooled_r_squared(sums, weights)
        draws[b] = r2[1] - r2[0]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"r2_base": float(point[0]), "r2_new": float(point[1]), "delta": float(point[1] - point[0]),
            "ci_low": float(low), "ci_high": float(high)}


def bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of per-cluster values with a 95 % bootstrap interval (NaNs ignored)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(values, size=len(values)).mean() for _ in range(n_boot)])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(values))}
