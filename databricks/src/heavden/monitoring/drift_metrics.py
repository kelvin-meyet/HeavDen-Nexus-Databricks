"""Distribution-shift metrics for monitoring (Plan.md §10).

PSI and Jensen-Shannon distance compare a *baseline* sample (e.g. the champion's training data)
with a *current* window, using bins fixed from the baseline. Unlike KS-test p-values, they
measure **how big** a shift is, so they don't fire on trivial differences just because the
sample is large. Both return NaN when either sample is too small to judge.
"""

from __future__ import annotations

import numpy as np

MIN_SAMPLES = 500
_EPS = 1e-4  # keeps empty bins from producing log(0)

# Conventional PSI reading: < 0.1 stable, 0.1-0.25 moderate shift (warn), > 0.25 major (act).
PSI_WARN, PSI_ACT = 0.10, 0.25


def baseline_bins(baseline: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin edges at the baseline's quantiles, open-ended at both ends."""
    baseline = np.asarray(baseline, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    inner = np.unique(np.quantile(baseline, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], inner, [np.inf]])


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of non-missing values falling in each bin."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(counts.sum(), 1)


def _shares(baseline, current, n_bins, min_samples):
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    if min(np.sum(~np.isnan(baseline)), np.sum(~np.isnan(current))) < min_samples:
        return None
    edges = baseline_bins(baseline, n_bins)
    expected = np.clip(bin_shares(baseline, edges), _EPS, None)
    actual = np.clip(bin_shares(current, edges), _EPS, None)
    return expected / expected.sum(), actual / actual.sum()


def psi(
    baseline: np.ndarray, current: np.ndarray, n_bins: int = 10, min_samples: int = MIN_SAMPLES
) -> float:
    """Population Stability Index: sum over bins of (actual - expected) * ln(actual / expected)."""
    shares = _shares(baseline, current, n_bins, min_samples)
    if shares is None:
        return float("nan")
    expected, actual = shares
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def js_distance(
    baseline: np.ndarray, current: np.ndarray, n_bins: int = 10, min_samples: int = MIN_SAMPLES
) -> float:
    """Jensen-Shannon distance (base 2): 0 = identical distributions, 1 = no overlap."""
    shares = _shares(baseline, current, n_bins, min_samples)
    if shares is None:
        return float("nan")
    p, q = shares
    m = (p + q) / 2
    divergence = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    return float(np.sqrt(max(divergence, 0.0)))


def psi_level(value: float) -> str:
    if np.isnan(value):
        return "not enough data"
    if value >= PSI_ACT:
        return "act"
    if value >= PSI_WARN:
        return "warn"
    return "stable"
