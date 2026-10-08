import numpy as np
import pytest

from heavden.monitoring.drift_metrics import js_distance, psi, psi_level

rng = np.random.default_rng(0)
BASE = rng.normal(97, 1.5, 20_000)


def test_identical_distributions_score_near_zero():
    same = rng.normal(97, 1.5, 20_000)
    assert psi(BASE, same) < 0.01
    assert js_distance(BASE, same) < 0.05


def test_bigger_shift_gives_bigger_psi():
    small = psi(BASE, BASE - 0.3)
    big = psi(BASE, BASE - 3.0)
    assert 0 < small < big
    assert psi_level(big) == "act"


def test_js_distance_is_bounded_and_grows_with_shift():
    # With baseline-quantile bins a far-away sample piles into the top open bin, so the
    # distance tops out a little below 1.
    moderate = js_distance(BASE, BASE - 1.5)
    far = js_distance(BASE, BASE + 100)
    assert 0 < moderate < far <= 1.0
    assert far > 0.8


def test_small_samples_return_nan():
    assert np.isnan(psi(BASE, BASE[:100]))
    assert psi_level(psi(BASE, BASE[:100])) == "not enough data"


def test_missing_values_are_ignored():
    with_gaps = np.concatenate([BASE, np.full(5000, np.nan)])
    assert psi(BASE, with_gaps) == pytest.approx(0, abs=1e-9)


def test_psi_does_not_explode_with_sample_size_like_ks_pvalues():
    """A tiny, clinically meaningless shift stays 'stable' however much data we have."""
    tiny = rng.normal(97.05, 1.5, 200_000)
    assert psi_level(psi(BASE, tiny)) == "stable"
