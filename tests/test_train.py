import numpy as np
import pandas as pd
import pytest

from heavden.ml import evaluate, train
from heavden.ml.features import PROFILE_FLAGS

T0 = pd.Timestamp("2026-11-01", tz="UTC")


def _table(n_patients: int = 60, days: int = 14, seed: int = 0) -> pd.DataFrame:
    """A small synthetic patient-hour table where low SpO2 and fast breathing raise risk."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(T0, periods=days * 24, freq="h")
    df = pd.MultiIndex.from_product(
        [[f"E{i}" for i in range(n_patients)], hours], names=["encounter_id", "prediction_ts"]
    ).to_frame(index=False)
    n = len(df)
    df["patient_id"] = df["encounter_id"]
    df["spo2_min_1h"] = rng.normal(95, 2, n)
    df["resp_rate_mean_1h"] = rng.normal(17, 3, n)
    risk = -7 + 0.8 * (95 - df["spo2_min_1h"]) + 0.5 * (df["resp_rate_mean_1h"] - 17)
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(float)
    df["news2_total"] = (df["spo2_min_1h"] < 92) * 3 + (df["resp_rate_mean_1h"] > 24) * 3
    df["unit_type"] = rng.choice(["general", "step_down", "respiratory"], n)
    df["sex"] = rng.choice(["F", "M"], n)
    df["site_id"], df["unit_id"] = "SITE_A", "SITE_A-GENERAL"
    df["label_known_at"] = df["prediction_ts"] + pd.Timedelta(hours=6)
    for flag in PROFILE_FLAGS:
        df[flag] = False
    return df


def test_alert_budget_is_two_alerts_per_nurse_per_shift():
    assert evaluate.AlertBudget().fraction == pytest.approx(2 / 60)


def test_threshold_hits_the_budget_on_continuous_scores():
    scores = np.random.default_rng(0).random(10_000)
    thr = evaluate.threshold_for_budget(scores, 0.05)
    assert (scores >= thr).mean() == pytest.approx(0.05, abs=0.002)


def test_alert_metrics_count_caught_events():
    y = np.array([1, 1, 0, 0, 0])
    m = evaluate.alert_metrics(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3]), threshold=0.5)
    assert m == {"alert_rate": 0.4, "precision_at_budget": 0.5, "recall_at_budget": 0.5}


def test_time_split_is_ordered_with_gaps():
    tr, va, te = train.time_split(_table())
    assert tr["prediction_ts"].max() + train.GAP <= va["prediction_ts"].min()
    assert va["prediction_ts"].max() + train.GAP <= te["prediction_ts"].min()
    assert len(tr) > len(te) > 0


def test_categoricals_are_one_hot_in_a_fixed_order():
    df = _table(2, 1)
    cols = train.model_columns(df)
    assert "unit_type=respiratory" in cols and "sex=M" in cols and "unit_type" not in cols
    m = train.to_matrix(df, cols)
    assert m[[c for c in cols if c.startswith("unit_type=")]].sum(axis=1).eq(1).all()


def test_candidates_are_calibrated_and_beat_a_weak_baseline():
    tr, va, _ = train.time_split(_table())
    results, baseline = train.train_candidates(tr, va)
    assert {r.name for r in results} == {"logistic_regression", "lightgbm"}
    for r in results:
        p = r.model.predict_proba(va)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()
        # calibration keeps the ranking of the raw model
        raw = r.model.raw_scores(va)
        assert np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(raw)))[0, 1] > 0.99
        assert r.valid_metrics["alert_rate"] == pytest.approx(
            evaluate.AlertBudget().fraction, abs=0.01
        )
    champion = train.select_champion(results, baseline)
    assert champion is not None
    assert champion.valid_metrics["auprc"] == max(
        r.valid_metrics["auprc"] for r in results if train.passes_gates(r.valid_metrics, baseline)
    )


def test_no_champion_when_nothing_beats_news2():
    weak = train.CandidateResult("weak", None, {"auprc": 0.1, "precision_at_budget": 0.1})
    assert train.select_champion([weak], {"auprc": 0.2, "precision_at_budget": 0.2}) is None
