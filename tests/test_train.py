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


def test_alert_budget_converts_alerts_per_nurse_shift():
    budget = evaluate.AlertBudget()
    assert budget.alerts_per_patient_hour == pytest.approx(2 / 60)
    assert budget.per_nurse_shift(alerts=10, patient_hours=300) == pytest.approx(2.0)


def _stay(flags, encounter="E1", labels=None):
    hours = pd.date_range(T0, periods=len(flags), freq="h")
    df = pd.DataFrame({"encounter_id": encounter, "prediction_ts": hours})
    df["label"] = labels if labels is not None else 0.0
    return df, np.array(flags, dtype=bool)


def test_alert_onsets_fire_on_entry_and_suppress_repeats():
    # enters at 0, leaves, re-enters at 3 (suppressed), re-enters at 7 (>= 6 h later: alerts)
    df, high = _stay([1, 1, 0, 1, 0, 0, 0, 1, 1])
    assert np.flatnonzero(evaluate.alert_onsets(df, high)).tolist() == [0, 7]
    # a gap in the hourly series (e.g. a transfer without data) is not "still High"
    gap = df.drop(index=1)
    assert evaluate.alert_onsets(gap, high[[0, 2, 3, 4, 5, 6, 7, 8]]).sum() == 2


def test_alert_onsets_keep_the_input_row_order():
    a, high_a = _stay([0, 1, 1], "A")
    b, high_b = _stay([1, 0, 0], "B")
    df = pd.concat([a.assign(high=high_a), b.assign(high=high_b)], ignore_index=True)
    shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    onset = evaluate.alert_onsets(shuffled, shuffled["high"].to_numpy())
    alerted = shuffled[onset]
    pairs = zip(alerted["encounter_id"], alerted["prediction_ts"].dt.hour, strict=True)
    assert sorted(pairs) == [
        ("A", 1),
        ("B", 0),
    ]


def test_alert_metrics_count_alerts_not_hours():
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # escalation in the hour after the last row
    df, _ = _stay([0] * 10, labels=labels)
    scores = np.array([0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.9])  # one long High period
    m = evaluate.alert_metrics(df, scores, threshold=0.5)
    assert m["alerts"] == 1 and m["alert_precision"] == 1.0  # 5 High hours, 1 alert
    assert m["escalations"] == 1 and m["escalations_flagged"] == 1.0
    assert m["median_warning_hours"] == 5 and m["high_hour_share"] == 0.5


def test_threshold_for_alert_budget_stays_within_budget():
    rng = np.random.default_rng(0)
    frames = []
    for e in range(60):
        df, _ = _stay([0] * 48, f"E{e}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    scores = rng.random(len(df))
    budget = evaluate.AlertBudget()
    thr = evaluate.threshold_for_alert_budget(df, scores, budget)
    alerts = evaluate.alert_onsets(df, scores >= thr).sum()
    allowed = budget.alerts_per_patient_hour * len(df)
    assert alerts <= allowed
    # and it's the lowest such threshold: a little lower goes over budget
    assert evaluate.alert_onsets(df, scores >= thr - 0.02).sum() > allowed


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
        assert r.valid_metrics["alerts_per_nurse_shift"] <= 2.0
    champion = train.select_champion(results, baseline)
    assert champion is not None
    assert champion.valid_metrics["auprc"] == max(
        r.valid_metrics["auprc"] for r in results if train.passes_gates(r.valid_metrics, baseline)
    )


def test_no_champion_when_nothing_beats_news2():
    baseline = {"auprc": 0.2, "escalations_flagged": 0.6}
    weak = train.CandidateResult("weak", None, {"auprc": 0.1, "escalations_flagged": 0.9})
    misses_more = train.CandidateResult("x", None, {"auprc": 0.4, "escalations_flagged": 0.5})
    assert train.select_champion([weak, misses_more], baseline) is None
