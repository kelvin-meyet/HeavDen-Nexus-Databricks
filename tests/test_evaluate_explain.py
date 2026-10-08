import numpy as np
import pandas as pd
import pytest
from test_train import T0, _table

from heavden.ml import evaluate, explain, train


def test_top_k_alerts_exactly_k_rows():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.11])
    assert evaluate.top_k_metrics(y, s, 0.2) == {"precision_at_k": 0.5, "recall_at_k": 0.5}


def _ward(n_stays=200, hours=20, seed=0):
    """Stays with one escalation each in ~15% of them; a good and a weak score."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_stays):
        t = pd.date_range(T0, periods=hours, freq="h")
        label = np.zeros(hours)
        if rng.random() < 0.15:
            label[-6:] = 1  # escalation in the hour after the last row
        rows.append(pd.DataFrame({"encounter_id": f"E{e}", "prediction_ts": t, "label": label}))
    df = pd.concat(rows, ignore_index=True)
    y = df["label"].to_numpy()
    good = y + rng.normal(0, 0.6, len(y))
    weak = y + rng.normal(0, 3, len(y))
    return df, good, weak


def test_cluster_bootstrap_brackets_the_point_estimate():
    df, good, weak = _ward()
    thresholds = {
        "good": evaluate.threshold_for_alert_budget(df, good),
        "weak": evaluate.threshold_for_alert_budget(df, weak),
    }
    reps = evaluate.cluster_bootstrap(df, {"good": good, "weak": weak}, thresholds, n_boot=200)
    ci = evaluate.confidence_intervals(reps)
    point = evaluate.evaluate(df, good, thresholds["good"], probabilities=False)
    for metric in ("auprc", "alert_precision", "escalations_flagged"):
        assert ci.loc["good", ("low", metric)] <= point[metric] <= ci.loc["good", ("high", metric)]
    diff = evaluate.paired_difference(reps, "good", "weak")
    assert diff.loc["auprc", "low"] > 0 and diff.loc["auprc", "share_model_better"] == 1


def test_slices_use_one_threshold():
    df, good, _ = _ward(n_stays=40)
    by = pd.Series(np.where(df["encounter_id"].str[1:].astype(int) < 20, "A", "B"))
    thr = evaluate.threshold_for_alert_budget(df, good)
    out = evaluate.slice_metrics(df, good, thr, by)
    whole = evaluate.alert_metrics(df, good, thr)
    assert out["patient_hours"].sum() == len(df)
    assert out["escalations"].sum() == whole["escalations"]
    weights = out["patient_hours"] / len(df)
    assert (out["alerts_per_nurse_shift"] * weights).sum() == pytest.approx(
        whole["alerts_per_nurse_shift"]
    )


def test_age_bands():
    assert evaluate.age_band(pd.Series([18, 49, 50, 84, 85, 94])).tolist() == [
        "18-49",
        "18-49",
        "50-69",
        "70-84",
        "85+",
        "85+",
    ]


def test_warning_time_counts_hours_before_the_event():
    hours = pd.date_range(T0, periods=10, freq="h")
    df = pd.DataFrame({"encounter_id": "E1", "prediction_ts": hours})
    df["label"] = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  # event during the hour after row 7
    scores = np.zeros(10)
    scores[4] = 1  # first alert at row 4 -> rows 4..7 -> 4 hours of warning
    w = evaluate.warning_times(df, scores, threshold=0.5)
    assert len(w) == 1 and w.loc[0, "caught"] and w.loc[0, "lead_hours"] == 4
    missed = evaluate.warning_times(df, np.zeros(10), threshold=0.5)
    assert not missed.loc[0, "caught"] and np.isnan(missed.loc[0, "lead_hours"])


@pytest.fixture(scope="module")
def candidates():
    tr, va, te = train.time_split(_table())
    results, _ = train.train_candidates(tr, va)
    return tr, te, results


def test_contributions_add_up_to_the_raw_log_odds(candidates):
    tr, te, results = candidates
    rows = te.head(200)
    for r in results:
        e = explain.explain(r.model, rows, background=tr)
        total = e.base_value + e.contributions.sum(axis=1)
        np.testing.assert_allclose(total, explain.raw_log_odds(r.model, rows), atol=1e-6)


def test_top_factors_name_the_signal(candidates):
    tr, te, results = candidates
    lr = next(r for r in results if r.name == "logistic_regression")
    rows = te.head(500)
    e = explain.explain(lr.model, rows, background=tr)
    sickest = rows["spo2_min_1h"].idxmin()  # very low SpO2 drives risk in the test table
    factors = explain.top_factors(e, rows, sickest, k=2)
    assert factors[0]["factor"] == "oxygen saturation"
    assert factors[0]["direction"] == "raises risk"
    assert factors[0]["driver"] == "SpO2 (lowest, last hour)"


def test_feature_names_and_groups():
    assert explain.describe_feature("resp_rate_trend_3h") == "breathing rate (change over 3 h)"
    assert explain.feature_group("news2_spo2") == "oxygen saturation"
    assert explain.feature_group("news2_total") == "NEWS2 total"
    assert explain.feature_group("age") == explain.BACKGROUND
