import numpy as np
import pandas as pd
import pytest
from test_train import T0, _table

from heavden.ml import evaluate, explain, train


def test_top_k_alerts_exactly_k_rows():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.11])
    assert evaluate.top_k_metrics(y, s, 0.2) == {"precision_at_k": 0.5, "recall_at_k": 0.5}


def test_cluster_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 20)
    y = (rng.random(len(groups)) < 0.05).astype(int)
    good = y + rng.normal(0, 0.8, len(y))
    weak = y + rng.normal(0, 3, len(y))
    reps = evaluate.cluster_bootstrap(
        y, {"good": good, "weak": weak}, {"good": 0.5, "weak": 0.5}, groups, n_boot=200
    )
    ci = evaluate.confidence_intervals(reps)
    point = evaluate.ranking_metrics(y, good)["auprc"]
    assert ci.loc["good", ("low", "auprc")] < point < ci.loc["good", ("high", "auprc")]
    diff = evaluate.paired_difference(reps, "good", "weak")
    assert diff.loc["auprc", "low"] > 0 and diff.loc["auprc", "share_model_better"] == 1


def test_slices_use_one_threshold_and_tolerate_single_class_slices():
    y = np.array([1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.9, 0.2, 0.1])
    out = evaluate.slice_metrics(y, s, 0.5, pd.Series(list("AAABBB")))
    assert out.loc["A", "alert_rate"] == pytest.approx(2 / 3)
    assert np.isnan(out.loc["B", "auroc"]) and out.loc["B", "precision_at_budget"] == 0


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
