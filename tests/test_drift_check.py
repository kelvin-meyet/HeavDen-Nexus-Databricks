import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from heavden.monitoring import drift_check as dc

T0 = pd.Timestamp("2026-11-01", tz="UTC")
SITES = ("SITE_A", "SITE_B", "SITE_C")


def _hours(stays: int, hours: int, seed: int, start=T0, spo2_shift: dict | None = None):
    """Patient-hours at three sites with every monitored feature, plus a `risk`."""
    rng = np.random.default_rng(seed)
    df = pd.MultiIndex.from_product(
        [[f"E{seed}-{i}" for i in range(stays)], pd.date_range(start, periods=hours, freq="h")],
        names=["encounter_id", "prediction_ts"],
    ).to_frame(index=False)
    df["site_id"] = np.array(SITES)[df["encounter_id"].str.split("-").str[1].astype(int) % 3]
    for f in dc.FEATURES:
        df[f] = rng.normal(50, 10, len(df))
    for site, shift in (spo2_shift or {}).items():
        df.loc[df["site_id"] == site, "spo2_mean_1h"] += shift
    df["risk"] = 1 / (1 + np.exp(-(rng.normal(-5, 1, len(df)) - (df["spo2_mean_1h"] - 50) / 10)))
    return df


def _by(rows: list[dict], signal: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df[df["signal"] == signal].set_index(["feature", "site_id"])


def test_a_shift_at_one_site_is_caught_at_that_site():
    baseline = _hours(90, 48, seed=1)
    current = _hours(90, 24, seed=2, start=T0 + pd.Timedelta(days=3), spo2_shift={"SITE_B": -12})
    psi = _by(dc.input_drift(baseline, current), "feature_psi")
    assert psi.loc[("spo2_mean_1h", "SITE_B"), "level"] == "act"
    others = psi.drop(("spo2_mean_1h", "SITE_B")).drop("spo2_mean_1h", level="feature")
    assert (others["level"] == "stable").all()
    js = pd.DataFrame(dc.prediction_drift(baseline, current)).set_index("site_id")
    assert js.loc["SITE_B", "level"] == "act" and js.loc["SITE_A", "level"] == "stable"


def test_slices_too_small_to_judge_say_so():
    baseline = _hours(90, 48, seed=1)
    current = _hours(9, 24, seed=2)  # 3 stays x 24 hours per site
    psi = pd.DataFrame(dc.input_drift(baseline, current))
    assert (psi.loc[psi["site_id"] != dc.ALL_SITES, "level"] == "not enough data").all()


def test_alert_load_counts_stays_entering_high_in_the_window():
    log = _hours(60, 24, seed=3)
    hour = log["prediction_ts"].dt.hour
    log["risk_band"] = np.where(hour.isin([0, 1, 12]), "High", "Low")  # alerts at 0 and 12 h
    rows = pd.DataFrame(dc.alert_load(log, T0)).set_index("site_id")
    # 120 alerts over 1,440 patient-hours, 5 patients per nurse, 12-hour shifts
    assert rows.loc[dc.ALL_SITES, "value"] == pytest.approx(120 / 1440 * 60)
    assert rows.loc[dc.ALL_SITES, "level"] == "act"
    later = pd.DataFrame(dc.alert_load(log, T0 + pd.Timedelta(hours=1))).set_index("site_id")
    assert later.loc[dc.ALL_SITES, "value"] == pytest.approx(60 / 1380 * 60)  # 1 h: still High


def _labelled(seed: int, signal: float) -> pd.DataFrame:
    """Two days of stays; 30 of them escalate (a 6-hour run of positive hours). `signal` sets
    how much higher their risk is."""
    df = _hours(150, 48, seed=seed)[["encounter_id", "prediction_ts", "site_id"]]
    rng = np.random.default_rng(seed)
    escalated = df["encounter_id"].isin(df["encounter_id"].unique()[:30])
    df["label"] = (escalated & (df["prediction_ts"] >= T0 + pd.Timedelta(hours=30))).astype(float)
    df["label"] = df["label"].where(df["prediction_ts"] < T0 + pd.Timedelta(hours=36), 0.0)
    df["risk"] = 1 / (1 + np.exp(-(rng.normal(-5, 1, len(df)) + signal * df["label"])))
    return df


def test_performance_judges_only_a_drop_beyond_the_noise():
    same = _labelled(4, signal=3.0)
    reference = {
        "auprc": average_precision_score(same["label"], same["risk"]),
        "brier": brier_score_loss(same["label"], same["risk"]),
    }
    normal = pd.DataFrame(dc.performance(_labelled(5, signal=3.0), reference))
    assert (normal["level"] == "stable").all() and (normal["n"] == 30).all()
    worse = pd.DataFrame(dc.performance(_labelled(6, signal=0.5), reference)).set_index("signal")
    assert worse.loc["auprc_drop", "level"] == "act"
    assert worse.loc["auprc_drop", "observed"] < reference["auprc"]
    few = _labelled(7, signal=3.0)
    few.loc[few["encounter_id"].isin(few["encounter_id"].unique()[5:30]), "label"] = 0.0
    assert (pd.DataFrame(dc.performance(few, reference))["level"] == "not enough data").all()


def test_a_dropped_field_and_quarantined_readings_are_flagged():
    fields = ["spo2", "motion", "etco2"]
    quality = dc.ReadingQuality(
        readings=1000,
        quarantined=60,
        missing_now=pd.Series([0.0, 1.0, 0.0], index=fields),  # motion dropped, etco2 added
        missing_in_training=pd.Series([0.0, 0.0, 1.0], index=fields),
    )
    rows = pd.DataFrame(dc.data_quality(quality)).set_index(["signal", "feature"])
    assert rows.loc[("quarantine_rate", None), "level"] == "act"  # 60 of 1,060
    assert rows.loc[("missing_rate_change", "spo2"), "level"] == "stable"
    assert rows.loc[("missing_rate_change", "motion"), "level"] == "act"
    assert rows.loc[("missing_rate_change", "etco2"), "level"] == "act"


class _SpO2Model:
    """Risk falls as SpO2 rises, like the real model's strongest factor."""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = 1 / (1 + np.exp(5 + (df["spo2_mean_1h"].to_numpy() - 50) / 10))
        return np.column_stack([1 - p, p])


def test_the_check_scores_with_the_model_it_judges_and_reports_the_worst_first():
    baseline = _hours(90, 48, seed=1)
    features = _hours(90, 24, seed=2, start=T0 + pd.Timedelta(days=3), spo2_shift={"SITE_B": -12})
    # the log was scored by an older model whose scores have nothing to do with these
    log = features.assign(risk=0.5, risk_band="Low")
    training_set = _hours(90, 48, seed=3, start=T0 + pd.Timedelta(days=2))
    training_set["label"] = 0.0
    quality = dc.ReadingQuality(1000, 0, pd.Series({"spo2": 0.0}), pd.Series({"spo2": 0.0}))
    metrics = dc.run_check(
        _SpO2Model(), features, log, training_set, baseline, {"auprc": 0.5, "brier": 0.01}, quality
    )
    assert set(metrics["signal"]) == set(dc.THRESHOLDS)
    js = metrics[metrics["signal"] == "prediction_js"].set_index("site_id")
    assert js.loc["SITE_A", "level"] == "stable"  # the log's 0.5s would be "act" everywhere
    assert js.loc["SITE_B", "level"] == "act"
    result = dc.summary(metrics)
    assert result["status"] == "act" and result["breaches"][0].endswith("(act)")
    assert "feature_psi spo2_mean_1h at SITE_B" in " ".join(result["breaches"])
    # no escalations in the labelled window: performance can't be judged
    assert (metrics.loc[metrics["signal"] == "auprc_drop", "level"] == "not enough data").all()
