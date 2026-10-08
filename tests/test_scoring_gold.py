import json

import numpy as np
import pandas as pd
import pytest

from heavden.analytics import gold
from heavden.ml import evaluate, explain, features, news2, scoring, train

T0 = pd.Timestamp("2026-11-01", tz="UTC")
STATS = ("mean_1h", "median_1h", "min_1h", "max_1h", "mean_3h", "mean_6h", "trend_3h")
NORMAL = {"heart_rate": 80, "resp_rate": 16, "spo2": 96, "temp_c": 37.0, "sbp": 125, "dbp": 75}
SPREAD = {"heart_rate": 10, "resp_rate": 3, "spo2": 2, "temp_c": 0.4, "sbp": 12, "dbp": 8}


def full_table(n_patients: int = 40, days: int = 6, seed: int = 0) -> pd.DataFrame:
    """A synthetic patient-hour table with every gold feature column; low SpO2 and fast
    breathing drive escalation."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(T0 + pd.Timedelta(hours=1), periods=days * 24, freq="h")
    df = pd.MultiIndex.from_product(
        [[f"E-{i:03d}" for i in range(n_patients)], hours], names=["encounter_id", "prediction_ts"]
    ).to_frame(index=False)
    n = len(df)
    df["patient_id"] = "P" + df["encounter_id"]
    df["site_id"] = np.where(df["encounter_id"].str[-1].astype(int) % 2, "SITE_A", "SITE_B")
    df["unit_type"] = "general"
    df["unit_id"] = df["site_id"] + "-GENERAL"
    for v, mu in NORMAL.items():
        level = rng.normal(mu, SPREAD[v], n)
        for stat in STATS:
            df[f"{v}_{stat}"] = 0.0 if stat == "trend_3h" else level
    df["readings_1h"], df["missing_1h"], df["missing_6h"] = 12.0, 0.0, 0.0
    df["hours_since_admission"] = df.groupby("encounter_id").cumcount() + 1.0
    df["age"], df["sex"] = 70, np.where(rng.random(n) < 0.5, "F", "M")
    for flag in features.PROFILE_FLAGS:
        df[flag] = False
    df["n_conditions"] = 1
    df["acvpu"], df["on_oxygen"], df["o2_flow_lpm"] = "A", False, 0.0
    df["hours_since_obs"], df["new_confusion"] = 2.0, False
    df = pd.concat([df, news2.news2(df, suffix="_median_1h")], axis=1)
    risk = -6 + 0.9 * (96 - df["spo2_min_1h"]) + 0.5 * (df["resp_rate_mean_1h"] - 16)
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(float)
    df["label_known_at"] = df["prediction_ts"] + pd.Timedelta(hours=6)
    return df


@pytest.fixture(scope="module")
def fitted():
    table = full_table()
    cut1, cut2 = T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=5)
    tr = table[table["prediction_ts"] < cut1]
    va = table[(table["prediction_ts"] >= cut1) & (table["prediction_ts"] < cut2)]
    results, _ = train.train_candidates(tr, va)
    model = next(r for r in results if r.name == "logistic_regression").model
    bands = scoring.RiskBands.fit(va, model.predict_proba(va)[:, 1])
    return table, tr, model, bands


def test_calibrated_probabilities_never_reach_0_or_1(fitted):
    table, _, model, _ = fitted
    extreme = table.head(2).copy()
    extreme["spo2_min_1h"] = [50, 100]
    extreme["resp_rate_mean_1h"] = [60, 4]
    p = model.predict_proba(pd.concat([table, extreme]))[:, 1]
    assert 0 < p.min() and p.max() < 1


def test_risk_bands_follow_the_alert_budget(fitted):
    table, _, model, bands = fitted
    va = table[(table["prediction_ts"] >= T0 + pd.Timedelta(days=3))]
    va = va[va["prediction_ts"] < T0 + pd.Timedelta(days=5)]
    p = model.predict_proba(va)[:, 1]
    budget = evaluate.AlertBudget()
    alerts = evaluate.alert_onsets(va, p >= bands.high).sum()
    assert budget.per_nurse_shift(alerts, len(va)) <= budget.alerts_per_nurse_per_shift
    assert bands.medium <= bands.high


def test_bands_classify_by_cut_off():
    bands = scoring.RiskBands(medium=0.01, high=0.05)
    assert list(bands.band([0.05, 0.01, 0.009])) == ["High", "Medium", "Low"]


def test_vectorised_top_factors_match_the_single_row_version(fitted):
    table, tr, model, _ = fitted
    rows = table.sample(20, random_state=1)
    expl = explain.explain(model, rows, background=tr)
    batch = scoring.top_factors_table(expl, rows, k=3)
    for i, label in enumerate(rows.index):
        assert batch[i] == explain.top_factors(expl, rows, label, k=3)


def test_score_table_has_one_row_per_patient_hour(fitted):
    table, tr, model, bands = fitted
    scored = scoring.score_table(model, table.head(100), bands, tr, model_version="7")
    assert len(scored) == 100 and set(scored["risk_band"]) <= set(scoring.BANDS)
    assert len(json.loads(scored["top_factors"].iloc[0])) == 3
    assert (scored["model_version"] == "7").all()


def test_what_if_moves_the_latest_hour_and_recomputes_news2(fitted):
    table, tr, model, bands = fitted
    row = table.iloc[-1]
    new = scoring.apply_changes(row, {"spo2": 85, "acvpu": "C", "on_oxygen": True})
    delta = 85 - row["spo2_mean_1h"]
    assert new["spo2_median_1h"] == new["spo2_min_1h"] == 85
    assert new["spo2_mean_6h"] == pytest.approx(row["spo2_mean_6h"] + delta / 6)
    assert new["spo2_trend_3h"] == pytest.approx(row["spo2_trend_3h"] + delta)
    assert new["news2_spo2"] == 3 and new["news2_consciousness"] == 3 and new["news2_oxygen"] == 2
    assert new["new_confusion"] and new["o2_flow_lpm"] > 0

    result = scoring.what_if(model, row, {"spo2": 85, "resp_rate": 30}, bands, tr)
    assert result["risk_after"] > result["risk_before"]
    assert result["news2_after"] > result["news2_before"]
    assert result["top_factors"][0]["direction"] == "raises risk"


def test_what_if_rejects_unknown_fields(fitted):
    row = fitted[0].iloc[0]
    with pytest.raises(ValueError, match="cannot change"):
        scoring.apply_changes(row, {"age": 20})
    with pytest.raises(ValueError, match="acvpu"):
        scoring.apply_changes(row, {"acvpu": "X"})


def _risk_rows(bands_by_hour: list[str], encounter="E1") -> pd.DataFrame:
    hours = pd.date_range(T0, periods=len(bands_by_hour), freq="h")
    return pd.DataFrame(
        {
            "encounter_id": encounter,
            "patient_id": "P1",
            "prediction_ts": hours,
            "site_id": "SITE_A",
            "unit_id": "SITE_A-GENERAL",
            "risk": [0.5 if b == "High" else 0.001 for b in bands_by_hour],
            "risk_band": bands_by_hour,
            "news2_total": 1.0,
            "top_factors": "[]",
        }
    )


def test_alerts_fire_on_entering_high_and_suppress_repeats():
    # High at 0-1, drops, High again at 3 (suppressed: within 6 h), High again at 10 (new alert)
    bands = ["High", "High", "Low", "High", "Low", "Low", "Low", "Low", "Low", "Low", "High"]
    risk = _risk_rows(bands)
    outcomes = pd.DataFrame(
        {"encounter_id": ["E1"], "event_ts": [T0 + pd.Timedelta(hours=12, minutes=30)]}
    )
    alerts = gold.alerts_fact(risk, outcomes, as_of=T0 + pd.Timedelta(hours=10))
    assert list(alerts["alert_ts"]) == [T0, T0 + pd.Timedelta(hours=10)]
    assert alerts["escalated_within_6h"].iloc[0] == 0  # window closed, no event within 6 h
    assert alerts["escalated_within_6h"].iloc[1] == 1  # event 2.5 h later is already known
    assert alerts["hours_to_escalation"].iloc[1] == pytest.approx(2.5)


def test_site_kpis_count_bands_alerts_and_escalations():
    risk = pd.concat([_risk_rows(["Low", "High"], "E1"), _risk_rows(["Medium", "Medium"], "E2")])
    outcomes = pd.DataFrame(
        {"encounter_id": ["E2"], "event_ts": [T0 + pd.Timedelta(hours=1, minutes=20)]}
    )
    alerts = gold.alerts_fact(risk, outcomes, as_of=T0 + pd.Timedelta(hours=1))
    kpis = gold.site_kpis_hourly(risk, alerts, outcomes).set_index("hour_ts")
    second = kpis.loc[T0 + pd.Timedelta(hours=1)]
    assert (second["census"], second["n_high"], second["n_medium"]) == (2, 1, 1)
    assert second["alerts_raised"] == 1 and second["escalations"] == 1


def test_device_health_counts_outages_and_stuck_sensors():
    ts = pd.date_range(T0, periods=288, freq="5min")
    keep = np.ones(288, dtype=bool)
    keep[100:110] = False  # a 50-minute outage
    rng = np.random.default_rng(0)
    readings = pd.DataFrame(
        {
            "device_id": "DEV-A-0001",
            "ts": ts[keep],
            **{v: rng.normal(80, 5, keep.sum()).round() for v in gold.STUCK_VITALS},
            "spo2": 96.0,
            "battery_pct": np.linspace(100, 60, keep.sum()).round(),
            "firmware": "3.1.4",
        }
    )
    readings.loc[200:207, "heart_rate"] = 77.0  # frozen for 8 readings = 40 minutes
    assignments = pd.DataFrame(
        {"device_id": ["DEV-A-0001"], "start_ts": [T0], "end_ts": [T0 + pd.Timedelta(days=1)]}
    )
    day = gold.device_health_daily(readings, assignments, T0, T0 + pd.Timedelta(days=1)).iloc[0]
    assert day["site_id"] == "SITE_A" and day["messages_expected"] == 288
    assert day["messages_received"] == 278 and day["battery_outages"] == 1
    assert day["uptime_pct"] == pytest.approx(100 * 278 / 288)
    assert day["stuck_minutes"] == 40


def test_escalations_fact_records_whether_the_patient_was_flagged():
    risk = pd.concat(
        [
            _risk_rows(["Low", "Low", "High", "High"], "E1"),  # flagged 2 h before the event
            _risk_rows(["Low", "Low", "Low", "Low"], "E2"),  # never flagged
        ]
    )
    event = T0 + pd.Timedelta(hours=4)
    outcomes = pd.DataFrame(
        {
            "encounter_id": ["E1", "E2", "E3"],  # E3 has no scores: skipped
            "event_ts": [event, event, event],
            "event_type": ["rapid_response", "icu_transfer", "rapid_response"],
        }
    )
    esc = gold.escalations_fact(risk, outcomes, as_of=event).set_index("encounter_id")
    assert list(esc.index) == ["E1", "E2"]
    assert esc.loc["E1", "flagged_6h_before"] and esc.loc["E1", "hours_flagged_before"] == 2
    assert not esc.loc["E2", "flagged_6h_before"] and np.isnan(
        esc.loc["E2", "hours_flagged_before"]
    )
