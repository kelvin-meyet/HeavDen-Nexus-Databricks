import mlflow
import numpy as np
import pandas as pd
import pytest
from test_scoring_gold import fitted  # noqa: F401  (fixture)
from test_train import _table

from heavden.ml import batch_scoring, tracking, train, training

NAME = "deterioration_risk_scoring_test"


@pytest.fixture
def scored(fitted):  # noqa: F811
    table, tr, model, bands = fitted
    as_of = table["prediction_ts"].max() + pd.Timedelta(hours=1)
    # escalations for the stays whose risk is highest at some hour, 2 h after that hour
    peak = table.assign(risk=model.predict_proba(table)[:, 1]).sort_values("risk").tail(40)
    peak = peak.drop_duplicates("encounter_id").head(6)
    outcomes = pd.DataFrame(
        {
            "encounter_id": peak["encounter_id"].to_numpy(),
            "event_ts": (peak["prediction_ts"] + pd.Timedelta(hours=2)).to_numpy(),
            "event_type": "rapid_response",
        }
    )
    outcomes["recorded_ts"] = outcomes["event_ts"] + pd.Timedelta(hours=6)
    outcomes = outcomes[outcomes["recorded_ts"] <= as_of]  # only recorded outcomes exist
    tables = batch_scoring.score_hours(table, outcomes, model, bands.high, "7", as_of, tr.head(500))
    return table, outcomes, as_of, bands, tables


def test_every_patient_hour_is_scored_at_the_training_threshold(scored):
    table, _, _, bands, tables = scored
    risk = tables["risk_scores"]
    assert len(risk) == len(table)
    assert set(tables) == {
        "risk_scores",
        "alerts_fact",
        "escalations_fact",
        "site_kpis_hourly",
        "new_scores",
    }
    pd.testing.assert_frame_equal(tables["new_scores"], risk)  # first run: every hour is new
    np.testing.assert_array_equal(risk["risk_band"] == "High", risk["risk"] >= bands.high)
    assert (risk["model_version"] == "7").all()


def test_alert_outcomes_wait_for_the_recording_delay(scored):
    _, _, as_of, _, tables = scored
    alerts = tables["alerts_fact"]
    final_by = as_of - batch_scoring.RECORDING_DELAY - pd.Timedelta(hours=6)
    open_ = alerts[(alerts["alert_ts"] > final_by) & alerts["hours_to_escalation"].isna()]
    closed = alerts[alerts["alert_ts"] <= final_by]
    assert len(open_) > 0 and open_["escalated_within_6h"].isna().all()
    assert closed["escalated_within_6h"].notna().all()


def test_escalations_are_matched_to_the_scores_before_them(scored):
    _, outcomes, _, _, tables = scored
    esc = tables["escalations_fact"]
    assert len(esc) == len(outcomes) > 0
    assert esc["flagged_6h_before"].dtype == bool
    kpis = tables["site_kpis_hourly"]
    assert kpis["escalations"].sum() == len(outcomes)


def test_each_hour_is_scored_once_and_keeps_its_model(fitted):  # noqa: F811
    table, tr, model, bands = fitted
    as_of = table["prediction_ts"].max() + pd.Timedelta(hours=1)
    cut = table["prediction_ts"].quantile(0.6).floor("h")
    no_outcomes = pd.DataFrame(columns=["encounter_id", "event_ts", "event_type", "recorded_ts"])
    early = table[table["prediction_ts"] < cut]
    first = batch_scoring.score_hours(
        early, no_outcomes, model, bands.high, "1", as_of, tr.head(500)
    )
    # a later run, with more hours and a different model whose alert threshold is lower
    later = batch_scoring.score_hours(
        table, no_outcomes, model, bands.high / 2, "2", as_of, tr.head(500), first["risk_scores"]
    )
    log, new = later["risk_scores"], later["new_scores"]
    assert len(log) == len(table) and not log.duplicated(batch_scoring.KEYS).any()
    assert (new["prediction_ts"] >= cut).all() and (new["model_version"] == "2").all()
    old = log[log["prediction_ts"] < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, first["risk_scores"])  # untouched: version 1, its bands
    np.testing.assert_array_equal(old["risk_band"] == "High", old["risk"] >= bands.high)
    np.testing.assert_array_equal(new["risk_band"] == "High", new["risk"] >= bands.high / 2)


def test_a_run_with_nothing_new_changes_nothing(scored, fitted):  # noqa: F811
    table, outcomes, as_of, bands, tables = scored
    model, background = fitted[2], fitted[1].head(500)
    again = batch_scoring.score_hours(
        table, outcomes, model, bands.high, "8", as_of, background, tables["risk_scores"]
    )
    assert again["new_scores"].empty
    for name in ("risk_scores", "alerts_fact", "escalations_fact", "site_kpis_hourly"):
        pd.testing.assert_frame_equal(again[name], tables[name])


def test_background_comes_from_the_training_block():
    table = _table()
    split = train.Split()
    background = batch_scoring.background_sample(train.time_split(table, split)[0])
    cut = table["prediction_ts"].min().floor("D") + pd.Timedelta(days=split.train_days)
    assert len(background) == batch_scoring.BACKGROUND_ROWS
    assert (background["prediction_ts"] < cut - split.gap).all()


@pytest.fixture
def store(tmp_path):
    tracking.use_local_store(tmp_path / "mlflow", experiment="scoring-test")
    yield
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


def test_registered_model_comes_with_its_training_threshold(store):
    table = _table()
    with pytest.raises(LookupError, match="no versions"):
        batch_scoring.load_registered_model(NAME, "latest")
    first = training.run_training(table, NAME)
    model, threshold, version = batch_scoring.load_registered_model(NAME, "champion")
    assert version == "1"
    assert threshold == pytest.approx(first.chosen.valid_metrics["threshold"])
    sample = table.head(50)
    np.testing.assert_allclose(
        model.predict_proba(sample)[:, 1], first.chosen.model.predict_proba(sample)[:, 1]
    )
    training.run_training(table, NAME)
    assert batch_scoring.load_registered_model(NAME, "latest")[2] == "2"
    assert batch_scoring.load_registered_model(NAME, "champion")[2] == "1"
    # staging and dev: the champion once there is one, the newest version until then
    assert batch_scoring.load_registered_model(NAME, "champion_or_latest")[2] == "1"
    mlflow.MlflowClient().delete_registered_model_alias(NAME, "champion")
    assert batch_scoring.load_registered_model(NAME, "champion_or_latest")[2] == "2"


def test_to_spark_turns_missing_numbers_into_nulls(spark):
    df = pd.DataFrame(
        {
            "alert_ts": pd.to_datetime(["2026-11-01 10:00", "2026-11-01 11:00"]).tz_localize("UTC"),
            "escalated_within_6h": [1.0, np.nan],
            "alert_id": ["E1@20261101T10", "E2@20261101T11"],
        }
    )
    out = batch_scoring.to_spark(spark, df)
    assert out.where("escalated_within_6h IS NULL").count() == 1
    back = out.toPandas()
    assert back["alert_ts"].iloc[0] == pd.Timestamp("2026-11-01 10:00")  # UTC session
