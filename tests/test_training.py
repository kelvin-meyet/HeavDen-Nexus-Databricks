import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from test_train import _table

from heavden.ml import batch_scoring, tracking, train, training

NAME = "deterioration_risk_test"


@pytest.fixture
def store(tmp_path):
    tracking.use_local_store(tmp_path / "mlflow", experiment="training-test")
    yield
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


@pytest.fixture(scope="module")
def table():
    return _table()


def test_first_model_becomes_champion_and_later_ones_only_challengers(store, table):
    first = training.run_training(table, NAME)
    assert first.passed_gates and first.version == 1
    assert first.aliases == ["challenger", "champion"]

    second = training.run_training(table, NAME)
    assert second.version == 2 and second.aliases == ["challenger"]
    client = mlflow.MlflowClient()
    assert int(client.get_model_version_by_alias(NAME, "champion").version) == 1
    assert int(client.get_model_version_by_alias(NAME, "challenger").version) == 2


def test_registered_model_loads_and_scores_like_the_trained_one(store, table):
    outcome = training.run_training(table, NAME)
    loaded = mlflow.sklearn.load_model(f"models:/{NAME}@champion")
    sample = table.head(200)
    np.testing.assert_allclose(
        loaded.predict_proba(sample)[:, 1], outcome.chosen.model.predict_proba(sample)[:, 1]
    )
    # the model carries our code and pinned libraries, so it loads without the project wheel
    model_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{NAME}@champion"))
    assert (model_dir / "code" / "heavden" / "ml" / "train.py").exists()
    requirements = (model_dir / "requirements.txt").read_text()
    assert "scikit-learn==" in requirements and "lightgbm==" in requirements


def test_runs_record_metrics_gates_and_test_scores_once(store, table):
    outcome = training.run_training(table, NAME, tags={"source_table": "test.gold.training_set"})
    runs = mlflow.search_runs(experiment_names=["training-test"])
    assert len(runs) == 2  # one run per candidate
    chosen = runs[runs["tags.chosen"] == "true"].iloc[0]
    other = runs[runs["tags.chosen"] == "false"].iloc[0]
    assert chosen["tags.mlflow.runName"] == outcome.chosen.name
    assert chosen["metrics.test_auprc"] == pytest.approx(outcome.test["auprc"])
    assert "metrics.test_auprc" in runs and np.isnan(other["metrics.test_auprc"])
    assert chosen["tags.source_table"] == "test.gold.training_set"
    assert int(chosen["tags.rows_valid"]) == outcome.rows["valid"] > 0
    version = mlflow.MlflowClient().get_model_version(NAME, str(outcome.version))
    assert version.tags["passed_gates"] == "true"
    assert "validation AUPRC" in version.description


def test_a_model_that_fails_the_gates_is_registered_without_aliases(store, table, monkeypatch):
    monkeypatch.setattr(train, "passes_gates", lambda metrics, baseline: False)
    outcome = training.run_training(table, NAME)
    assert not outcome.passed_gates and outcome.aliases == []
    assert not tracking.has_alias(NAME, "champion")
    version = mlflow.MlflowClient().get_model_version(NAME, str(outcome.version))
    assert version.tags["passed_gates"] == "false"


def test_a_split_without_validation_data_is_refused(store, table):
    with pytest.raises(ValueError, match="not enough labelled data"):
        training.run_training(table, NAME, train.Split(train_days=30, valid_days=2))


def test_model_requirements_pin_the_installed_libraries():
    pins = dict(r.split("==") for r in training.model_requirements())
    assert set(pins) == set(training.MODEL_LIBRARIES)
    assert all(pins.values())


def test_spark_timestamps_become_utc():
    import pandas as pd

    df = pd.DataFrame({"prediction_ts": pd.to_datetime(["2026-11-01 10:00"])})
    out = training.to_utc(df, ("prediction_ts",))
    assert str(out["prediction_ts"].dt.tz) == "UTC"
    assert out["prediction_ts"].iloc[0] == pd.Timestamp("2026-11-01 10:00", tz="UTC")


class _FixedModel:
    """A stand-in champion: `oracle` knows the labels, otherwise it scores at random."""

    def __init__(self, table: pd.DataFrame, oracle: bool):
        labels = table.set_index(["encounter_id", "prediction_ts"])["label"]
        guesses = np.random.default_rng(0).uniform(0.001, 0.05, len(labels))
        self.risk = labels * 0.9 + 0.05 if oracle else pd.Series(guesses, index=labels.index)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_frame(df[["encounter_id", "prediction_ts"]])
        p = self.risk.reindex(keys).to_numpy()
        return np.column_stack([1 - p, p])


def test_a_retrain_uses_the_latest_days_and_must_beat_the_champion(store, table):
    first = training.run_training(table, NAME)
    weak = training.Champion(_FixedModel(table, oracle=False), 0.5, "1")
    retrained = training.run_training(table, NAME, train.RecentSplit(2, 2), champion=weak)
    newest = table["prediction_ts"].max()
    blocks = tracking.data_blocks(NAME, str(retrained.version))
    assert blocks["test"][1] == newest
    assert blocks["test"][0] > newest - pd.Timedelta(days=2) > blocks["valid"][1]
    assert blocks["train"][0] == table["prediction_ts"].min()  # all earlier history trains
    assert retrained.vs_champion["beats_champion"] == 1.0
    assert retrained.vs_champion["auroc_better_share"] >= training.BETTER_SHARE
    assert retrained.vs_champion["flagged_at_least_share"] >= training.BETTER_SHARE
    assert retrained.passed_gates and retrained.aliases == ["challenger"]
    client = mlflow.MlflowClient()
    assert int(client.get_model_version_by_alias(NAME, "champion").version) == first.version
    version = client.get_model_version(NAME, str(retrained.version))
    assert "Against champion v1" in version.description

    unbeatable = training.Champion(_FixedModel(table, oracle=True), 0.5, "1")
    lost = training.run_training(table, NAME, train.RecentSplit(2, 2), champion=unbeatable)
    assert lost.vs_champion["beats_champion"] == 0.0
    assert not lost.passed_gates and lost.aliases == []
    assert int(client.get_model_version_by_alias(NAME, "challenger").version) == retrained.version


def test_a_version_knows_the_rows_it_was_built_on(store, table):
    outcome = training.run_training(table, NAME, train.RecentSplit(2, 2))
    wrong_split = train.Split(train_days=1, valid_days=1)  # what job settings would rebuild
    tr, va = batch_scoring.model_blocks(table, NAME, str(outcome.version), wrong_split)
    assert (len(tr), len(va)) == (outcome.rows["train"], outcome.rows["valid"])


def test_the_result_line_reports_the_gates(store, table):
    weak = training.Champion(_FixedModel(table, oracle=False), 0.5, "1")
    outcome = training.run_training(table, NAME, train.RecentSplit(2, 2), champion=weak)
    marker, payload = training.result_line(NAME, outcome).split(" ", 1)
    read = json.loads(payload)
    assert marker == training.RESULT_MARKER and read["version"] == outcome.version
    assert read["passed_gates"] is True and read["vs_champion"]["beats_champion"] == 1.0


def test_known_data_faults_are_left_out_of_training(table):
    fault = {
        "site_id": "SITE_A",
        "from": "2026-11-03T00:00:00Z",
        "to": "2026-11-03T23:00:00Z",
        "reason": "test fault",
    }
    kept = training.drop_excluded(table, [fault])
    day = pd.Timestamp("2026-11-03", tz="UTC")
    in_fault = table["prediction_ts"].between(day, day + pd.Timedelta(hours=23))
    assert len(kept) == len(table) - in_fault.sum() and in_fault.sum() > 0
    elsewhere = training.drop_excluded(table, [fault | {"site_id": "SITE_B"}])
    assert len(elsewhere) == len(table)  # the fault was at another site
    everywhere = training.drop_excluded(table, [fault | {"site_id": "all"}])
    assert len(everywhere) == len(kept)  # this test table has one site
    assert len(training.drop_excluded(table, [])) == len(table)


def test_the_exclusions_file_lists_every_catalog_with_valid_entries():
    for catalog in ("heavden_dev", "heavden_staging", "heavden_prod"):
        for entry in training.load_exclusions(catalog):
            assert set(entry) == {"site_id", "from", "to", "reason"}
            assert pd.Timestamp(entry["from"]).tz is not None  # UTC, written with a Z
            assert pd.Timestamp(entry["from"]) < pd.Timestamp(entry["to"])
