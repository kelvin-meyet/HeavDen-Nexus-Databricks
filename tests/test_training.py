from pathlib import Path

import mlflow
import numpy as np
import pytest
from test_train import _table

from heavden.ml import tracking, train, training

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
