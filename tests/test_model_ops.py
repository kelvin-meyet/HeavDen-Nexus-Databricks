import json

import mlflow
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from heavden.ml import model_ops, tracking

NAME = "deterioration_risk_ops_test"
CHECK = "cc6d54c8-3c99-4135-bee0-722140851436"


def _version(
    passed_gates: bool, compared_with: str | None = None, gates: dict | None = None
) -> str:
    """Register a tiny model as a new version, tagged the way the training job tags them."""
    with mlflow.start_run() as run:
        if compared_with is not None:
            mlflow.set_tag("compared_with_version", compared_with)
        mlflow.log_metrics({f"vs_champion_{k}": v for k, v in (gates or {}).items()})
        model = DummyClassifier(strategy="prior").fit(np.zeros((4, 1)), [0, 0, 0, 1])
        logged = mlflow.sklearn.log_model(model, name="model")
    client = mlflow.MlflowClient()
    version = str(mlflow.register_model(logged.model_uri, NAME).version)
    assert client.get_model_version(NAME, version).run_id == run.info.run_id
    client.set_model_version_tag(NAME, version, "passed_gates", str(passed_gates).lower())
    return version


@pytest.fixture
def registry(tmp_path):
    """v1 is the champion; v2 is a challenger that beat v1; v3 failed its gates."""
    tracking.use_local_store(tmp_path / "mlflow", experiment="ops-test")
    client = mlflow.MlflowClient()
    v1 = _version(True)
    v2 = _version(True, compared_with=v1, gates={"auroc": 0.81, "champion_auroc": 0.75})
    v3 = _version(False, compared_with=v1)
    client.set_registered_model_alias(NAME, "champion", v1)
    client.set_registered_model_alias(NAME, "challenger", v2)
    yield client, (v1, v2, v3)
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


def test_promotion_moves_champion_and_remembers_the_previous_one(registry):
    client, (v1, v2, _) = registry
    change = model_ops.promote(client, NAME)
    assert (change["version"], change["from_version"]) == (v2, v1)
    assert change["gates"] == {"auroc": 0.81, "champion_auroc": 0.75}
    assert model_ops.aliases(client, NAME) == {"champion": v2, "previous_champion": v1}
    with pytest.raises(ValueError, match="already @champion"):
        model_ops.promote(client, NAME, v2)


def test_only_a_version_that_beat_the_current_champion_is_promoted(registry):
    client, (v1, v2, v3) = registry
    with pytest.raises(ValueError, match="did not pass its gates"):
        model_ops.promote(client, NAME, v3)
    not_compared = _version(True)  # passed the NEWS2 gates, but was never a retrain
    with pytest.raises(ValueError, match="not a retrain"):
        model_ops.promote(client, NAME, not_compared)
    model_ops.promote(client, NAME, v2)
    stale = _version(True, compared_with=v1)  # beat v1, but the champion is v2 now
    with pytest.raises(
        ValueError, match=f"compared with version {v1}, but @champion is version {v2}"
    ):
        model_ops.promote(client, NAME, stale)
    assert model_ops.aliases(client, NAME)["champion"] == v2  # refusals change nothing
    client.delete_registered_model_alias(NAME, "champion")
    client.delete_registered_model_alias(NAME, "previous_champion")
    with pytest.raises(ValueError, match="nothing to promote"):
        model_ops.promote(client, NAME)


def test_rollback_returns_to_the_previous_champion_and_can_be_undone(registry):
    client, (v1, v2, v3) = registry
    with pytest.raises(ValueError, match="nothing to roll back to"):
        model_ops.rollback(client, NAME)
    model_ops.promote(client, NAME)
    back = model_ops.rollback(client, NAME)
    assert (back["version"], back["from_version"]) == (v1, v2)
    assert model_ops.aliases(client, NAME) == {"champion": v1, "previous_champion": v2}
    forward = model_ops.rollback(client, NAME)  # undo: the version rolled back from
    assert forward["version"] == v2 and model_ops.aliases(client, NAME)["champion"] == v2
    with pytest.raises(ValueError, match="did not pass its gates"):
        model_ops.rollback(client, NAME, v3)
    with pytest.raises(ValueError, match="already @champion"):
        model_ops.rollback(client, NAME, v2)


def test_status_and_selftest_leave_the_aliases_alone(registry):
    client, (v1, v2, v3) = registry
    before = model_ops.status(client, NAME)
    assert before["aliases"] == {"champion": v1, "challenger": v2}
    assert before["versions"][-1] == {"version": v3, "passed_gates": "false"}
    result = model_ops.selftest(client, NAME)
    assert result["alias_round_trip"] == "ok" and result["on_version"] == v3
    assert model_ops.status(client, NAME) == before
    assert not tracking.has_alias(NAME, model_ops.SELFTEST_ALIAS)
    assert model_ops.status(client, NAME, newest=1)["versions"] == [before["versions"][-1]]


def test_only_not_found_counts_as_an_alias_that_is_not_set(registry):
    """Read as "no champion", a permission error would let a promotion skip its check."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import PERMISSION_DENIED

    client, _ = registry

    class Denied:
        def get_model_version_by_alias(self, name, alias):
            raise MlflowException("no access", error_code=PERMISSION_DENIED)

    with pytest.raises(MlflowException, match="no access"):
        model_ops.aliases(Denied(), NAME)
    client.delete_registered_model_alias(NAME, "challenger")
    assert set(model_ops.aliases(client, NAME)) == {"champion"}  # the others: simply not set


def test_context_survives_the_trip_through_a_job_parameter():
    context = {"actor": "kelvin-meyet", "reason": 'sensor fault, "FSN-2026-11" = no retrain; é'}
    word = model_ops.encode_context(context)
    assert "," not in word and '"' not in word and " " not in word
    assert model_ops.decode_context(word) == context
    assert model_ops.decode_context("") == {} and model_ops.decode_context(None) == {}


def test_audit_rows_have_every_column_and_valid_json():
    context = {"actor": "kelvin-meyet", "reason": "", "run_url": "https://x/1", "check_id": CHECK}
    row = model_ops.audit_row(
        "retrain", "failed_gates", "c.ml.m", 7, "1", context, {"auroc": float("nan"), "n": 21}
    )
    assert list(row) == model_ops.AUDIT_COLUMNS
    assert (row["version"], row["from_version"], row["reason"], row["episode"]) == (
        "7",
        "1",
        None,
        None,
    )
    assert json.loads(row["details"]) == {"auroc": None, "n": 21}  # NaN is not JSON
    assert row["event_ts"].tzinfo is not None and row["event_ts"] == row["recorded_at"]


def _review(run_id: int, workflow: str, title: str, comment: str = "") -> dict:
    return {
        "workflow": workflow,
        "run_id": run_id,
        "title": title,
        "run_url": f"https://github.com/o/r/actions/runs/{run_id}",
        "decided_at": "2026-10-13T01:55:50Z",
        "environment": "retrain-approval" if workflow == "retrain" else "production-approval",
        "actor": "kelvin-meyet",
        "comment": comment,
    }


def test_rejected_requests_become_rows_once():
    reviews = [
        _review(
            11, "retrain", f"Retrain request: episode sensor-bias, check {CHECK}", "sensor fault"
        ),
        _review(12, "promote", "Promotion request: version 7"),
        _review(13, "retrain", "retrain"),  # a test run from before requests had titles
    ]
    rows = model_ops.rejection_rows("c.ml.m", reviews, known=set())
    assert [(r["event"], r["outcome"]) for r in rows] == [
        ("retrain", "rejected"),
        ("promote", "rejected"),
    ]
    retrain, promotion = rows
    assert (retrain["check_id"], retrain["episode"], retrain["reason"]) == (
        CHECK,
        "sensor-bias",
        "sensor fault",
    )
    assert retrain["event_id"] == "github-11-retrain-approval" and retrain["version"] is None
    assert retrain["event_ts"].isoformat() == "2026-10-13T01:55:50+00:00"
    assert (promotion["version"], promotion["reason"], promotion["check_id"]) == ("7", None, None)
    again = model_ops.rejection_rows("c.ml.m", reviews, known={r["event_id"] for r in rows})
    assert again == []


def test_audit_rows_fit_the_table_schema(spark):
    rows = [
        model_ops.audit_row("promote", "promoted", "c.ml.m", "2", "1", {"actor": "kelvin-meyet"}),
        model_ops.audit_row("rollback", "rolled_back", "c.ml.m", "1", "2", {}),
    ]
    frame = model_ops.audit_frame(spark, rows)
    assert frame.columns == model_ops.AUDIT_COLUMNS
    assert frame.where("actor IS NULL AND details IS NULL AND reason IS NULL").count() == 1
    back = frame.toPandas()
    assert back["outcome"].tolist() == ["promoted", "rolled_back"]
    assert back["actor"].iloc[0] == "kelvin-meyet" and back["details"].isna().all()
    assert back["event_ts"].iloc[0] == rows[0]["event_ts"].replace(tzinfo=None)  # UTC session
