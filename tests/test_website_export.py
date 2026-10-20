"""The website's snapshot exported from Gold (heavden_api.export) and the checks that guard it."""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from heavden.agent import retrieval
from heavden_api import export, snapshot
from heavden_api.config import Settings
from heavden_api.deps import build_state
from heavden_api.documents import build_index
from heavden_api.routes.health import health
from heavden_api.snapshot import GOLD_TABLES, Snapshot, publish_snapshot, write_snapshot

ROOT = Path(__file__).resolve().parents[1]


def gold(demo_snapshot: Path) -> dict[str, pd.DataFrame]:
    """The demo snapshot's tables, standing in for an environment's Gold."""
    return {t: pd.read_parquet(demo_snapshot / "gold" / f"{t}.parquet") for t in GOLD_TABLES}


def as_of(tables) -> pd.Timestamp:
    return snapshot.snapshot_as_of(tables["patient_hour_features"])


def export_to(folder: Path, demo_snapshot: Path, days: float = 2) -> Snapshot:
    tables = gold(demo_snapshot)
    now = as_of(tables)
    loaded = Snapshot.load(demo_snapshot)
    m = loaded.manifest
    write_snapshot(
        folder,
        export.website_tables(tables, now, days),
        loaded.model,
        loaded.background,
        m.risk_bands,
        m.model_name,
        m.model_version,
        now,
    )
    return Snapshot.load(folder)


def test_the_website_gets_the_last_days_up_to_now_and_their_stays(demo_snapshot):
    tables = gold(demo_snapshot)
    now = as_of(tables)
    for name in ("patient_hour_features", "risk_scores"):  # E-000 left 2 days before "now"
        t = tables[name]
        late = t["prediction_ts"] > now - pd.Timedelta(days=2)
        tables[name] = t[~(t["encounter_id"].eq("E-000") & late)]
    out = export.website_tables(tables, now, days=2)
    for name, column in export.TIME_COLUMNS.items():
        times = out[name][column]
        assert times.gt(now - pd.Timedelta(days=2)).all() and times.le(now).all(), name
    assert out["patient_hour_features"]["prediction_ts"].nunique() == 48
    assert "E-000" not in set(out["encounters"]["encounter_id"])
    assert set(out["encounters"]["encounter_id"]) == set(
        out["patient_hour_features"]["encounter_id"]
    )


def test_hours_without_a_risk_score_are_refused(demo_snapshot):
    tables = gold(demo_snapshot)
    now = as_of(tables)
    tables["risk_scores"] = tables["risk_scores"][tables["risk_scores"]["prediction_ts"] < now]
    with pytest.raises(ValueError, match="have no risk score: run the score job first"):
        export.website_tables(tables, now)


def test_the_export_is_a_snapshot_the_website_loads(demo_snapshot, tmp_path):
    loaded = export_to(tmp_path / "web", demo_snapshot)
    m = loaded.manifest
    assert m.rows["patient_hour_features"] == 40 * 48
    assert pd.Timestamp(m.data_start) == m.as_of_ts - pd.Timedelta(hours=47)
    assert m.libraries["scikit-learn"] and m.schema_version == 2
    check = pd.read_parquet(tmp_path / "web" / "model" / "check.parquet")
    assert len(check) == snapshot.CHECK_ROWS and snapshot.CHECK_RISK in check
    assert loaded.rag_dir is None  # documents come with the image


def test_the_log_check_re_scores_the_models_own_hours(demo_snapshot):
    now = as_of(gold(demo_snapshot))
    tables = export.website_tables(gold(demo_snapshot), now)
    model = Snapshot.load(demo_snapshot).model
    assert "re-scored as logged" in export.log_check(model, "9", tables, now)  # the fixture's
    tables["risk_scores"]["risk"] += 1e-6
    with pytest.raises(ValueError, match="v9 scores .* logged hours differently"):
        export.log_check(model, "9", tables, now)


def test_a_model_that_did_not_score_now_is_refused(demo_snapshot):
    # prod on 20 Oct: v3 became champion after the last data, so "now" is v1's in the log
    now = as_of(gold(demo_snapshot))
    tables = export.website_tables(gold(demo_snapshot), now)
    model = Snapshot.load(demo_snapshot).model
    with pytest.raises(ValueError, match=r"scored by v9, not v10: score new hours with v10 first"):
        export.log_check(model, "10", tables, now)


def test_a_model_that_scores_its_check_hours_differently_is_refused(demo_snapshot, tmp_path):
    export_to(tmp_path / "web", demo_snapshot)
    path = tmp_path / "web" / "model" / "check.parquet"
    check = pd.read_parquet(path)
    check.assign(check_risk=check["check_risk"] + 1e-6).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="scores its 200 check hours differently here"):
        Snapshot.load(tmp_path / "web")


class PickledElsewhere:
    """A model whose pickle came from other library versions (prod's v1 in scikit-learn 1.9.1)."""

    def predict_proba(self, frame):
        raise AttributeError("'SimpleImputer' object has no attribute '_fill_dtype'")


def test_a_model_that_cannot_score_here_is_refused_with_both_library_versions(
    demo_snapshot, tmp_path
):
    export_to(tmp_path / "web", demo_snapshot)
    manifest_path = tmp_path / "web" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["libraries"]["scikit-learn"] = "1.7.2"
    manifest_path.write_text(json.dumps(manifest))
    joblib.dump(PickledElsewhere(), tmp_path / "web" / "model" / "model.joblib")
    with pytest.raises(
        RuntimeError, match=r"can't score here.*scikit-learn 1\.7\.2 there, \S+ here"
    ):
        Snapshot.load(tmp_path / "web")


def test_publishing_copies_plain_files(demo_snapshot, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):  # a volume can't set permissions or times
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(snapshot.shutil, "copystat", refuse)
    monkeypatch.setattr(snapshot.shutil, "copymode", refuse)
    pointer = publish_snapshot(demo_snapshot, tmp_path / "volume", "s1")
    assert pointer["files"] == snapshot.file_listing(demo_snapshot)


def test_the_images_document_index_serves_a_snapshot_without_one(demo_snapshot, tmp_path):
    web = tmp_path / "web"
    export_to(web, demo_snapshot)
    embedder = retrieval.HashingEmbedder()
    settings = dict(snapshot_dir=web, llm_provider="openai", openai_api_key=None)
    assert build_state(Settings(**settings), embedder=embedder).retriever() is None
    n = build_index(tmp_path / "rag", embedder)  # what the image build does, from git
    state = build_state(Settings(**settings, rag_dir=tmp_path / "rag"), embedder=embedder)
    assert len(state.retriever().chunks) == n > 50
    body = health(state)  # what /health says on Container Apps
    assert body["documents"] and body["snapshot"] == "web"


def test_the_job_publishes_to_the_volume_terraform_creates():
    bundle = yaml.safe_load((ROOT / "databricks" / "databricks.yml").read_text())
    prod = bundle["targets"]["prod"]["variables"]
    assert bundle["variables"]["website_volume"]["default"] == ""  # staging: build and check
    terraform = (ROOT / "infra" / "terraform" / "website.tf").read_text()
    volume = re.search(r'resource "databricks_volume" "website" \{(.*?)\n\}', terraform, re.S)[1]
    name = re.search(r'^\s*name\s*=\s*"(\w+)"', volume, re.M)[1]
    env, schema = re.search(
        r'schema_name\s*=\s*databricks_schema\.layer\["(\w+)\.(\w+)"\]', volume
    ).groups()
    assert prod["catalog"] == f"heavden_{env}"
    assert prod["website_volume"] == f"/Volumes/{prod['catalog']}/{schema}/{name}"
    job = yaml.safe_load(
        (ROOT / "databricks" / "resources" / "website_snapshot.job.yml").read_text()
    )
    task = job["resources"]["jobs"]["website_snapshot"]["tasks"][0]
    assert "--volume=${var.website_volume}" in task["spark_python_task"]["parameters"]
    env = job["resources"]["jobs"]["website_snapshot"]["environments"][0]["spec"]
    assert env["environment_version"] == "6" and "lightgbm==4.6.0" in env["dependencies"]


def test_a_release_moves_nothing_the_website_reads_until_its_image_passed():
    job = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())["jobs"][
        "website"
    ]
    order = [
        "Export the website snapshot (not served yet)",
        "Build the image",
        "Smoke-test the image on the snapshot",
        "Push the image to GHCR",
        "Point the website at the snapshot",
        "Deploy to Container Apps",
    ]
    steps = {s["name"]: s for s in job["steps"] if "name" in s}
    assert [name for name in steps if name in order] == order
    assert "--params move_pointer=false" in steps[order[0]]["run"]
    assert "smoke_api.py http://127.0.0.1:8000" in steps[order[2]]["run"]
    assert 'smoke_api.py "https://$fqdn"' in steps[order[5]]["run"]  # and once deployed
    bundle = yaml.safe_load((ROOT / "databricks" / "databricks.yml").read_text())
    assert job["env"]["VOLUME"] == bundle["targets"]["prod"]["variables"]["website_volume"]
    snapshot_job = yaml.safe_load(
        (ROOT / "databricks" / "resources" / "website_snapshot.job.yml").read_text()
    )["resources"]["jobs"]["website_snapshot"]
    assert snapshot_job["parameters"] == [{"name": "move_pointer", "default": "true"}]  # data runs
    terraform = (ROOT / "infra" / "terraform" / "website.tf").read_text()
    assert 'name   = "api"' in terraform and "--container-name api" in steps[order[5]]["run"]


def test_prod_and_staging_runs_pass_every_data_run_input():
    workflows = ROOT / ".github" / "workflows"
    inputs = yaml.safe_load((workflows / "data-run.yml").read_text())[True]["workflow_call"][
        "inputs"
    ]
    for caller in ("prod-run.yml", "staging-run.yml"):
        flow = yaml.safe_load((workflows / caller).read_text())
        passed = next(iter(flow["jobs"].values()))["with"]
        assert set(inputs) == set(passed), caller
        assert passed["website_snapshot"] == "${{ inputs.website_snapshot }}"
