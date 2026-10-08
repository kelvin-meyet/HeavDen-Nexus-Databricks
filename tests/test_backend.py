from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_scoring_gold import T0, full_table

from heavden.agent import corpus, retrieval
from heavden.ml import scoring, train
from heavden_api.app import create_app
from heavden_api.config import Settings
from heavden_api.snapshot import Snapshot, build_snapshot

pytest.importorskip("faiss")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

ROOT = Path(__file__).resolve().parents[1]


def _activity(table: pd.DataFrame) -> SimpleNamespace:
    enc = table.groupby("encounter_id").agg(
        patient_id=("patient_id", "first"),
        site_id=("site_id", "first"),
        unit_id=("unit_id", "first"),
        admit_ts=("prediction_ts", "min"),
    )
    enc = enc.reset_index()
    enc["admit_ts"] -= pd.Timedelta(hours=1)
    enc["bed_id"] = enc["unit_id"] + "-" + enc.index.astype(str).str.zfill(3)
    enc["discharge_ts"] = pd.NaT
    enc["device_id"] = "DEV-" + enc["site_id"].str[-1] + "-" + enc.index.astype(str).str.zfill(4)
    patients = pd.DataFrame({"patient_id": enc["patient_id"], "age": 70, "sex": "F"})
    for flag in ("copd", "heart_failure", "diabetes", "ckd", "hypertension", "atrial_fibrillation"):
        patients[flag] = False
    patients.loc[0, "copd"] = True
    end = table["prediction_ts"].max() + pd.Timedelta(hours=1)
    assignments = enc[["device_id", "encounter_id", "patient_id"]].assign(
        start_ts=enc["admit_ts"], end_ts=pd.NaT
    )
    return SimpleNamespace(
        encounters=enc, patients=patients, device_assignments=assignments, start=T0, end=end
    )


def _readings(activity) -> pd.DataFrame:
    ts = pd.date_range(
        activity.end - pd.Timedelta(hours=2), activity.end, freq="5min", inclusive="left"
    )
    rows = [
        {
            "device_id": d,
            "ts": t,
            "heart_rate": 80.0 + i % 3,
            "sbp": 120.0,
            "dbp": 70.0 + i % 2,
            "spo2": 96.0,
            "battery_pct": 80,
            "firmware": "3.1.4",
        }
        for d in activity.device_assignments["device_id"]
        for i, t in enumerate(ts)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("snap")
    table = full_table()
    cut1, cut2 = T0 + pd.Timedelta(days=3), T0 + pd.Timedelta(days=5)
    tr = table[table["prediction_ts"] < cut1]
    va = table[(table["prediction_ts"] >= cut1) & (table["prediction_ts"] < cut2)]
    results, _ = train.train_candidates(tr, va)
    model = next(r for r in results if r.name == "logistic_regression").model
    bands = scoring.RiskBands.fit(va, model.predict_proba(va)[:, 1])
    activity = _activity(table)
    outcomes = pd.DataFrame(
        {
            "encounter_id": ["E-001"],
            "patient_id": ["PE-001"],
            "event_type": ["rapid_response"],
            "event_ts": [T0 + pd.Timedelta(days=4, minutes=30)],
        }
    )

    embedder = retrieval.HashingEmbedder()
    rag = tmp / "rag_src"
    chunks = corpus.chunk_corpus(corpus.load_corpus(ROOT / "docs" / "corpus"))
    retrieval.Retriever(chunks, embedder).save(rag)

    build_snapshot(
        tmp / "snapshot",
        table,
        activity,
        outcomes,
        _readings(activity),
        model,
        tr,
        bands,
        model_name="deterioration_risk",
        model_version="9",
        rag_dir=rag,
    )
    app = create_app(Settings(snapshot_dir=tmp / "snapshot"), embedder=embedder)
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health_describes_the_snapshot(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "demo" and body["synthetic"]
    assert body["model"] == {"name": "deterioration_risk", "version": "9", "type": "Pipeline"}
    assert body["documents"] and body["rows"]["risk_scores"] > 0


def test_summary_counts_the_ward_now(client):
    all_sites = client.get("/analytics/summary").json()
    site_a = client.get("/analytics/summary", params={"site_id": "SITE_A"}).json()
    assert all_sites["census"] == 40 and 0 < site_a["census"] < 40
    assert all_sites["high_risk"] + all_sites["medium_risk"] <= all_sites["census"]
    assert client.get("/analytics/summary", params={"site_id": "SITE_Z"}).status_code == 422


@pytest.mark.parametrize(
    "path,params",
    [
        ("/analytics/census", {"hours": 6}),
        ("/analytics/units", {}),
        ("/analytics/alerts", {"days": 3}),
        ("/analytics/devices", {"days": 1}),
    ],
)
def test_analytics_series_return_rows_with_iso_times(client, path, params):
    rows = client.get(path, params=params).json()
    assert rows and all("site_id" in r for r in rows)
    times = [v for r in rows for k, v in r.items() if k in ("hour_ts", "day")]
    assert all(t.endswith("+00:00") for t in times)  # UTC, never local time


def test_board_is_sorted_by_risk_and_explained(client):
    body = client.get("/risk/board", params={"limit": 10}).json()
    risks = [p["risk"] for p in body["patients"]]
    assert risks == sorted(risks, reverse=True) and len(risks) == 10
    first = body["patients"][0]
    assert first["patient_label"].startswith("P-") and "patient_id" not in first
    assert len(first["top_factors"]) == 3 and first["risk_6h_ago"] is not None
    high = client.get("/risk/board", params={"band": "High"}).json()["patients"]
    assert all(p["risk_band"] == "High" for p in high)


def test_patient_detail_and_unknown_patient(client):
    body = client.get("/risk/patients/E-001", params={"hours": 6}).json()
    assert body["patient"]["patient_label"].startswith("P-") and "patient_id" not in body["patient"]
    assert len(body["series"]) == 6 and body["series"][-1]["prediction_ts"] == body["as_of"]
    assert client.get("/risk/patients/E-999").status_code == 404


def test_what_if_scores_the_change(client):
    body = client.post(
        "/risk/score",
        json={"encounter_id": "E-002", "changes": {"spo2": 84, "resp_rate": 30, "acvpu": "C"}},
    ).json()
    assert body["risk_after"] > body["risk_before"] and body["news2_after"] > body["news2_before"]
    assert 0 < body["risk_after"] < 1
    assert body["changes"] == {"spo2": 84.0, "resp_rate": 30.0, "acvpu": "C"}


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"encounter_id": "E-002", "changes": {"spo2": 140}}, 422),  # impossible value
        ({"encounter_id": "E-002", "changes": {"acvpu": "X"}}, 422),
        ({"encounter_id": "E-999", "changes": {"spo2": 90}}, 404),  # not on the ward
    ],
)
def test_what_if_rejects_bad_requests(client, payload, status):
    assert client.post("/risk/score", json=payload).status_code == status


def test_document_search_cites_and_filters_versions(client):
    body = client.get("/documents/search", params={"q": "RRT extension Valley", "k": 3}).json()
    assert body["results"][0]["citation"].startswith(body["results"][0]["title"])
    assert any(r["doc_id"] == "HD-OPS-001" for r in body["results"])
    old = client.get(
        "/documents/search",
        params={"q": "NEWS2 escalation protocol", "k": 5, "as_of": "2026-11-05"},
    ).json()["results"]
    assert {r["version"] for r in old if r["doc_id"] == "HD-CLIN-001"} == {"1.0"}
    assert client.get("/documents/search", params={"q": "x"}).status_code == 422


def test_snapshot_round_trips(client, tmp_path_factory):
    snapshot = client.app.state.heavden.snapshot
    reloaded = Snapshot.load(snapshot.root)
    assert reloaded.manifest == snapshot.manifest
    assert reloaded.manifest.as_of_ts == pd.Timestamp(snapshot.manifest.as_of)
    assert np.isclose(reloaded.manifest.risk_bands.high, snapshot.manifest.bands["high"])


def test_live_mode_is_not_available_yet(tmp_path):
    with pytest.raises(ValueError, match="MODE"):
        Settings(mode="prod")
    app = create_app(Settings(mode="live", snapshot_dir=tmp_path))
    with pytest.raises(RuntimeError, match="Phase 5"):
        with fastapi_testclient.TestClient(app):
            pass


def test_missing_snapshot_explains_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_demo_snapshot"):
        with fastapi_testclient.TestClient(create_app(Settings(snapshot_dir=tmp_path))):
            pass


def test_summary_labels_alert_and_escalation_metrics(client):
    body = client.get("/analytics/summary").json()
    for key in ("alerts_per_nurse_shift_24h", "alert_precision_7d", "escalations_flagged_7d"):
        assert key in body and key in body["definitions"]
    assert "enters the High band" in body["definitions"]["alerts_24h"]
    assert body["alert_budget_per_nurse_shift"] == 2.0
