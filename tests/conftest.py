"""Shared fixtures: a small demo snapshot built with the real snapshot builder."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_scoring_gold import T0, full_table

from heavden.agent import corpus, retrieval
from heavden.ml import scoring, train
from heavden_api.snapshot import build_snapshot

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


@pytest.fixture(scope="session")
def demo_snapshot(tmp_path_factory) -> Path:
    """A small demo snapshot (synthetic table, hashing-embedder document index)."""
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
    return tmp / "snapshot"
