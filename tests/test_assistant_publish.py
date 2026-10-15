"""What the assistant reads on Databricks: the published documents and the explanation function
(heavden.agent.publish)."""

import json

import pandas as pd
import pytest

from heavden.agent import publish
from heavden.agent.corpus import chunk_corpus, load_corpus
from heavden.ml.batch_scoring import to_spark

T0 = pd.Timestamp("2026-10-01", tz="UTC")


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return publish.chunk_rows(chunk_corpus(load_corpus(publish.corpus_dir())))


def test_every_document_is_published_in_chunks_with_its_front_matter(rows):
    assert publish.corpus_dir().name in {"corpus", "corpus_docs"}
    assert rows["chunk_id"].is_unique
    assert len(rows[["doc_id", "version"]].drop_duplicates()) == 9  # protocol v1.0 and v2.0
    protocols = rows[rows["title"].str.contains("Escalation Protocol")]
    assert {"current", "superseded"} <= set(protocols["status"])  # versions can be told apart
    first = rows.iloc[0]
    assert first["title"] in first["embedding_text"] and first["text"] in first["embedding_text"]
    assert rows["effective_date"].map(type).eq(type(T0.date())).all()


def _gold(spark):
    """Two stays of one patient (P-1001) and one of another, scored hourly, with alerts."""
    hours = pd.date_range(T0, periods=12, freq="h")
    scores = [
        {
            "encounter_id": enc,
            "prediction_ts": ts,
            "risk": 0.01 * (i + 1) * k,
            "risk_band": "Low",
            "news2_total": 2.0,
            "top_factors": json.dumps([{"factor": "spo2"}]),
            "model_version": "1",
        }
        for enc, k, stay_hours in (("E-1", 1, hours[:4]), ("E-2", 2, hours), ("E-3", 3, hours))
        for i, ts in enumerate(stay_hours)
    ]
    encounters = pd.DataFrame(
        {
            "encounter_id": ["E-1", "E-2", "E-3"],
            "patient_id": ["a", "a", "b"],
            "patient_label": ["P-1001", "P-1001", "P-1002"],
            "age": [70, 70, 55],
            "sex": ["F", "F", "M"],
            "conditions": ["copd", "copd", "none"],
            "site_id": ["SITE_A"] * 3,
            "unit_id": ["U1"] * 3,
            "bed_id": ["B1", "B2", "B3"],
            "admit_ts": [T0, T0, T0],
            "discharge_ts": [T0 + pd.Timedelta(hours=4), pd.NaT, pd.NaT],
        }
    )
    alerts = pd.DataFrame(
        {
            "encounter_id": ["E-2", "E-2", "E-3"],
            "alert_ts": [hours[3], hours[9], hours[5]],
            "risk": [0.08, 0.2, 0.15],
            "escalated_within_6h": [0.0, None, 1.0],
        }
    )
    old = pd.DataFrame(
        {
            "encounter_id": ["E-2"],
            "alert_ts": [T0 - pd.Timedelta(days=3)],
            "risk": [0.5],
            "escalated_within_6h": [1.0],
        }
    )
    for name, df in (
        ("risk_scores", pd.DataFrame(scores)),
        ("encounters", encounters),
        ("alerts_fact", pd.concat([alerts, old], ignore_index=True)),
    ):
        to_spark(spark, df).createOrReplaceTempView(f"gold_{name}")


def _explain(spark, patient: str) -> pd.DataFrame:
    body = publish.EXPLAIN_BODY.format(gold="gold", patient=f"'{patient}'")
    return spark.sql(body.replace("gold.", "gold_")).toPandas()


def test_the_function_explains_a_patients_latest_hour_like_the_apps_tool(spark):
    _gold(spark)
    for asked in ("P-1001", "p1001", "P1001", " p-1001 "):
        row = _explain(spark, asked).iloc[0]
        assert row["encounter_id"] == "E-2"  # the patient's latest stay and hour
        assert row["prediction_ts"] == pd.Timestamp(T0 + pd.Timedelta(hours=11)).tz_localize(None)
        assert row["risk"] == pytest.approx(0.24) and row["risk_6h_before"] == pytest.approx(0.12)
        assert bool(row["on_ward_now"])
        alerts = json.loads(row["recent_alerts_48h"])  # the last 48 h only, oldest first
        assert [a["risk"] for a in alerts] == [0.08, 0.2]
    by_stay = _explain(spark, "E-1").iloc[0]  # a stay id works too, and a discharged stay
    assert by_stay["patient_label"] == "P-1001" and not bool(by_stay["on_ward_now"])
    assert json.loads(by_stay["recent_alerts_48h"]) == []
    assert _explain(spark, "P-9999").empty


def test_the_function_returns_the_columns_it_declares(spark):
    _gold(spark)
    declared = [c.split()[0] for c in publish.EXPLAIN_COLUMNS.split(", ")]
    assert list(_explain(spark, "P-1002").columns) == declared
    sql = publish.explain_function_sql("heavden_staging")
    assert "FUNCTION heavden_staging.ml.explain_patient_risk(" in sql
    assert "heavden_staging.gold.encounters" in sql and "{" not in sql


def test_republishing_updates_only_what_changed(spark, rows):
    from conftest import DELTA

    if not DELTA:
        pytest.skip("Delta Lake runs on Linux (CI), not Windows")
    spark.sql("CREATE DATABASE IF NOT EXISTS ml")
    table = "ml.corpus_chunks"
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    assert "added" in publish.publish_chunks(spark, table, rows)
    changed = rows.iloc[1:].copy()  # one chunk gone, one document superseded
    changed.loc[changed.index[0], "status"] = "superseded"
    message = publish.publish_chunks(spark, table, changed)
    assert message.endswith("0 added, 1 changed, 1 removed"), message
    assert spark.read.table(table).count() == len(changed)
    props = spark.sql(f"SHOW TBLPROPERTIES {table}").toPandas().set_index("key")["value"]
    assert props.get("delta.enableChangeDataFeed") == "true"
