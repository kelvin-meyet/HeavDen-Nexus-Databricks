"""What the assistant reads on Databricks, published by the job `assistant_publish` (Plan.md §11.2).

* **`ml.corpus_chunks`**: the documents of `docs/corpus`, chunked as the app's own search chunks
  them (`heavden.agent.corpus`), one row per chunk with its document's front matter. Change data
  feed is on, so a Vector Search index can sync from it.
* **`ml.explain_patient_risk(patient)`**: a SQL table function answering "why is this patient
  flagged?" from Gold, as the app's `explain_patient` tool does from its snapshot: the stay's
  latest score, band, NEWS2 and top factors, the risk 6 hours earlier, and its alerts of the
  last 48 hours. The assistant calls it; anyone with access can too (`SELECT * FROM ...`).

The documents ship inside the wheel (`heavden/agent/corpus_docs`, from `docs/corpus`), so what
is published is what the code was tested with.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from heavden.agent.corpus import Chunk, chunk_corpus, load_corpus

PACKAGED = Path(__file__).with_name("corpus_docs")
REPO = Path(__file__).resolve().parents[4] / "docs" / "corpus"  # a source checkout


def corpus_dir() -> Path:
    """The documents: inside the installed wheel, or `docs/corpus` in a source checkout."""
    return PACKAGED if PACKAGED.is_dir() else REPO


def chunk_rows(chunks: list[Chunk]) -> pd.DataFrame:
    """One row per chunk: its text, what to embed (text plus its document and section, so a
    short chunk still knows where it belongs), the citation, and the document's front matter
    for filtering (current versions, or those in force on a date)."""
    return pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "doc_id": [c.doc_id for c in chunks],
            "title": [c.title for c in chunks],
            "version": [c.version for c in chunks],
            "status": [c.status for c in chunks],
            "effective_date": [c.effective_date for c in chunks],
            "section": [c.section for c in chunks],
            "text": [c.text for c in chunks],
            "embedding_text": [c.embedding_text for c in chunks],
            "citation": [c.citation for c in chunks],
        }
    )


# The function's query, with `{gold}` (the catalog's gold schema) and `{patient}` (the asked
# label or stay id) to fill in. A label may be typed as "p1042", "P1042" or "P-1042".
EXPLAIN_BODY = """
WITH asked AS (
  SELECT CASE WHEN trim({patient}) RLIKE '(?i)^P-?[0-9]+$'
              THEN concat('P-', regexp_extract(trim({patient}), '[0-9]+', 0))
              ELSE trim({patient}) END AS id
), now AS (
  SELECT max(prediction_ts) AS as_of FROM {gold}.risk_scores
), latest AS (
  SELECT e.encounter_id, e.patient_label, e.age, e.sex, e.conditions, e.site_id, e.unit_id,
         e.bed_id, e.admit_ts, e.discharge_ts, r.prediction_ts, r.risk, r.risk_band,
         r.news2_total, r.top_factors, r.model_version, earlier.risk AS risk_6h_before
  FROM {gold}.encounters e
  JOIN asked ON e.patient_label = asked.id OR e.encounter_id = asked.id
  JOIN {gold}.risk_scores r ON r.encounter_id = e.encounter_id
  LEFT JOIN {gold}.risk_scores earlier
    ON earlier.encounter_id = r.encounter_id
   AND earlier.prediction_ts = r.prediction_ts - INTERVAL 6 HOURS
  ORDER BY r.prediction_ts DESC
  LIMIT 1
), alerts AS (
  SELECT a.encounter_id,
         to_json(array_sort(collect_list(named_struct(
           'alert_ts', a.alert_ts, 'risk', round(a.risk, 4),
           'escalated_within_6h', a.escalated_within_6h)))) AS recent_alerts_48h
  FROM {gold}.alerts_fact a
  JOIN latest l ON a.encounter_id = l.encounter_id
  CROSS JOIN now
  WHERE a.alert_ts > now.as_of - INTERVAL 48 HOURS AND a.alert_ts <= now.as_of
  GROUP BY a.encounter_id
)
SELECT l.*, l.prediction_ts = now.as_of AS on_ward_now,
       coalesce(alerts.recent_alerts_48h, '[]') AS recent_alerts_48h
FROM latest l
CROSS JOIN now
LEFT JOIN alerts ON alerts.encounter_id = l.encounter_id
"""

EXPLAIN_COLUMNS = (
    "encounter_id STRING, patient_label STRING, age INT, sex STRING, conditions STRING, "
    "site_id STRING, unit_id STRING, bed_id STRING, admit_ts TIMESTAMP, discharge_ts TIMESTAMP, "
    "prediction_ts TIMESTAMP, risk DOUBLE, risk_band STRING, news2_total DOUBLE, "
    "top_factors STRING, model_version STRING, risk_6h_before DOUBLE, on_ward_now BOOLEAN, "
    "recent_alerts_48h STRING"
)


def explain_function_sql(catalog: str) -> str:
    """CREATE OR REPLACE FUNCTION for `<catalog>.ml.explain_patient_risk`."""
    body = EXPLAIN_BODY.format(gold=f"{catalog}.gold", patient="patient")
    return (
        f"CREATE OR REPLACE FUNCTION {catalog}.ml.explain_patient_risk(\n"
        "  patient STRING COMMENT 'A patient label such as P-1042, or a stay id'\n"
        f") RETURNS TABLE ({EXPLAIN_COLUMNS})\n"
        "COMMENT 'Why is this patient flagged? Latest risk score, band, NEWS2 and top factors, the "
        "risk 6 hours earlier and the alerts of the last 48 hours (synthetic data).'\n"
        f"RETURN {body}"
    )


CHUNKS_SCHEMA = (
    "chunk_id STRING, doc_id STRING, title STRING, version STRING, status STRING, "
    "effective_date DATE, section STRING, text STRING, embedding_text STRING, citation STRING"
)
# A chunk changed if its text, its document's version, status or date, or its title changed
COMPARED = ("embedding_text", "version", "status", "effective_date", "title")


def publish_chunks(spark, table: str, rows: pd.DataFrame) -> str:
    """Bring `table` in line with `rows` in one MERGE (new chunks in, changed ones updated, gone
    ones out), so a synced index re-embeds only what changed. Returns what it did."""
    from heavden.ml.batch_scoring import to_spark

    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table} ({CHUNKS_SCHEMA}) USING DELTA "
        "TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    typed = [
        f"CAST({name} AS {kind}) AS {name}"
        for name, kind in (column.split() for column in CHUNKS_SCHEMA.split(", "))
    ]
    to_spark(spark, rows).selectExpr(*typed).createOrReplaceTempView("heavden_corpus_chunks")
    before = spark.read.table(table).toPandas().set_index("chunk_id")
    changed = " OR ".join(f"t.{c} <> s.{c}" for c in COMPARED)
    spark.sql(
        f"MERGE INTO {table} t USING heavden_corpus_chunks s ON t.chunk_id = s.chunk_id "
        f"WHEN MATCHED AND ({changed}) THEN UPDATE SET * "
        "WHEN NOT MATCHED THEN INSERT * "
        "WHEN NOT MATCHED BY SOURCE THEN DELETE"
    )
    now = rows.set_index("chunk_id")
    kept = now.index.intersection(before.index)
    compared = list(COMPARED)
    differ = now.loc[kept, compared].astype(str) != before.loc[kept, compared].astype(str)
    added = len(now.index.difference(before.index))
    removed = len(before.index.difference(now.index))
    documents = len(rows[["doc_id", "version"]].drop_duplicates())  # each version counts
    return (
        f"{table}: {len(rows)} chunks from {documents} documents; "
        f"{added} added, {int(differ.any(axis=1).sum())} changed, {removed} removed"
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_staging")
    args = parser.parse_args(argv)
    spark = SparkSession.builder.getOrCreate()
    c = args.catalog

    rows = chunk_rows(chunk_corpus(load_corpus(corpus_dir())))
    print(publish_chunks(spark, f"{c}.ml.corpus_chunks", rows))
    spark.sql(explain_function_sql(c))
    print(f"{c}.ml.explain_patient_risk: created or replaced")
