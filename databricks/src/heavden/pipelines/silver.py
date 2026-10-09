"""Silver: checked, de-duplicated and linked data (Plan.md §7.3-7.4).

Plain PySpark, so the pipeline (`databricks/src/pipelines/medallion/`) and the local tests run
the same functions. The pipeline files only add the Lakeflow decorators.

Vitals:
* `typed_readings`: Bronze rows with every field present and typed (absent fields are NULL,
  so Silver's schema doesn't change when a firmware update adds or drops a field).
* `READING_RULES`: the expectations. Rows failing any rule go to `silver.vitals_quarantine`
  with the names of the rules they failed (`failed_rules`).
* `deduplicate`: one reading per (device_id, ts), with a 24 h watermark that bounds the state.
  Spark silently drops a reading more than 24 h older than the newest one it has seen (e.g. an
  older day uploaded after a newer one); `reconcile_readings` counts those so they can't vanish.
* `link_readings`: attach encounter and patient. Devices are reused, so a reading belongs to the
  assignment with start_ts <= ts < end_ts on its device (same rule as `heavden.ml.features`).
* `reconcile_readings`: one row of counts from Bronze to linked Silver, so every reading is
  accounted for (quarantined, a duplicate, missing from Silver, or not on any patient).

Azure SQL tables: Bronze holds every snapshot and change the ingestion job landed; AUTO CDC
keeps the latest version per key (SCD type 1) or the full history (type 2), ordered by
`CDC_SEQUENCE`. Snapshot rows are applied as upserts: AUTO CDC can't see a row that a snapshot
no longer contains. The source never deletes, so that only matters after reloading *different*
data, which needs a full refresh of the pipeline anyway.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from heavden.ingestion.sql_source import TABLES

VITALS = ("heart_rate", "resp_rate", "spo2", "temp_c", "sbp", "dbp")

# Every field a device may send (generator/landing.py VITALS_FIELDS), with its Silver type.
READING_FIELDS: dict[str, str] = {
    "device_id": "string",
    "ts": "timestamp",
    **dict.fromkeys(VITALS, "double"),
    "motion": "double",
    "battery_pct": "double",
    "firmware": "string",
    "etco2": "double",  # sent only after the firmware change (drift scenario)
}
# Added by Bronze: which file a row came from and when it was ingested.
LINEAGE: dict[str, str] = {"_source_file": "string", "_ingest_ts": "timestamp"}

# Each rule is TRUE or FALSE, never NULL (a missing vital is allowed; the features handle it).
READING_RULES: dict[str, str] = {
    "device_id_present": "device_id IS NOT NULL",
    "ts_present": "ts IS NOT NULL",
    "parsed_cleanly": "_rescued_data IS NULL",
    "heart_rate_in_range": "heart_rate IS NULL OR heart_rate BETWEEN 20 AND 250",
    "resp_rate_in_range": "resp_rate IS NULL OR resp_rate BETWEEN 2 AND 70",
    "spo2_in_range": "spo2 IS NULL OR spo2 BETWEEN 50 AND 100",
    "temp_c_in_range": "temp_c IS NULL OR temp_c BETWEEN 30 AND 45",
    "sbp_in_range": "sbp IS NULL OR sbp BETWEEN 40 AND 300",
    "dbp_in_range": "dbp IS NULL OR dbp BETWEEN 20 AND 200",
}

# Late readings older than this (behind the newest reading seen) are no longer de-duplicated
# against; Spark drops them. Gateways resend within minutes, a day is generous.
DEDUP_WATERMARK = "24 hours"

OUTCOME_FIELDS: dict[str, str] = {
    "patient_id": "string",
    "encounter_id": "string",
    "event_type": "string",
    "event_ts": "timestamp",
    "recorded_ts": "timestamp",
}
OUTCOME_RULES: dict[str, str] = {
    "keys_present": "encounter_id IS NOT NULL AND patient_id IS NOT NULL",
    "event_ts_present": "event_ts IS NOT NULL",
    "known_event_type": "coalesce(event_type IN ('rapid_response', 'icu_transfer'), false)",
    "recorded_after_event": "coalesce(recorded_ts >= event_ts, false)",
    "parsed_cleanly": "_rescued_data IS NULL",
}

# Azure SQL tables: type 2 keeps history (what was true when), type 1 only the latest row.
# Encounters change unit on transfer and gain discharge_ts; device assignments gain end_ts.
SCD_TYPE: dict[str, int] = {
    "sites": 2,
    "units": 1,
    "patients": 2,
    "conditions": 1,
    "medications": 1,
    "encounters": 2,
    "nurse_observations": 1,
    "device_assignments": 2,
}
SQL_KEYS = TABLES
# Ingestion bookkeeping: carried to Silver, but a change in them alone isn't a new version.
CDC_BOOKKEEPING = ["_change_version", "_change_op", "_ingest_ts"]
# Order of changes per key: source time first; ties (a re-read of the same row) go to the
# later copy, which is identical.
CDC_SEQUENCE = ("last_updated", "_change_version", "_ingest_ts")
SQL_CHANGE_RULES = {"not_a_delete": "_change_op <> 'D'"}  # the source contract: no deletes


def _typed(df: DataFrame, fields: dict[str, str]) -> DataFrame:
    """`fields` cast to their types (NULL when absent), then `_rescued_data` and lineage."""
    present = set(df.columns)
    columns = fields | {"_rescued_data": "string"} | LINEAGE
    return df.select(
        *[
            (F.col(name) if name in present else F.lit(None)).cast(kind).alias(name)
            for name, kind in columns.items()
        ]
    )


def typed_readings(raw: DataFrame) -> DataFrame:
    return _typed(raw, READING_FIELDS)


def typed_outcomes(raw: DataFrame) -> DataFrame:
    return _typed(raw, OUTCOME_FIELDS)


def failed_rules(rules: dict[str, str]) -> Column:
    """Names of the rules a row fails, as an array (empty when it passes all)."""
    names = [F.when(~F.expr(condition), F.lit(name)) for name, condition in rules.items()]
    return F.filter(F.array(*names), lambda name: name.isNotNull())


def quarantine(typed: DataFrame, rules: dict[str, str]) -> DataFrame:
    """Rows failing at least one rule, with the rules they failed."""
    return typed.withColumn("failed_rules", failed_rules(rules)).where(F.size("failed_rules") > 0)


def deduplicate(readings: DataFrame) -> DataFrame:
    """One reading per device and time.

    On a stream the watermark bounds the state Spark keeps; a batch (tests, notebooks) has no
    watermark and simply drops duplicates.
    """
    keys = ["device_id", "ts"]
    if not readings.isStreaming:
        return readings.dropDuplicates(keys)
    return readings.withWatermark("ts", DEDUP_WATERMARK).dropDuplicatesWithinWatermark(keys)


def device_readings(typed: DataFrame) -> DataFrame:
    """Silver device readings: valid rows only (the pipeline's expectations drop the rest)."""
    return deduplicate(typed).drop("_rescued_data")


def link_readings(readings: DataFrame, assignments: DataFrame) -> DataFrame:
    """Attach encounter_id and patient_id by device and time window; unmatched readings drop."""
    a = assignments.select(
        "device_id",
        "encounter_id",
        "patient_id",
        F.col("start_ts").alias("_start"),
        F.col("end_ts").alias("_end"),
    ).alias("a")
    r = readings.alias("r")
    inside = (
        (F.col("r.device_id") == F.col("a.device_id"))
        & (F.col("r.ts") >= F.col("a._start"))
        & (F.col("a._end").isNull() | (F.col("r.ts") < F.col("a._end")))
    )
    return r.join(a, inside, "inner").select("a.encounter_id", "a.patient_id", "r.*")


def reconcile_readings(
    typed: DataFrame, readings: DataFrame, linked: DataFrame, rules: dict[str, str] = READING_RULES
) -> DataFrame:
    """Where every Bronze reading went: one row of counts.

    `missing_from_silver` should be 0: a valid reading that isn't in Silver was dropped by the
    de-duplication watermark (it arrived too late). `not_on_a_patient` are readings outside any
    device assignment, e.g. taken at the very minute a stay ended.
    """
    keys = ["device_id", "ts"]
    passes = " AND ".join(f"({condition})" for condition in rules.values())
    valid = typed.where(passes)
    expected = valid.select(*keys).distinct()
    in_silver = readings.select(*keys).distinct()
    missing = expected.join(in_silver, keys, "left_anti")
    counts = [
        typed.agg(F.count("*").alias("bronze_rows")),
        typed.where(f"NOT ({passes})").agg(F.count("*").alias("quarantined")),
        valid.agg(F.count("*").alias("valid_rows")),
        expected.agg(F.count("*").alias("distinct_valid")),
        in_silver.agg(F.count("*").alias("in_silver")),
        missing.agg(
            F.count("*").alias("missing_from_silver"),
            F.min("ts").alias("missing_first_ts"),
            F.max("ts").alias("missing_last_ts"),
        ),
        linked.agg(F.count("*").alias("linked")),
    ]
    out = counts[0]
    for frame in counts[1:]:
        out = out.crossJoin(frame)
    return out.select(
        "bronze_rows",
        "quarantined",
        (F.col("valid_rows") - F.col("distinct_valid")).alias("duplicates"),
        "in_silver",
        "missing_from_silver",
        "missing_first_ts",
        "missing_last_ts",
        "linked",
        (F.col("in_silver") - F.col("linked")).alias("not_on_a_patient"),
    )


def current(scd2: DataFrame) -> DataFrame:
    """The latest version of each row of an SCD type 2 table."""
    return scd2.where(F.col("__END_AT").isNull()).drop("__START_AT", "__END_AT")


def cdc_sequence() -> Column:
    return F.struct(*CDC_SEQUENCE)


def outcomes(typed: DataFrame, encounters: DataFrame) -> DataFrame:
    """One row per escalation (the first copy that landed), with the stay's site.

    `encounters` is the current Silver encounters table; outcomes for unknown stays keep a NULL
    site_id (the pipeline counts them with an expectation rather than hiding them).
    """
    first = Window.partitionBy("encounter_id", "event_type", "event_ts").orderBy(
        "recorded_ts", "_ingest_ts", "_source_file"
    )
    deduped = (
        typed.withColumn("_n", F.row_number().over(first))
        .where("_n = 1")
        .drop("_n", "_rescued_data")
    )
    stays = encounters.select("encounter_id", "site_id")
    return deduped.join(stays, "encounter_id", "left")
