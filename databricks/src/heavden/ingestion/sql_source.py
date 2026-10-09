"""Azure SQL -> Bronze: incremental copies with SQL Server Change Tracking (Plan.md §7.2).

Runs as a serverless job task (`databricks/resources/sql_ingest.job.yml`). Spark reads the
source with `remote_query` over the Unity Catalog connection `heavden_sql`, so the SQL login
stays in Unity Catalog and never appears in this code or the job.

Each run:
1. one query reads the database's current Change Tracking version (`to_version`) and every
   table's minimum valid version (older changes have been cleaned up);
2. per table: with no watermark yet, a watermark older than the minimum valid version, or a
   table that was dropped and recreated since the last run, it copies the whole table
   (`_change_op` = 'S'); otherwise it reads the changes in (watermark, to_version] from
   `CHANGETABLE(CHANGES ...)` (`_change_op` = 'I' or 'U');
3. appends the rows to `bronze.sql_<table>` and then logs `to_version` in `bronze.sql_watermarks`
   (with row counts per change type), which is the next run's watermark.

Delivery is at-least-once: a run that fails between the two writes re-reads the same changes
next time, and Silver's AUTO CDC orders rows by (last_updated, version, ingestion time), so a
second copy of a row changes nothing. The simulation never deletes rows
(a discharge is an update), so a delete ('D') fails the run after it is logged: Silver would
otherwise keep a row that no longer exists.

Recreating a table (`generator.load --replace`) records no deletes and doesn't advance the
Change Tracking version, so reading changes would miss rows that disappeared. The job therefore
keeps each table's creation time (`sys.tables.create_date`) with its watermark and takes a
snapshot when it changes. Silver applies snapshot rows as upserts, so a row missing from a new
snapshot isn't removed there: after reloading *different* data, fully refresh the pipeline.

The file needs only the standard library and PySpark, so the job runs it as a plain Python file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ENVIRONMENTS = ("dev", "staging", "prod")

# Source tables and their primary keys (generator/sql/azure_sql_schema.sql).
TABLES: dict[str, tuple[str, ...]] = {
    "sites": ("site_id",),
    "units": ("unit_id",),
    "patients": ("patient_id",),
    "conditions": ("patient_id", "code", "start_date"),
    "medications": ("patient_id", "code", "start_ts"),
    "encounters": ("encounter_id",),
    "nurse_observations": ("encounter_id", "obs_ts"),
    "device_assignments": ("device_id", "start_ts"),
}

WATERMARKS_TABLE = "sql_watermarks"


@dataclass(frozen=True)
class Read:
    """What one run reads from one table."""

    table: str
    mode: str  # "snapshot" | "changes"
    from_version: int | None  # exclusive; None for a snapshot
    to_version: int  # inclusive
    table_created: dt.datetime | None = None  # the source table's create_date


# --- T-SQL sent to the source -----------------------------------------------------------
# The queries reach Spark as string literals inside remote_query(...), so they contain no
# quotes or backslashes (checked in `_literal`). Names come from TABLES and ENVIRONMENTS only.


def versions_query() -> str:
    """The current version, and each change-tracked table's minimum valid version and creation
    time."""
    return (
        "SELECT CHANGE_TRACKING_CURRENT_VERSION() AS current_version, "
        "s.name AS schema_name, t.name AS table_name, c.min_valid_version, "
        "t.create_date AS table_created "
        "FROM sys.change_tracking_tables AS c "
        "JOIN sys.tables AS t ON t.object_id = c.object_id "
        "JOIN sys.schemas AS s ON s.schema_id = t.schema_id"
    )


def snapshot_query(schema: str, table: str) -> str:
    return f"SELECT t.* FROM {_name(schema, table)} AS t"


def changes_query(schema: str, read: Read) -> str:
    """Rows changed in (from_version, to_version], as they are now, with the change type.

    CHANGETABLE returns one row per changed key with its latest version, so a row changed again
    after `to_version` is left for the next run. A deleted row has only NULLs from the table.
    """
    name = _name(schema, read.table)
    on = " AND ".join(f"t.{key} = c.{key}" for key in TABLES[read.table])
    return (
        f"SELECT t.*, c.SYS_CHANGE_VERSION AS _change_version, "
        f"c.SYS_CHANGE_OPERATION AS _change_op "
        f"FROM CHANGETABLE(CHANGES {name}, {int(read.from_version)}) AS c "
        f"LEFT JOIN {name} AS t ON {on} "
        f"WHERE c.SYS_CHANGE_VERSION <= {int(read.to_version)}"
    )


def _name(schema: str, table: str) -> str:
    if schema not in ENVIRONMENTS or table not in TABLES:
        raise ValueError(f"unknown source table {schema}.{table}")
    return f"{schema}.{table}"


# --- Deciding what to read ---------------------------------------------------------------


def plan_reads(
    schema: str,
    versions: list[dict[str, Any]],
    watermarks: dict[str, int],
    created: dict[str, dt.datetime | None] | None = None,
) -> list[Read]:
    """One Read per table, from the versions query and the last run's watermarks.

    `created` holds each table's creation time as recorded by the last run. A different time
    now means the table was recreated, so its old change history no longer describes it. An
    unknown time (None: rows written before 10 Oct 2026 didn't record it) also takes a snapshot,
    because a recreation can't be ruled out.
    """
    created = created or {}
    rows = [r for r in versions if r["schema_name"] == schema]
    min_valid = {r["table_name"]: r["min_valid_version"] for r in rows}
    now_created = {r["table_name"]: r.get("table_created") for r in rows}
    missing = [t for t in TABLES if t not in min_valid]
    if missing:
        raise RuntimeError(f"Change Tracking is not enabled on {schema}.{', '.join(missing)}")
    current = int(rows[0]["current_version"])

    reads = []
    for table in TABLES:
        last = watermarks.get(table)
        recreated = created.get(table) != now_created[table]
        if last is None or last < min_valid[table] or recreated:
            reads.append(Read(table, "snapshot", None, current, now_created[table]))
        else:
            reads.append(Read(table, "changes", last, current, now_created[table]))
    return reads


def with_retry(
    action: Callable[[], Any], attempts: int = 6, wait_seconds: float = 20, sleep=time.sleep
) -> Any:
    """Run `action`, waiting while a paused serverless database wakes up (error 40613).

    The same rule as generator.source_db.connect_with_retry; repeated here so this file has no
    dependencies.
    """
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            waking = "40613" in str(error) or "is not currently available" in str(error)
            if not waking or attempt == attempts:
                raise
            print(f"Source database is waking up (attempt {attempt}/{attempts}); retrying...")
            sleep(wait_seconds)
    raise AssertionError("unreachable")


# --- Spark ---------------------------------------------------------------------------------


def _literal(text: str) -> str:
    if "'" in text or "\\" in text:
        raise ValueError(f"not safe as a SQL string literal: {text!r}")
    return f"'{text}'"


def remote(spark, connection: str, database: str, query: str):
    """A DataFrame over a T-SQL query run on the source database."""
    return spark.sql(
        f"SELECT * FROM remote_query({_literal(connection)}, "
        f"database => {_literal(database)}, query => {_literal(query)})"
    )


def read_versions(spark, connection: str, database: str, sleep=time.sleep) -> list[dict[str, Any]]:
    """The versions query, retried while the database wakes up.

    The first query of a run is the one that wakes a paused database. spark.sql() connects
    while analysing the query (to learn its columns), so the whole call sits inside the retry.
    """
    return with_retry(
        lambda: [
            row.asDict() for row in remote(spark, connection, database, versions_query()).collect()
        ],
        sleep=sleep,
    )


def read_watermarks(
    spark, catalog: str, schema: str
) -> tuple[dict[str, int], dict[str, dt.datetime | None]]:
    """({table: last to_version}, {table: creation time recorded with it})."""
    table = f"{catalog}.bronze.{WATERMARKS_TABLE}"
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table} ("
        "source_schema STRING, table_name STRING, mode STRING, from_version BIGINT, "
        "to_version BIGINT, rows BIGINT, inserts BIGINT, updates BIGINT, deletes BIGINT, "
        "ingest_ts TIMESTAMP, table_created TIMESTAMP) "
        "COMMENT 'Change Tracking watermarks: one row per source table per ingestion run. "
        "The latest to_version per table is where the next run starts.'"
    )
    if "table_created" not in spark.table(table).columns:  # created before 10 Oct 2026
        spark.sql(f"ALTER TABLE {table} ADD COLUMNS (table_created TIMESTAMP)")
    latest = spark.sql(
        f"SELECT table_name, to_version, table_created FROM {table} "
        "WHERE source_schema = :schema "
        "QUALIFY row_number() OVER (PARTITION BY table_name ORDER BY ingest_ts DESC) = 1",
        args={"schema": schema},
    ).collect()
    return (
        {row.table_name: int(row.to_version) for row in latest},
        {row.table_name: row.table_created for row in latest},
    )


def ingest(spark, catalog: str, schema: str, connection: str, database: str) -> list[dict]:
    from pyspark.sql import functions as F

    run_ts = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    watermarks_table = f"{catalog}.bronze.{WATERMARKS_TABLE}"
    watermarks, created = read_watermarks(spark, catalog, schema)
    versions = read_versions(spark, connection, database)
    log = []
    for read in plan_reads(schema, versions, watermarks, created):
        target = f"{catalog}.bronze.sql_{read.table}"
        if read.mode == "snapshot":
            df = remote(spark, connection, database, snapshot_query(schema, read.table))
            df = df.withColumn("_change_version", F.lit(read.to_version).cast("bigint"))
            df = df.withColumn("_change_op", F.lit("S"))
        else:
            df = remote(spark, connection, database, changes_query(schema, read))
        df.withColumn("_ingest_ts", F.lit(run_ts).cast("timestamp")).write.mode(
            "append"
        ).saveAsTable(target)

        # Count what landed from Bronze (Delta), not from the source, so it is read only once.
        ops = {
            row._change_op: row.n
            for row in spark.table(target)
            .where(F.col("_ingest_ts") == F.lit(run_ts).cast("timestamp"))
            .groupBy("_change_op")
            .agg(F.count("*").alias("n"))
            .collect()
        }
        entry = {
            "source_schema": schema,
            "table_name": read.table,
            "mode": read.mode,
            "from_version": read.from_version,
            "to_version": read.to_version,
            "rows": sum(ops.values()),
            "inserts": ops.get("I", 0) + ops.get("S", 0),
            "updates": ops.get("U", 0),
            "deletes": ops.get("D", 0),
            "ingest_ts": run_ts,
            "table_created": read.table_created,
        }
        # Logged per table, so a failure on a later table doesn't make this one re-read.
        spark.createDataFrame([entry], schema=spark.table(watermarks_table).schema).write.mode(
            "append"
        ).saveAsTable(watermarks_table)
        log.append(entry)
        print(f"{target}: {read.mode} ({read.from_version}, {read.to_version}] -> {ops or 0}")

    deleted = [entry["table_name"] for entry in log if entry["deletes"]]
    if deleted:
        raise RuntimeError(
            f"Rows were deleted in {schema}.{', '.join(deleted)}; the source contract says it "
            "never deletes. They landed in Bronze with _change_op = 'D' and NULL columns: "
            "check them before the next Silver run."
        )
    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="e.g. heavden_dev")
    parser.add_argument("--sql-schema", required=True, choices=ENVIRONMENTS)
    parser.add_argument("--database", required=True, help="Azure SQL database, e.g. heavden")
    parser.add_argument("--connection", required=True, help="Unity Catalog connection name")
    args = parser.parse_args(argv)

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    ingest(spark, args.catalog, args.sql_schema, args.connection, args.database)


if __name__ == "__main__":
    main()
