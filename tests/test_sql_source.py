import re

import pytest

from generator import source_db
from heavden.ingestion import sql_source
from heavden.ingestion.sql_source import Read


def _ddl_primary_key(table: str) -> tuple[str, ...]:
    text = source_db.SCHEMA_SQL.read_text(encoding="utf-8")
    body = re.search(rf"CREATE TABLE dbo\.{table} \((.*?)\n\);", text, flags=re.DOTALL).group(1)
    composite = re.search(r"PRIMARY KEY \(([^)]*)\)", body)
    if composite:
        return tuple(c.strip() for c in composite.group(1).split(","))
    return tuple(re.findall(r"^\s*(\w+)\s+.*\bPRIMARY KEY\b", body, flags=re.MULTILINE))


def test_tables_and_keys_match_the_source_ddl():
    assert set(sql_source.TABLES) == set(source_db.TABLE_ORDER)
    for table, keys in sql_source.TABLES.items():
        assert keys == _ddl_primary_key(table), table


def _versions(schema="dev", current=100, min_valid=10, skip=()):
    rows = [
        {"current_version": current, "schema_name": s, "table_name": t, "min_valid_version": mv}
        for s, mv in ((schema, min_valid), ("staging", 1))
        for t in sql_source.TABLES
        if (s, t) not in skip
    ]
    return rows


def test_first_run_takes_snapshots_up_to_the_current_version():
    reads = sql_source.plan_reads("dev", _versions(), watermarks={})
    assert {r.mode for r in reads} == {"snapshot"}
    assert {r.to_version for r in reads} == {100}
    assert [r.table for r in reads] == list(sql_source.TABLES)


def test_later_runs_read_changes_since_the_watermark():
    watermarks = dict.fromkeys(sql_source.TABLES, 40)
    reads = sql_source.plan_reads("dev", _versions(), watermarks)
    assert {(r.mode, r.from_version, r.to_version) for r in reads} == {("changes", 40, 100)}


def test_a_watermark_older_than_the_retained_changes_forces_a_snapshot():
    watermarks = dict.fromkeys(sql_source.TABLES, 40) | {"encounters": 5}
    reads = {r.table: r for r in sql_source.plan_reads("dev", _versions(min_valid=10), watermarks)}
    assert reads["encounters"].mode == "snapshot"
    assert reads["sites"].mode == "changes"


def test_a_table_without_change_tracking_fails_the_run():
    with pytest.raises(RuntimeError, match="dev.encounters"):
        sql_source.plan_reads("dev", _versions(skip={("dev", "encounters")}), {})


def test_changes_query_bounds_the_window_and_joins_on_the_whole_key():
    query = sql_source.changes_query("dev", Read("conditions", "changes", 40, 100))
    assert "CHANGETABLE(CHANGES dev.conditions, 40)" in query
    assert "c.SYS_CHANGE_VERSION <= 100" in query
    on = "t.patient_id = c.patient_id AND t.code = c.code AND t.start_date = c.start_date"
    assert on in query


def test_queries_are_safe_as_spark_string_literals():
    queries = [sql_source.versions_query()]
    for table in sql_source.TABLES:
        queries.append(sql_source.snapshot_query("prod", table))
        queries.append(sql_source.changes_query("prod", Read(table, "changes", 1, 2)))
    for query in queries:
        assert sql_source._literal(query) == f"'{query}'"
    with pytest.raises(ValueError):
        sql_source._literal("x'; DROP TABLE y")


def test_only_known_schemas_and_tables_are_queried():
    with pytest.raises(ValueError):
        sql_source.snapshot_query("dbo", "sites")
    with pytest.raises(ValueError):
        sql_source.snapshot_query("dev", "sys.objects")


def test_retry_waits_for_a_waking_database_and_nothing_else():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("[40613] Database 'heavden' is not currently available.")
        return "ok"

    assert sql_source.with_retry(flaky, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2

    def broken():
        raise RuntimeError("Login failed for user")

    with pytest.raises(RuntimeError, match="Login failed"):
        sql_source.with_retry(broken, sleep=sleeps.append)
    assert len(sleeps) == 2
