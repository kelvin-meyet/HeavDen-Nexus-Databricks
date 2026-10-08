"""A locked-down DuckDB for the assistant's text-to-SQL tool.

Gold tables are copied into an in-memory database (internal ids such as `patient_id` dropped),
then external access is disabled and the configuration locked: even SQL that slipped past
`sql_guard` can't read files, attach databases or change settings. Each query also gets a
time limit.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
import pandas as pd

from heavden.agent.sql_guard import MAX_ROWS, SqlRejected, check_sql
from heavden_api.snapshot import Snapshot

# Tables the assistant may query, with what each row is (for the prompt).
CHAT_TABLES = {
    "site_kpis_hourly": "one row per unit per hour: census, patients per risk band, new alerts, "
    "escalations, mean NEWS2",
    "alerts_fact": "one row per alert (a patient entering the High band; no repeat for the same "
    "stay within 6 h): risk and NEWS2 at alert time, whether an escalation followed within 6 h",
    "escalations_fact": "one row per escalation (rapid_response or icu_transfer): where, when, and "
    "whether the patient was flagged High in the 6 h before",
    "risk_scores": "one row per patient per hour: risk (probability of escalation within 6 h), "
    "risk_band (Low or High; High = alert threshold), NEWS2 total",
    "encounters": "one row per hospital stay: patient_label (e.g. P-1042), age, sex, conditions, "
    "site, unit, bed, admit and discharge times",
    "device_health_daily": "one row per wearable device per day: uptime, outages, stuck-sensor "
    "minutes, battery, firmware, mean SpO2",
    "patient_hour_features": "one row per patient per hour: vital-sign summaries (e.g. "
    "spo2_min_1h, resp_rate_mean_1h, heart_rate_trend_3h), nurse observations, NEWS2 parts",
}
HIDDEN_COLUMNS = {"patient_id", "top_factors", "label", "label_known_at"}
TIMEOUT_SECONDS = 5.0


@dataclass
class QueryResult:
    sql: str  # what actually ran (with the row limit)
    columns: list[str]
    rows: list[list]
    truncated: bool


class SqlSandbox:
    def __init__(self, snapshot: Snapshot):
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone = 'UTC'")
        self.con.execute("CREATE SCHEMA gold")
        self.columns: dict[str, list[tuple[str, str]]] = {}
        for name in CHAT_TABLES:
            path = snapshot.table_path(name).as_posix()
            self.con.execute(f"CREATE TABLE gold.{name} AS SELECT * FROM read_parquet('{path}')")
            present = [
                d[0] for d in self.con.execute(f"SELECT * FROM gold.{name} LIMIT 0").description
            ]
            for column in (c for c in present if c in HIDDEN_COLUMNS):
                self.con.execute(f'ALTER TABLE gold.{name} DROP COLUMN "{column}"')
            self.columns[name] = [
                (row[0], row[1]) for row in self.con.execute(f"DESCRIBE gold.{name}").fetchall()
            ]
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")
        self._lock = threading.Lock()

    def schema_text(self) -> str:
        """Table and column list for the LLM prompt."""
        parts = []
        for name, about in CHAT_TABLES.items():
            cols = ", ".join(f"{c} {t.lower()}" for c, t in self.columns[name])
            parts.append(f"gold.{name}: {about}.\n  columns: {cols}")
        return "\n".join(parts)

    def run(self, sql: str) -> QueryResult:
        """Guard, then run with a row cap and a time limit. Raises `SqlRejected` on failure."""
        safe = check_sql(sql, set(CHAT_TABLES))
        # One query at a time on the main connection: a cursor would be a new connection with
        # the default (local) time zone, and the locked configuration can't be changed.
        with self._lock:
            timer = threading.Timer(TIMEOUT_SECONDS, self.con.interrupt)
            timer.start()
            try:
                frame: pd.DataFrame = self.con.execute(safe).df()
            except duckdb.InterruptException as exc:
                raise SqlRejected(f"query took longer than {TIMEOUT_SECONDS:.0f} s") from exc
            except duckdb.Error as exc:
                raise SqlRejected(f"query failed: {str(exc).splitlines()[0]}") from exc
            finally:
                timer.cancel()
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].map(lambda t: None if pd.isna(t) else t.isoformat())
        frame = frame.astype(object).where(frame.notna(), None)
        rows = [[v.item() if hasattr(v, "item") else v for v in r] for r in frame.values.tolist()]
        return QueryResult(safe, list(frame.columns), rows, truncated=len(rows) >= MAX_ROWS)
