"""Where the API gets its data: one interface, two implementations.

Routes run the **same SQL** (from `queries.py`) in both modes, against tables named
`gold.<table>` with Databricks-style `:name` parameters:

* `DemoSource`: DuckDB views over the snapshot's Parquet files (this module).
* Live mode (Phase 5): the Databricks SQL warehouse via the SQL connector, authenticated with
  the backend's service principal (OAuth M2M), with `gold` resolving to `heavden_prod.gold`.
"""

from __future__ import annotations

import re
from typing import Protocol

import duckdb
import pandas as pd

from heavden_api.snapshot import GOLD_TABLES, Snapshot

_PARAM = re.compile(r"(?<![:\w]):(\w+)")


class DataSource(Protocol):
    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame: ...


class DemoSource:
    """DuckDB over the snapshot. Timestamps are kept in UTC."""

    def __init__(self, snapshot: Snapshot):
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone = 'UTC'")
        self.con.execute("CREATE SCHEMA gold")
        for name in GOLD_TABLES:
            path = snapshot.table_path(name).as_posix()
            self.con.execute(f"CREATE VIEW gold.{name} AS SELECT * FROM read_parquet('{path}')")

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        # :name (Databricks) -> $name (DuckDB); a cursor per call keeps threads independent
        cursor = self.con.cursor()
        try:
            cursor.execute("SET TimeZone = 'UTC'")  # settings don't carry over to cursors
            return cursor.execute(_PARAM.sub(r"$\1", sql), params or {}).df()
        finally:
            cursor.close()
