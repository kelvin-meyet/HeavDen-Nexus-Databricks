"""Where the API gets its data: one interface, two implementations.

Routes run the **same SQL** (from `queries.py`) in both modes, against tables named
`gold.<table>` with Databricks-style `:name` parameters:

* `DemoSource`: DuckDB views over the snapshot's Parquet files.
* `LiveSource`: the Databricks SQL warehouse (Statement Execution API) as the backend's own
  identity, with `gold` resolving to `<catalog>.gold` (`heavden_prod`). Once the day's warehouse
  allowance is used up (`chat/limits.py`), it answers from the snapshot instead.

Each also says what "now" is: the snapshot's hour in demo mode, the latest scored hour in live
mode (the hour the Genie space also calls now).
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import duckdb
import numpy as np
import pandas as pd

from heavden_api.chat.limits import AllowanceUsed
from heavden_api.snapshot import GOLD_TABLES, Snapshot

if TYPE_CHECKING:
    from heavden_api.chat.databricks_tools import Warehouse

_PARAM = re.compile(r"(?<![:\w]):(\w+)")


class DataSource(Protocol):
    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame: ...

    def now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """("now", first hour) of the data, in UTC."""
        ...

    def known_now(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """`now()` if it is known without a query (None if not)."""
        ...


class DemoSource:
    """DuckDB over the snapshot. Timestamps are kept in UTC."""

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
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

    def now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        m = self.snapshot.manifest
        return m.as_of_ts, pd.Timestamp(m.data_start)

    def known_now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.now()


def _parameter(name: str, value) -> dict:
    """A query parameter as the Statement Execution API takes it: a string and a SQL type."""
    if value is None:
        return {"name": name, "type": "STRING"}  # no value: NULL
    if isinstance(value, bool | np.bool_):
        return {"name": name, "value": str(bool(value)).lower(), "type": "BOOLEAN"}
    if isinstance(value, int | np.integer):
        return {"name": name, "value": str(int(value)), "type": "INT"}
    if isinstance(value, float | np.floating):
        return {"name": name, "value": repr(float(value)), "type": "DOUBLE"}
    if isinstance(value, datetime):
        utc = pd.Timestamp(value).tz_convert("UTC") if value.tzinfo else pd.Timestamp(value)
        return {"name": name, "value": utc.strftime("%Y-%m-%d %H:%M:%S"), "type": "TIMESTAMP"}
    return {"name": name, "value": str(value), "type": "STRING"}


def _frame(response: dict) -> pd.DataFrame:
    """A statement's rows (all strings in JSON_ARRAY) as a typed frame, times in UTC, so the
    routes see what DuckDB gives them in demo mode."""
    columns = response["manifest"]["schema"].get("columns") or []
    rows = (response.get("result") or {}).get("data_array") or []
    frame = pd.DataFrame(rows, columns=[c["name"] for c in columns], dtype=object)
    for c in columns:
        kind, values = (c.get("type_name") or "STRING").upper(), frame[c["name"]]
        if kind in ("TIMESTAMP", "TIMESTAMP_NTZ", "DATE"):
            frame[c["name"]] = pd.to_datetime(values, utc=True)
        # Python's own parsing: pandas' fast one can change the last digit of a decimal
        elif kind in ("INT", "LONG", "SHORT", "BYTE"):
            frame[c["name"]] = pd.array([None if v is None else int(v) for v in values], "Int64")
        elif kind in ("DOUBLE", "FLOAT", "DECIMAL"):
            frame[c["name"]] = [np.nan if v is None else float(v) for v in values]
        elif kind == "BOOLEAN":
            frame[c["name"]] = values.map({"true": True, "false": False})
    return frame


class LiveSource:
    """The SQL warehouse. Answers are kept per query and parameters: every page query names
    the hour it is about (`:as_of`), so a cached answer stays right until "now" moves, which only
    a data run does. That spares the warehouse (billed while awake) repeat page views. `max_age`
    bounds it anyway (labels can arrive later for the same hours).

    When the warehouse refuses because the day's allowance is used up (`AllowanceUsed`), the
    same SQL runs on `fallback`, the snapshot (prod's Gold as of its export), about the
    snapshot's own hour, so its rows stay consistent even if prod has moved on since."""

    NOW_SQL = (
        "SELECT min(prediction_ts) AS first, max(prediction_ts) AS latest FROM gold.risk_scores"
    )

    def __init__(
        self,
        warehouse: Warehouse,
        now_seconds: float = 1800,
        max_age: float = 3600,
        size: int = 256,
        fallback: DemoSource | None = None,
    ):
        self.warehouse, self.fallback = warehouse, fallback
        self.now_seconds, self.max_age, self.size = now_seconds, max_age, size
        self._cache: OrderedDict[tuple, tuple[float, pd.DataFrame]] = OrderedDict()
        self._now: tuple[float, tuple[pd.Timestamp, pd.Timestamp]] | None = None
        self._lock = threading.Lock()

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        params = params or {}
        key = (sql, tuple(sorted((k, str(v)) for k, v in params.items())))
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < self.max_age:
                self._cache.move_to_end(key)
                return hit[1].copy()
        used = set(_PARAM.findall(sql))  # routes pass a shared dict; send what this query names
        parameters = [_parameter(k, v) for k, v in params.items() if k in used]
        try:
            frame = _frame(self.warehouse.execute(sql, parameters))
        except AllowanceUsed:
            if self.fallback is None:
                raise
            if "as_of" in params:
                params = params | {"as_of": self.fallback.now()[0]}
            return self.fallback.query(sql, params)
        with self._lock:
            self._cache[key] = (time.monotonic(), frame)
            while len(self._cache) > self.size:
                self._cache.popitem(last=False)
        return frame.copy()

    def now(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The latest and first scored hours, asked again every `now_seconds` (30 min: "now"
        only moves with a data run, and each check may wake the warehouse)."""
        with self._lock:
            if self._now and time.monotonic() - self._now[0] < self.now_seconds:
                return self._now[1]
        try:
            row = _frame(self.warehouse.execute(self.NOW_SQL)).iloc[0]
        except AllowanceUsed:
            if self.fallback is None:
                raise
            return self.fallback.now()
        if pd.isna(row["latest"]):
            raise RuntimeError("gold.risk_scores has no scored hours")
        with self._lock:
            self._now = (time.monotonic(), (row["latest"], row["first"]))
            return self._now[1]

    def known_now(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """ "Now" without a query: the warehouse's last answer however old (it only moves with a
        data run), else the snapshot's hour."""
        with self._lock:
            if self._now:
                return self._now[1]
        return self.fallback.now() if self.fallback else None
