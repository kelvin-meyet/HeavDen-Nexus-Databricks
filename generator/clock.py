"""The hospital's calendar: one set of real dates everywhere.

The hospital starts on `hospital.SIM_START`, a real date in the recent past, and its clock never
runs ahead of the real one. Dates ahead of the reader's calendar make no sense on a dashboard,
and tools that follow the real clock would not see them (Lakehouse Monitoring looks back 30 days).

Two helpers keep it that way:

* `move` shifts every date and time in a table by whole days. The patient histories were
  generated as of another date (`synthea.SyntheaRun.generated_for`); `synthea.load_tables` moves
  them so they end on the hospital's first day. Whole days change nothing about a patient: ages,
  how long a condition has lasted and the hour of day are all unchanged.
* `require_past` refuses a load that would put data in the future.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


def _is_date(value) -> bool:
    return isinstance(value, dt.date) and not isinstance(value, dt.datetime)


def move(df: pd.DataFrame, by: pd.Timedelta) -> pd.DataFrame:
    """A copy of `df` with every date and time column moved by `by`; other columns untouched."""
    if not by:
        return df
    days = by.to_pytimedelta()

    def moved(value):
        return value + days if _is_date(value) else value

    out = df.copy()
    for column in out.columns:
        values = out[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            out[column] = (values + by).astype(values.dtype)  # keep the column's time unit
        elif values.dtype == object and values.map(_is_date).any():  # e.g. birth_date
            out[column] = values.map(moved)
    return out


def require_past(end: pd.Timestamp, now: pd.Timestamp | None = None) -> None:
    """The hospital never runs ahead of the calendar: refuse a load that ends in the future."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if end > now:
        raise SystemExit(
            f"This load ends on {end:%d %b %Y %H:%M} UTC, which is still in the future "
            f"(now: {now:%d %b %Y %H:%M} UTC). Load it once that time has passed."
        )
