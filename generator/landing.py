"""Device and outcome files: what lands in ADLS `landing/<env>/` for Auto Loader (Plan.md §5.2–5.4).

Monitors don't write to a database; a device gateway drops batches of readings as JSON files.
This module turns one simulation into those files:

    <env>/vitals/vitals_YYYYMMDD_HHMMSS_<uuid>.json      one file per simulated hour
    <env>/outcomes/outcomes_YYYYMMDD_HHMMSS_<uuid>.json  one file per hour with recorded outcomes

Each file is newline-delimited JSON (one object per line). Readings carry `device_id`, never a
patient: Silver links them through `device_assignments`. A field that is missing (NaN, e.g.
`motion` after the firmware change) is left out of the object, as a real device would.

Outcomes are known only some hours after the event (the label delay): each carries `event_ts`
and `recorded_ts`, and only outcomes recorded by the end of the simulation are written.

File names are deterministic (the uuid is derived from the environment, kind and batch time), so
writing the same simulation twice replaces files instead of adding duplicates.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path

import pandas as pd

LABEL_DELAY = pd.Timedelta(hours=6)
BATCH = pd.Timedelta(hours=1)
_NAMESPACE = uuid.UUID("5b0c3c4e-2f6d-4a8e-9d51-4e8f1a6c7d20")  # fixed: stable file names

VITALS_FIELDS = (
    "device_id",
    "ts",
    "heart_rate",
    "resp_rate",
    "spo2",
    "temp_c",
    "sbp",
    "dbp",
    "motion",
    "battery_pct",
    "firmware",
    "etco2",  # only after the firmware change (drift scenario)
)
OUTCOME_FIELDS = ("patient_id", "encounter_id", "event_type", "event_ts", "recorded_ts")


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_value(value):
    """numpy/pandas scalars to plain JSON; None for missing (the caller drops those keys)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return _iso(value)
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def to_ndjson(df: pd.DataFrame, fields: tuple[str, ...]) -> str:
    """One JSON object per row, keys in `fields` order, missing values left out."""
    columns = [f for f in fields if f in df.columns]
    lines = []
    for row in df[columns].itertuples(index=False, name=None):
        values = map(_json_value, row)
        record = {k: v for k, v in zip(columns, values, strict=True) if v is not None}
        lines.append(json.dumps(record, separators=(",", ":")))
    return "\n".join(lines) + "\n" if lines else ""


def file_name(env: str, kind: str, batch_start: pd.Timestamp) -> str:
    stamp = batch_start.tz_convert("UTC").strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{stamp}_{uuid.uuid5(_NAMESPACE, f'{env}/{kind}/{stamp}')}.json"


def recorded_outcomes(outcomes: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Outcomes with their recording time, keeping only those recorded by `as_of`."""
    out = outcomes.assign(recorded_ts=outcomes["event_ts"] + LABEL_DELAY)
    return out[out["recorded_ts"] <= as_of].sort_values("recorded_ts", ignore_index=True)


def landing_files(
    env: str, readings: pd.DataFrame, outcomes: pd.DataFrame, as_of: pd.Timestamp
) -> dict[str, str]:
    """{relative path under the landing container: file content} for one simulation."""
    files = {}
    batches = [
        ("vitals", readings.sort_values(["ts", "device_id"]), "ts", VITALS_FIELDS),
        ("outcomes", recorded_outcomes(outcomes, as_of), "recorded_ts", OUTCOME_FIELDS),
    ]
    for kind, df, time_column, fields in batches:
        for batch_start, batch in df.groupby(df[time_column].dt.floor(BATCH)):
            files[f"{env}/{kind}/{file_name(env, kind, batch_start)}"] = to_ndjson(batch, fields)
    return files


def write_local(files: dict[str, str], root: Path) -> Path:
    """Write the files under `root` (cleared first, so it holds exactly this simulation)."""
    if root.exists():
        shutil.rmtree(root)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    return root


def upload(az: str, local_root: Path, env: str, account: str, replace: bool) -> None:
    """Upload `<local_root>/<env>/` to `landing/<env>/` with the signed-in Azure CLI identity."""
    common = ["--account-name", account, "--auth-mode", "login", "--only-show-errors"]
    common += ["--output", "none"]  # the upload otherwise prints every blob
    if replace:
        subprocess.run(
            [az, "storage", "blob", "delete-batch", "--source", "landing", "--pattern", f"{env}/*"]
            + common,
            check=True,
        )
    subprocess.run(
        [az, "storage", "blob", "upload-batch", "--destination", "landing"]
        + ["--destination-path", env, "--source", str(local_root / env), "--overwrite"]
        + common,
        check=True,
    )
