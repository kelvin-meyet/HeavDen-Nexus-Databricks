"""The demo snapshot: the gold tables, the model and the RAG index as files (Plan.md §11, §13).

In Phase 0 the snapshot is built locally from the generator (`backend/scripts/
build_demo_snapshot.py`). In Phase 6 the same layout is exported from Databricks, and the
backend keeps serving it after the credits end. Layout:

    manifest.json                  as_of, model version, risk bands, row counts
    gold/<table>.parquet           one file per gold table (queried as gold.<table> in DuckDB)
    model/model.joblib             the champion CalibratedModel
    model/background.parquet       training rows that explanations are measured against
    rag/                           the FAISS retriever files (heavden.agent.retrieval)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from heavden.analytics import gold
from heavden.ml import scoring
from heavden.ml.train import CalibratedModel

SCHEMA_VERSION = 1
GOLD_TABLES = (
    "patient_hour_features",
    "risk_scores",
    "alerts_fact",
    "site_kpis_hourly",
    "device_health_daily",
    "encounters",
)
BACKGROUND_ROWS = 2000
FULL_CENSUS = 0.5  # "now" is the latest hour with at least half the usual census


def snapshot_as_of(table: pd.DataFrame) -> pd.Timestamp:
    """The latest prediction hour with a full ward (a stray stay can run to the very end)."""
    census = table.groupby("prediction_ts").size()
    return census[census >= FULL_CENSUS * census.median()].index.max()


@dataclass
class Manifest:
    as_of: str  # the snapshot's "now": the latest prediction hour
    data_start: str
    model_name: str
    model_version: str
    model_type: str
    bands: dict[str, float]
    rows: dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema_version: int = SCHEMA_VERSION
    synthetic: bool = True

    @property
    def as_of_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of)

    @property
    def risk_bands(self) -> scoring.RiskBands:
        return scoring.RiskBands(**self.bands)


def patient_labels(patient_ids: pd.Series) -> pd.Series:
    """Short display ids (P-1001, P-1002, ...) in place of the Synthea UUIDs."""
    unique = sorted(set(patient_ids))
    lookup = {pid: f"P-{1001 + i}" for i, pid in enumerate(unique)}
    return patient_ids.map(lookup)


def encounters_dimension(activity, table: pd.DataFrame) -> pd.DataFrame:
    """One row per stay for the app: masked patient label, age, sex, conditions, where they are.

    No names or dates of birth: governance (Plan.md §15) masks them even though they're fake.
    """
    enc = activity.encounters.copy()
    profile = activity.patients.set_index("patient_id")
    latest = table.sort_values("prediction_ts").groupby("encounter_id").tail(1)
    flags = ["copd", "heart_failure", "diabetes", "ckd", "hypertension", "atrial_fibrillation"]
    out = pd.DataFrame(
        {
            "encounter_id": enc["encounter_id"],
            "patient_id": enc["patient_id"],
            "patient_label": patient_labels(enc["patient_id"]),
            "age": enc["patient_id"].map(profile["age"]),
            "sex": enc["patient_id"].map(profile["sex"]),
            "conditions": enc["patient_id"].map(
                lambda p: (
                    ", ".join(f.replace("_", " ") for f in flags if profile.at[p, f]) or "none"
                )
            ),
            "site_id": enc["site_id"],
            "unit_id": enc["encounter_id"]
            .map(latest.set_index("encounter_id")["unit_id"])
            .fillna(enc["unit_id"]),
            "bed_id": enc["bed_id"],
            "admit_ts": enc["admit_ts"],
            "discharge_ts": enc["discharge_ts"],
        }
    )
    return out


def build_snapshot(
    out_dir: Path,
    table: pd.DataFrame,
    activity,
    outcomes: pd.DataFrame,
    readings: pd.DataFrame,
    model: CalibratedModel,
    background: pd.DataFrame,
    bands: scoring.RiskBands,
    model_name: str,
    model_version: str,
    rag_dir: Path | None = None,
) -> Manifest:
    """Score every patient-hour, derive the gold tables and write the snapshot to `out_dir`."""
    out_dir = Path(out_dir)
    as_of = snapshot_as_of(table)
    table = table[table["prediction_ts"] <= as_of]
    risk = scoring.score_table(model, table, bands, background, model_version)
    alerts = gold.alerts_fact(risk, outcomes, as_of)
    tables = {
        "patient_hour_features": table,
        "risk_scores": risk,
        "alerts_fact": alerts,
        "site_kpis_hourly": gold.site_kpis_hourly(risk, alerts, outcomes),
        "device_health_daily": gold.device_health_daily(
            readings, activity.device_assignments, activity.start, activity.end
        ),
        "encounters": encounters_dimension(activity, table),
    }

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "gold").mkdir(parents=True)
    (out_dir / "model").mkdir()
    for name, frame in tables.items():
        frame.to_parquet(out_dir / "gold" / f"{name}.parquet", index=False)
    joblib.dump(model, out_dir / "model" / "model.joblib")
    sample = background.sample(min(BACKGROUND_ROWS, len(background)), random_state=0)
    sample.to_parquet(out_dir / "model" / "background.parquet", index=False)
    if rag_dir is not None:
        shutil.copytree(rag_dir, out_dir / "rag")

    manifest = Manifest(
        as_of=as_of.isoformat(),
        data_start=table["prediction_ts"].min().isoformat(),
        model_name=model_name,
        model_version=str(model_version),
        model_type=type(model.base).__name__,
        bands=bands.to_dict(),
        rows={name: len(frame) for name, frame in tables.items()},
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=1))
    return manifest


@dataclass
class Snapshot:
    """A loaded snapshot: manifest, model and background (tables are queried via DuckDB)."""

    root: Path
    manifest: Manifest
    model: CalibratedModel
    background: pd.DataFrame

    @classmethod
    def load(cls, root: Path) -> Snapshot:
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No demo snapshot at {root}. Build one with "
                "`uv run python backend/scripts/build_demo_snapshot.py`."
            )
        manifest = Manifest(**json.loads(manifest_path.read_text()))
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"snapshot schema {manifest.schema_version}, expected {SCHEMA_VERSION}"
            )
        return cls(
            root=root,
            manifest=manifest,
            model=joblib.load(root / "model" / "model.joblib"),
            background=pd.read_parquet(root / "model" / "background.parquet"),
        )

    def table_path(self, name: str) -> Path:
        return self.root / "gold" / f"{name}.parquet"

    @property
    def rag_dir(self) -> Path | None:
        path = self.root / "rag"
        return path if path.exists() else None


def to_records(frame: pd.DataFrame) -> list[dict]:
    """JSON-friendly rows: timestamps as ISO strings, NaN as None."""
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].map(lambda t: None if pd.isna(t) else t.isoformat())
    frame = frame.astype(object).where(frame.notna(), None)
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        for row in frame.to_dict("records")
    ]
