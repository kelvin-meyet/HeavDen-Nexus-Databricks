"""The snapshot: the gold tables and the model as files (Plan.md §11, §13).

The website's snapshot is exported from prod's Gold on Databricks (`heavden_api.export`, job
`website_snapshot`); the demo snapshot is built locally from the generator (`backend/scripts/
build_demo_snapshot.py`). Both are written by `write_snapshot`.

On Container Apps the snapshot is data, not part of the image. The export publishes it to the
volume `heavden_prod.ml.website` (`publish_snapshot`: `snapshots/<name>/` and its pointer, which
names it and lists each file's size and SHA-256; then, unless a release stages it for the new
image's smoke test, `current-v<format>.json`). That volume is a folder of the `website` container
in Azure Blob Storage, and the API downloads what the pointer for its own format names from there
when it starts (`fetch_snapshot` with `blob_reader`), so starting needs no Databricks. A new
snapshot needs no new image, and pointing back (`point_to`) is the rollback. Layout:

    manifest.json                  as_of, model version, risk bands, row counts, library versions
    gold/<table>.parquet           one file per gold table (queried as gold.<table> in DuckDB)
    model/model.joblib             the champion CalibratedModel
    model/background.parquet       training rows that explanations are measured against
    model/check.parquet            patient-hours with the risk the model gave them where the
                                   snapshot was written: loading re-scores them (`check_model`)
    rag/                           demo only: the FAISS retriever files (the image has its own,
                                   built from the documents in git: heavden_api.documents)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd

from heavden.analytics import gold
from heavden.ml import scoring
from heavden.ml.train import CalibratedModel

# The snapshot's format. Raise it with any change the previous code can't read (a file, a column,
# a manifest field it needs): the export then writes a pointer only the new code reads (POINTER).
# 2: model/check.parquet and the library versions.
SCHEMA_VERSION = 2
GOLD_TABLES = (
    "patient_hour_features",
    "risk_scores",
    "alerts_fact",
    "escalations_fact",
    "site_kpis_hourly",
    "device_health_daily",
    "encounters",
)
BACKGROUND_ROWS = 2000
FULL_CENSUS = 0.5  # "now" is the latest hour with at least half the usual census
CHECK_ROWS = 200
CHECK_RISK = "check_risk"
# the same libraries give the same floats; any real difference is far larger
CHECK_TOLERANCE = 1e-9
LIBRARIES = ("scikit-learn", "lightgbm", "pandas", "numpy")


def library_versions() -> dict[str, str]:
    return {name: version(name) for name in LIBRARIES}


def snapshot_as_of(table: pd.DataFrame) -> pd.Timestamp:
    """The latest prediction hour with a full ward (a stray stay can run to the very end)."""
    return full_ward_hour(table.groupby("prediction_ts").size())


def full_ward_hour(census: pd.Series) -> pd.Timestamp:
    """The latest hour of `census` (patients per prediction hour) with a full ward."""
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
    libraries: dict[str, str] = field(default_factory=library_versions)  # where it was written

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
    as_of = snapshot_as_of(table)
    table = table[table["prediction_ts"] <= as_of]
    risk = scoring.score_table(model, table, bands, background, model_version)
    alerts = gold.alerts_fact(risk, outcomes, as_of)
    tables = {
        "patient_hour_features": table,
        "risk_scores": risk,
        "alerts_fact": alerts,
        "escalations_fact": gold.escalations_fact(risk, outcomes, as_of),
        "site_kpis_hourly": gold.site_kpis_hourly(risk, alerts, outcomes),
        "device_health_daily": gold.device_health_daily(
            readings, activity.device_assignments, activity.start, activity.end
        ),
        "encounters": encounters_dimension(activity, table),
    }
    return write_snapshot(
        out_dir, tables, model, background, bands, model_name, model_version, as_of, rag_dir
    )


def check_hours(model: CalibratedModel, features: pd.DataFrame) -> pd.DataFrame:
    """CHECK_ROWS patient-hours with the risk `model` gives them here (`CHECK_RISK`)."""
    rows = features.sample(min(CHECK_ROWS, len(features)), random_state=0)
    return rows.assign(**{CHECK_RISK: model.predict_proba(rows)[:, 1]})


def write_snapshot(
    out_dir: Path,
    tables: dict[str, pd.DataFrame],
    model: CalibratedModel,
    background: pd.DataFrame,
    bands: scoring.RiskBands,
    model_name: str,
    model_version: str,
    as_of: pd.Timestamp,
    rag_dir: Path | None = None,
) -> Manifest:
    """Write a snapshot of the gold `tables` (every one of GOLD_TABLES) and `model` to
    `out_dir`, with background rows sampled from `background` (the model's training rows) and
    check hours scored here."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "gold").mkdir(parents=True)
    (out_dir / "model").mkdir()
    for name in GOLD_TABLES:
        tables[name].to_parquet(out_dir / "gold" / f"{name}.parquet", index=False)
    joblib.dump(model, out_dir / "model" / "model.joblib")
    sample = background.sample(min(BACKGROUND_ROWS, len(background)), random_state=0)
    sample.to_parquet(out_dir / "model" / "background.parquet", index=False)
    check = check_hours(model, tables["patient_hour_features"])
    check.to_parquet(out_dir / "model" / "check.parquet", index=False)
    if rag_dir is not None:
        shutil.copytree(rag_dir, out_dir / "rag")

    manifest = Manifest(
        as_of=as_of.isoformat(),
        data_start=tables["patient_hour_features"]["prediction_ts"].min().isoformat(),
        model_name=model_name,
        model_version=str(model_version),
        model_type=type(model.base).__name__,
        bands=bands.to_dict(),
        rows={name: len(tables[name]) for name in GOLD_TABLES},
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=1))
    return manifest


def check_model(root: Path, manifest: Manifest) -> CalibratedModel:
    """Load the snapshot's model and re-score its check hours: it must give the risks it gave
    where the snapshot was written. A model pickled by other library versions can fail to load,
    fail to score or score differently (20 Oct 2026: prod's v1, pickled by scikit-learn 1.7.2,
    can't score in 1.9.1), and the website would show wrong numbers without this."""
    here = library_versions()
    versions = ", ".join(
        f"{name} {manifest.libraries.get(name, '?')} there, {here[name]} here" for name in here
    )
    check = pd.read_parquet(root / "model" / "check.parquet")
    try:
        model = joblib.load(root / "model" / "model.joblib")
        risk = model.predict_proba(check.drop(columns=CHECK_RISK))[:, 1]
    except Exception as error:
        raise RuntimeError(
            f"snapshot {root}: its model can't score here ({type(error).__name__}: {error}). "
            f"Libraries: {versions}"
        ) from error
    gap = float(np.max(np.abs(risk - check[CHECK_RISK].to_numpy())))
    if gap > CHECK_TOLERANCE:
        raise RuntimeError(
            f"snapshot {root}: its model scores its {len(check)} check hours differently here "
            f"(max gap {gap:.2e}). Libraries: {versions}"
        )
    return model


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
                f"snapshot schema {manifest.schema_version}, expected {SCHEMA_VERSION}: "
                "rebuild it (the demo: backend/scripts/build_demo_snapshot.py)"
            )
        return cls(
            root=root,
            manifest=manifest,
            model=check_model(root, manifest),
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


# One pointer per snapshot format: an image reads only the one for the format its code reads, so a
# snapshot in a newer format (written by a newer release's export) never reaches an older image.
POINTER = f"current-v{SCHEMA_VERSION}.json"
_NAME = re.compile(r"[\w.-]+")


def staged_pointer(name: str) -> str:
    """Where a published snapshot's own pointer waits until the website is pointed at it."""
    return f"snapshots/{name}.json"


def file_listing(folder: Path) -> dict[str, dict]:
    """Every file of a snapshot folder with its size and SHA-256, as its pointer lists them."""
    folder = Path(folder)
    return {
        path.relative_to(folder).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def publish_snapshot(folder: Path, volume: Path, name: str, move_pointer: bool = True) -> dict:
    """Copy a snapshot folder to `<volume>/snapshots/<name>/` and write its pointer next to it
    (`snapshots/<name>.json`: the name, the format, every file's size and SHA-256). With
    `move_pointer` it is copied to POINTER last, so the website serves the new snapshot from its
    next start and the previous one until then. Without (a release), the snapshot waits there
    until the new image has passed its smoke test on it (`point_to`). `volume` is a volume path
    (`/Volumes/...`). Files are copied without their metadata: a volume can't set any."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"a snapshot name is letters, digits, '.', '-' or '_', not {name!r}")
    target = Path(volume) / "snapshots" / name
    if target.exists():
        raise FileExistsError(f"{target} exists: a published snapshot is never changed")
    for path in file_listing(folder):
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(folder) / path, target / path)
    pointer = {"snapshot": name, "schema_version": SCHEMA_VERSION, "files": file_listing(target)}
    text = json.dumps(pointer, indent=1)
    (Path(volume) / staged_pointer(name)).write_text(text)
    if move_pointer:
        (Path(volume) / POINTER).write_text(text)
    return pointer


def read_pointer(read: Callable[[str], bytes], path: str = POINTER) -> dict:
    """A pointer, refused unless it names a snapshot in the format this code reads."""
    pointer = json.loads(read(path))
    if not _NAME.fullmatch(str(pointer.get("snapshot"))):
        raise ValueError(f"{path} names {pointer.get('snapshot')!r}, not a snapshot")
    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: snapshot format {pointer.get('schema_version')}, this code reads "
            f"{SCHEMA_VERSION}"
        )
    return pointer


def point_to(read: Callable[[str], bytes], write: Callable[[str, bytes], None], name: str) -> dict:
    """Make `name` the snapshot the website serves from its next start: copy its own pointer to
    POINTER. Also the rollback, to an earlier snapshot's name."""
    pointer = read_pointer(read, staged_pointer(name))
    if pointer["snapshot"] != name:
        raise ValueError(f"{staged_pointer(name)} names {pointer['snapshot']!r}, not {name!r}")
    write(POINTER, json.dumps(pointer, indent=1).encode())
    return pointer


def fetch_snapshot(read: Callable[[str], bytes], into: Path, pointer: str = POINTER) -> Path:
    """Download the snapshot `pointer` names (by default the one the website serves) into
    `into/<name>/`, checking every file against the size and SHA-256 listed there, and return
    that folder. `read(path)` gives a file of the volume (`blob_reader` on Container Apps)."""
    listing = read_pointer(read, pointer)
    name = listing["snapshot"]
    folder = Path(into) / name
    for path, expected in listing["files"].items():
        parts = PurePosixPath(path).parts
        if PurePosixPath(path).is_absolute() or ".." in parts:
            raise ValueError(f"{pointer} lists {path!r}, outside the snapshot")
        data = read(f"snapshots/{name}/{path}")
        found = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if found != expected:
            raise ValueError(f"snapshot {name}: {path} is not the file {pointer} lists")
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_bytes(data)
    return folder


def blob_reader(url: str, token: Callable[[], str]) -> Callable[[str], bytes]:
    """Files under `url` (https://<account>.blob.core.windows.net/<container>/<folder>) read
    from Azure Blob Storage with an Entra ID token: the website's managed identity, which has
    Storage Blob Data Reader on that container and nothing else."""

    def read(path: str) -> bytes:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/{quote(path)}",
            # an Entra ID token needs the service version named (2017-11-09 or later)
            headers={"Authorization": f"Bearer {token()}", "x-ms-version": "2021-08-06"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    return read
