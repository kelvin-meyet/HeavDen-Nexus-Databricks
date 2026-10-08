"""Run Synthea and turn its CSV output into patient profiles for the pretend hospital.

Synthea (https://github.com/synthetichealth/synthea) creates realistic synthetic patients with
lifetime medical histories. We use it for *who* the patients are (age, sex, conditions,
medications). Who is currently in hospital, and where, is decided by `hospital.py`.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_JAVA_EXE = "java.exe" if os.name == "nt" else "java"
DEFAULT_JAVA = Path.home() / "tools" / "jdk-21" / "bin" / _JAVA_EXE
DEFAULT_SYNTHEA_JAR = Path.home() / "tools" / "synthea" / "synthea-with-dependencies.jar"

# Synthea writes many CSVs; these are the ones the platform uses.
TABLES = ("patients", "conditions", "medications", "encounters")

# Conditions that change a patient's baseline vitals or their risk of deterioration.
# Regex patterns, matched case-insensitively against Synthea's condition DESCRIPTION text.
CONDITION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "copd": ("chronic obstructive", "pulmonary emphysema"),
    "heart_failure": ("heart failure",),
    "diabetes": ("(?<!pre)diabetes",),  # not "prediabetes"
    "ckd": ("chronic kidney disease",),
    "hypertension": ("hypertension",),
    "atrial_fibrillation": ("atrial fibrillation",),
}

# Beta-blockers lower heart rate, so the vitals simulator needs to know about them.
BETA_BLOCKERS = ("metoprolol", "carvedilol", "atenolol", "propranolol", "bisoprolol", "nebivolol")


@dataclass(frozen=True)
class SyntheaRun:
    """Settings for one reproducible Synthea run."""

    population: int = 2000
    seed: int = 42
    reference_date: str = "20261101"  # YYYYMMDD: the "today" of the generated histories
    min_age: int = 18
    max_age: int = 95
    state: str = "Massachusetts"


def tool_paths() -> tuple[Path, Path]:
    """Java and Synthea jar locations, overridable with HEAVDEN_JAVA and SYNTHEA_JAR."""
    java = Path(os.environ.get("HEAVDEN_JAVA", DEFAULT_JAVA))
    jar = Path(os.environ.get("SYNTHEA_JAR", DEFAULT_SYNTHEA_JAR))
    return java, jar


def build_command(run: SyntheaRun, java: Path, jar: Path) -> list[str]:
    """The exact command line for a run (CSV export only)."""
    return [
        str(java),
        "-jar",
        str(jar),
        "-p",
        str(run.population),
        "-s",
        str(run.seed),
        "-cs",
        str(run.seed),
        "-r",
        run.reference_date,
        "-a",
        f"{run.min_age}-{run.max_age}",
        "--exporter.csv.export=true",
        "--exporter.csv.included_files=" + ",".join(f"{t}.csv" for t in TABLES),
        "--exporter.fhir.export=false",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        run.state,
    ]


def run_synthea(run: SyntheaRun, work_dir: Path, *, force: bool = False) -> Path:
    """Run Synthea in `work_dir` and return the folder holding its CSVs.

    Skips the run if the CSVs already exist (same settings give the same output), unless `force`.
    """
    csv_dir = work_dir / "output" / "csv"
    if (csv_dir / "patients.csv").exists() and not force:
        return csv_dir

    java, jar = tool_paths()
    for tool in (java, jar):
        if not tool.exists():
            raise FileNotFoundError(f"{tool} not found. See CLAUDE.md (Synthea setup).")

    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(run, java, jar), cwd=work_dir, check=True, capture_output=True)
    return csv_dir


def load_tables(csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the Synthea CSVs the platform uses."""
    date_cols = {
        "patients": ["BIRTHDATE", "DEATHDATE"],
        "conditions": ["START", "STOP"],
        "medications": ["START", "STOP"],
        "encounters": ["START", "STOP"],
    }
    tables = {}
    for name in TABLES:
        df = pd.read_csv(csv_dir / f"{name}.csv", dtype=str)
        for col in date_cols[name]:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        tables[name] = df
    return tables


def _flag_patients(df: pd.DataFrame, keywords: tuple[str, ...]) -> set[str]:
    pattern = "|".join(keywords)
    matches = df["DESCRIPTION"].str.contains(pattern, case=False, na=False)
    return set(df.loc[matches, "PATIENT"])


def build_profiles(tables: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per living patient: demographics plus the condition and medication flags.

    Only conditions and medications that are *active* at `as_of` count (no STOP date yet).
    """
    as_of = pd.Timestamp(as_of).tz_localize("UTC") if as_of.tzinfo is None else as_of

    patients = tables["patients"]
    alive = patients[patients["DEATHDATE"].isna() | (patients["DEATHDATE"] > as_of)]
    profiles = pd.DataFrame(
        {
            "patient_id": alive["Id"],
            "first_name": alive["FIRST"],
            "last_name": alive["LAST"],
            "birth_date": alive["BIRTHDATE"].dt.date,
            "sex": alive["GENDER"],
            "age": ((as_of - alive["BIRTHDATE"]).dt.days // 365.25).astype(int),
        }
    ).reset_index(drop=True)

    def active(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["START"] <= as_of) & (df["STOP"].isna() | (df["STOP"] > as_of))]

    conditions = active(tables["conditions"])
    for flag, keywords in CONDITION_KEYWORDS.items():
        profiles[flag] = profiles["patient_id"].isin(_flag_patients(conditions, keywords))

    meds = active(tables["medications"])
    profiles["on_beta_blocker"] = profiles["patient_id"].isin(_flag_patients(meds, BETA_BLOCKERS))

    flags = list(CONDITION_KEYWORDS)
    profiles["n_conditions"] = profiles[flags].sum(axis=1)
    return profiles
