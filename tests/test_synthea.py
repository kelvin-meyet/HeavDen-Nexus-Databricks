from pathlib import Path

import pandas as pd

from generator.synthea import SyntheaRun, build_command, build_profiles

AS_OF = pd.Timestamp("2026-11-01", tz="UTC")


def _ts(value: str | None) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") if value else pd.NaT


def _tables() -> dict[str, pd.DataFrame]:
    patients = pd.DataFrame(
        {
            "Id": ["p1", "p2", "p3"],
            "FIRST": ["Ann", "Bob", "Cy"],
            "LAST": ["A", "B", "C"],
            "BIRTHDATE": [_ts("1950-01-01"), _ts("1990-06-01"), _ts("1940-01-01")],
            "DEATHDATE": [pd.NaT, pd.NaT, _ts("2020-01-01")],
            "GENDER": ["F", "M", "M"],
        }
    )
    conditions = pd.DataFrame(
        {
            "PATIENT": ["p1", "p1", "p2", "p2"],
            "DESCRIPTION": [
                "Chronic obstructive bronchitis (disorder)",
                "Diabetes mellitus type 2 (disorder)",
                "Prediabetes (finding)",
                "Essential hypertension (disorder)",
            ],
            "START": [_ts("2015-01-01"), _ts("2018-01-01"), _ts("2019-01-01"), _ts("2010-01-01")],
            "STOP": [pd.NaT, pd.NaT, pd.NaT, _ts("2012-01-01")],  # p2's hypertension resolved
        }
    )
    medications = pd.DataFrame(
        {
            "PATIENT": ["p1", "p2"],
            "DESCRIPTION": ["Metoprolol succinate 100 MG", "Metoprolol succinate 100 MG"],
            "START": [_ts("2020-01-01"), _ts("2020-01-01")],
            "STOP": [pd.NaT, _ts("2021-01-01")],  # p2 stopped it
        }
    )
    return {"patients": patients, "conditions": conditions, "medications": medications}


def test_profiles_exclude_dead_patients_and_compute_age():
    profiles = build_profiles(_tables(), AS_OF).set_index("patient_id")
    assert list(profiles.index) == ["p1", "p2"]
    assert profiles.loc["p1", "age"] == 76


def test_only_active_conditions_and_medications_count():
    profiles = build_profiles(_tables(), AS_OF).set_index("patient_id")
    assert profiles.loc["p1", "copd"] and profiles.loc["p1", "diabetes"]
    assert profiles.loc["p1", "on_beta_blocker"]
    assert not profiles.loc["p2", "hypertension"]  # stopped before AS_OF
    assert not profiles.loc["p2", "on_beta_blocker"]
    assert profiles.loc["p1", "n_conditions"] == 2


def test_prediabetes_is_not_diabetes():
    profiles = build_profiles(_tables(), AS_OF).set_index("patient_id")
    assert not profiles.loc["p2", "diabetes"]


def test_command_is_seeded_and_exports_only_needed_csvs():
    cmd = build_command(SyntheaRun(population=10, seed=7), Path("java"), Path("s.jar"))
    assert cmd[cmd.index("-s") + 1] == "7"
    assert cmd[cmd.index("-p") + 1] == "10"
    included = next(c for c in cmd if c.startswith("--exporter.csv.included_files="))
    assert "patients.csv" in included and "observations.csv" not in included
