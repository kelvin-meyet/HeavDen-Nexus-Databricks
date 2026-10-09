import json

import numpy as np
import pandas as pd

from generator import landing

T0 = pd.Timestamp("2026-11-01 08:00", tz="UTC")


def _readings() -> pd.DataFrame:
    ts = [T0, T0 + pd.Timedelta(minutes=5), T0 + pd.Timedelta(minutes=65)]
    return pd.DataFrame(
        {
            "device_id": ["DEV-A-001", "DEV-A-001", "DEV-B-002"],
            "ts": ts,
            "heart_rate": np.array([88, 90, 72]),
            "resp_rate": np.array([18, 19, 14]),
            "spo2": np.array([95, 94, 98]),
            "temp_c": [37.1, 37.2, 36.8],
            "sbp": np.array([120, 118, 130]),
            "dbp": np.array([80, 79, 85]),
            "motion": [0.1, np.nan, 0.3],  # dropped by the firmware change
            "battery_pct": np.array([99, 99, 87]),
            "firmware": "3.1.4",
        }
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "encounter_id": ["E1", "E2"],
            "event_type": ["rapid_response", "icu_transfer"],
            "event_ts": [T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=20)],
        }
    )


def _lines(content: str) -> list[dict]:
    return [json.loads(line) for line in content.splitlines()]


def test_records_are_plain_json_with_missing_fields_left_out():
    records = _lines(landing.to_ndjson(_readings(), landing.VITALS_FIELDS))
    assert records[0] == {
        "device_id": "DEV-A-001",
        "ts": "2026-11-01T08:00:00Z",
        "heart_rate": 88,
        "resp_rate": 18,
        "spo2": 95,
        "temp_c": 37.1,
        "sbp": 120,
        "dbp": 80,
        "motion": 0.1,
        "battery_pct": 99,
        "firmware": "3.1.4",
    }
    assert "motion" not in records[1]
    assert isinstance(records[0]["heart_rate"], int)


def test_one_vitals_file_per_simulated_hour_without_patient_ids():
    files = landing.landing_files("dev", _readings(), _outcomes(), as_of=T0 + pd.Timedelta(days=2))
    vitals = sorted(p for p in files if p.startswith("dev/vitals/"))
    assert len(vitals) == 2
    assert vitals[0].startswith("dev/vitals/vitals_20261101_080000_")
    assert len(_lines(files[vitals[0]])) == 2
    assert all("patient_id" not in r for p in vitals for r in _lines(files[p]))


def test_outcomes_land_only_once_recorded():
    as_of = T0 + pd.Timedelta(hours=20)  # second event happened, but isn't recorded until +26 h
    files = landing.landing_files("dev", _readings(), _outcomes(), as_of)
    outcomes = [r for p in files if "/outcomes/" in p for r in _lines(files[p])]
    assert outcomes == [
        {
            "patient_id": "p1",
            "encounter_id": "E1",
            "event_type": "rapid_response",
            "event_ts": "2026-11-01T09:00:00Z",
            "recorded_ts": "2026-11-01T15:00:00Z",
        }
    ]
    assert any(p.startswith("dev/outcomes/outcomes_20261101_150000_") for p in files)


def test_file_names_are_stable_and_distinct_per_environment_and_kind():
    name = landing.file_name("dev", "vitals", T0)
    assert name == landing.file_name("dev", "vitals", T0)
    assert name != landing.file_name("prod", "vitals", T0)
    assert name != landing.file_name("dev", "outcomes", T0)
    assert name.endswith(".json")


def test_write_local_replaces_the_previous_simulation(tmp_path):
    landing.write_local({"dev/vitals/old.json": "{}\n"}, tmp_path)
    landing.write_local({"dev/vitals/new.json": "{}\n"}, tmp_path)
    assert [p.name for p in (tmp_path / "dev" / "vitals").iterdir()] == ["new.json"]
