import datetime as dt

import numpy as np
import pandas as pd
import pytest

from generator import activity, hospital, source_db, truth
from generator.synthea import CONDITION_KEYWORDS

START = hospital.SIM_START
HOURS = 48
REFERENCE = START  # Synthea's reference date is the simulation start


def _population(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "first_name": [f"First{i}" for i in range(n)],
            "last_name": [f"Last{i}" for i in range(n)],
            "birth_date": [
                dt.date(1950, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 20000, n)
            ],
            "age": rng.integers(18, 95, size=n),
            "sex": rng.choice(["F", "M"], size=n),
        }
    )
    for flag in [*CONDITION_KEYWORDS, "on_beta_blocker"]:
        df[flag] = rng.random(n) < 0.1
    df["n_conditions"] = df[list(CONDITION_KEYWORDS)].sum(axis=1)
    return df


def _synthea(patient_ids: list[str]) -> dict[str, pd.DataFrame]:
    t = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    p0, p1 = patient_ids[0], patient_ids[1]
    conditions = pd.DataFrame(
        {
            "PATIENT": [p0, p0, p1, p1, "not-admitted"],
            "CODE": ["44054006", "44054006", "38341003", "195662009", "38341003"],
            "DESCRIPTION": ["Diabetes", "Diabetes", "Hypertension", "Future diagnosis", "x"],
            "START": [
                t("2010-01-01"),
                t("2010-01-01"),
                t("2015-05-05"),
                t("2027-01-01"),
                t("2012-01-01"),
            ],
            "STOP": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )
    medications = pd.DataFrame(
        {
            "PATIENT": [p0, p0, p1],
            "CODE": ["860975", "197361", "314076"],
            "DESCRIPTION": ["Metformin", "Old drug", "Lisinopril"],
            "START": [t("2011-01-01T08:00"), t("2011-01-01T08:00"), t("2016-01-01T08:00")],
            "STOP": [pd.NaT, t("2012-01-01"), t("2030-01-01")],
        }
    )
    return {"conditions": conditions, "medications": medications}


@pytest.fixture(scope="module")
def ward():
    pop = _population()
    inpatients = hospital.admit_inpatients(pop, n=120)
    plan = activity.schedule_activity(pop, inpatients, START, HOURS, seed=1)
    return truth.simulate_ward(inpatients, START, HOURS, seed=2, activity=plan)


def _tables(ward, as_of):
    ids = list(ward.activity.encounters.sort_values("admit_ts")["patient_id"].unique())
    return source_db.source_tables(
        _synthea(ids), ward.activity, ward.nurse_observations, as_of, REFERENCE
    )


@pytest.fixture(scope="module")
def tables(ward):
    return _tables(ward, ward.activity.end)


PRIMARY_KEYS = {
    "sites": ["site_id"],
    "units": ["unit_id"],
    "patients": ["patient_id"],
    "conditions": ["patient_id", "code", "start_date"],
    "medications": ["patient_id", "code", "start_ts"],
    "encounters": ["encounter_id"],
    "nurse_observations": ["encounter_id", "obs_ts"],
    "device_assignments": ["device_id", "start_ts"],
}


def test_tables_match_the_ddl_columns_in_load_order(tables):
    assert tuple(tables) == source_db.TABLE_ORDER
    for name, df in tables.items():
        assert list(df.columns) == source_db.ddl_columns(name), name


@pytest.mark.parametrize("table", source_db.TABLE_ORDER)
def test_primary_keys_are_unique(tables, table):
    assert not tables[table].duplicated(PRIMARY_KEYS[table]).any()


def test_foreign_keys_resolve(tables):
    patients = set(tables["patients"]["patient_id"])
    encounters = set(tables["encounters"]["encounter_id"])
    assert set(tables["encounters"]["patient_id"]) <= patients
    assert set(tables["encounters"]["unit_id"]) <= set(tables["units"]["unit_id"])
    assert set(tables["units"]["site_id"]) <= set(tables["sites"]["site_id"])
    for name in ("conditions", "medications"):
        assert set(tables[name]["patient_id"]) <= patients
    for name in ("nurse_observations", "device_assignments"):
        assert set(tables[name]["encounter_id"]) <= encounters


def test_last_updated_is_set_and_never_in_the_future(ward, tables):
    for name, df in tables.items():
        assert df["last_updated"].notna().all(), name
        assert (df["last_updated"] <= ward.activity.end).all(), name


def test_only_what_exists_at_as_of_is_included(ward):
    as_of = START + pd.Timedelta(hours=12)
    early = _tables(ward, as_of)
    assert (early["encounters"]["admit_ts"] <= as_of).all()
    assert (early["nurse_observations"]["obs_ts"] <= as_of).all()
    assert set(early["patients"]["patient_id"]) == set(early["encounters"]["patient_id"])
    assert len(early["encounters"]) < len(_tables(ward, ward.activity.end)["encounters"])


def test_history_is_limited_to_the_reference_date(tables):
    conditions, meds = tables["conditions"], tables["medications"]
    assert "Future diagnosis" not in set(conditions["description"])
    assert len(conditions) == 2  # the duplicate Diabetes row is dropped
    assert set(meds["description"]) == {"Metformin", "Lisinopril"}  # the stopped drug is not


def test_ddl_batches_target_one_environment():
    batches = source_db.ddl_batches("dev")
    assert batches[0] == "CREATE SCHEMA dev"
    text = "\n".join(batches)
    assert "dbo." not in text and "ALTER DATABASE" not in text
    assert "CREATE TABLE dev.encounters" in text
    assert "ALTER TABLE dev.encounters" in text and "ENABLE CHANGE_TRACKING" in text
    with pytest.raises(ValueError):
        source_db.ddl_batches("dbo; DROP TABLE x")


def test_insert_batches_respect_sql_server_limits(tables):
    table = "nurse_observations"
    n_columns = len(source_db.ddl_columns(table))
    total = 0
    for sql, params in source_db.insert_batches("dev", table, tables[table]):
        assert sql.startswith(f"INSERT INTO dev.{table} (")
        assert len(params) <= source_db.MAX_PARAMS
        assert sql.count("(?") == len(params) // n_columns <= source_db.MAX_ROWS
        total += len(params) // n_columns
    assert total == len(tables[table])


def test_values_are_converted_for_the_driver(tables):
    sql, params = next(source_db.insert_batches("dev", "encounters", tables["encounters"]))
    timestamps = [p for p in params if isinstance(p, dt.datetime)]
    assert timestamps and all(p.tzinfo is None for p in timestamps)
    assert None in params  # open stays have no discharge_ts
    assert not any(isinstance(p, (np.generic, pd.Timestamp)) or p is pd.NaT for p in params)


def test_connect_retries_while_the_database_wakes_up():
    calls, waits = [], []

    def open_connection():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("[40613] Database 'heavden' is not currently available")
        return "connection"

    result = source_db.connect_with_retry(open_connection, sleep=waits.append)
    assert result == "connection" and len(calls) == 3 and len(waits) == 2


def test_connect_does_not_retry_other_errors():
    def open_connection():
        raise RuntimeError("[18456] Login failed for user")

    with pytest.raises(RuntimeError, match="18456"):
        source_db.connect_with_retry(open_connection, sleep=lambda _: None)
