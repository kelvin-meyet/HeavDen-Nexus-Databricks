"""Silver transformations on local Spark, checked against the pandas reference.

The vitals go through the real file contract: the generator writes landing files, Spark reads
them the way Bronze does, and Silver must link every reading exactly as `features.py` does.
"""

import datetime as dt

import pandas as pd
import pytest
from test_source_db import HOURS, START, _population

from generator import activity, hospital, landing, truth
from heavden.ingestion.sql_source import TABLES
from heavden.ml import features

pytest.importorskip("pyspark")
from heavden.pipelines import bronze, silver  # noqa: E402  (needs pyspark)

T0 = dt.datetime(2026, 11, 1, 8, 0)


@pytest.fixture(scope="module")
def ward():
    pop = _population()
    inpatients = hospital.admit_inpatients(pop, n=120)
    plan = activity.schedule_activity(pop, inpatients, START, HOURS, seed=1)
    return truth.simulate_ward(inpatients, START, HOURS, seed=2, activity=plan)


@pytest.fixture(scope="module")
def landed(ward, tmp_path_factory):
    """The ward's landing files on disk, as the generator uploads them."""
    files = landing.landing_files("dev", ward.readings, ward.truth.outcomes, ward.activity.end)
    return landing.write_local(files, tmp_path_factory.mktemp("landing")) / "dev"


def _bronze(spark, folder, fields):
    """Read landing files the way Bronze does: Auto Loader's type hints, plus lineage.

    Lines are parsed with from_json rather than spark.read.json, which needs Hadoop's native
    Windows libraries to list files.
    """
    from pyspark.sql import functions as F

    lines = pd.DataFrame(
        [
            (line, str(path))
            for path in sorted(folder.iterdir())
            for line in path.read_text().splitlines()
        ],
        columns=["line", "_source_file"],
    )
    schema = bronze.schema_hints(fields) + ", _rescued_data STRING"
    parsed = spark.createDataFrame(lines).select(
        F.from_json("line", schema).alias("v"), "_source_file"
    )
    return parsed.select("v.*", "_source_file").withColumn("_ingest_ts", F.current_timestamp())


def _readings(spark, rows):
    """Typed readings from dicts; a key left out of a row is NULL, as in a JSON file."""
    fields = [f for f in silver.READING_FIELDS if any(f in r for r in rows)]
    schema = ", ".join(f"{f} {silver.READING_FIELDS[f]}" for f in fields)
    data = [tuple(r.get(f) for f in fields) for r in rows]
    return silver.typed_readings(spark.createDataFrame(data, schema))


def test_absent_fields_are_null_and_types_are_fixed(spark):
    typed = _readings(spark, [{"device_id": "D1", "ts": T0, "heart_rate": 88.0, "motion": 0.1}])
    row = typed.first().asDict()
    assert row["heart_rate"] == 88.0 and row["etco2"] is None and row["spo2"] is None
    assert dict(typed.dtypes)["battery_pct"] == "double"
    assert typed.columns[: len(silver.READING_FIELDS)] == list(silver.READING_FIELDS)


def test_quarantine_names_every_rule_a_reading_fails(spark):
    rows = [
        {"device_id": "D1", "ts": T0, "heart_rate": 88.0, "spo2": 97.0},  # valid
        {"device_id": "D1", "ts": T0, "heart_rate": 400.0, "spo2": 101.0},
        {"device_id": None, "ts": T0, "heart_rate": None, "spo2": None},
    ]
    bad = silver.quarantine(_readings(spark, rows), silver.READING_RULES).collect()
    assert sorted(sorted(r.failed_rules) for r in bad) == [
        ["device_id_present"],
        ["heart_rate_in_range", "spo2_in_range"],
    ]


def test_rules_are_never_null_so_expectations_and_quarantine_agree(spark):
    empty = _readings(spark, [{"device_id": None, "ts": None}])
    for name, condition in silver.READING_RULES.items():
        assert empty.selectExpr(f"({condition}) AS ok").first().ok is not None, name


def test_one_reading_per_device_and_time(spark):
    rows = [
        {"device_id": "D1", "ts": T0, "heart_rate": 88.0},
        {"device_id": "D1", "ts": T0, "heart_rate": 88.0},  # resent by the gateway
        {"device_id": "D2", "ts": T0, "heart_rate": 70.0},
    ]
    assert silver.device_readings(_readings(spark, rows)).count() == 2


def test_simulated_readings_all_pass_the_rules(spark, ward, landed):
    typed = silver.typed_readings(_bronze(spark, landed / "vitals", silver.READING_FIELDS))
    assert typed.count() == len(ward.readings)
    assert silver.quarantine(typed, silver.READING_RULES).count() == 0


def test_linking_matches_the_pandas_reference(spark, ward, landed):
    """Same encounter for every reading as features.link_readings, and the same drops."""
    typed = silver.typed_readings(_bronze(spark, landed / "vitals", silver.READING_FIELDS))
    sql = activity.sql_snapshot(ward.activity, ward.activity.end)["device_assignments"]
    assignments = spark.createDataFrame(
        sql[["device_id", "encounter_id", "patient_id", "start_ts", "end_ts"]]
    )
    got = (
        silver.link_readings(silver.device_readings(typed), assignments)
        .select("device_id", "ts", "encounter_id", "patient_id")
        .toPandas()
    )
    want = features.link_readings(ward.readings, sql)[
        ["device_id", "ts", "encounter_id", "patient_id"]
    ]
    key = ["device_id", "ts"]
    got = got.assign(ts=pd.to_datetime(got["ts"], utc=True)).sort_values(key, ignore_index=True)
    want = want.sort_values(key, ignore_index=True)
    assert len(got) == len(want) < len(ward.readings)  # readings at a stay's end_ts drop
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


def test_reconciliation_accounts_for_every_reading(spark):
    rows = [
        {"device_id": "D1", "ts": T0, "heart_rate": 88.0},
        {"device_id": "D1", "ts": T0, "heart_rate": 88.0},  # duplicate
        {"device_id": "D1", "ts": T0 + dt.timedelta(minutes=5), "heart_rate": 400.0},  # invalid
        {"device_id": "D2", "ts": T0, "heart_rate": 70.0},
        {"device_id": "D3", "ts": T0 - dt.timedelta(days=2), "heart_rate": 75.0},  # late
    ]
    typed = _readings(spark, rows)
    clean = silver.device_readings(typed.where(" AND ".join(silver.READING_RULES.values())))
    in_silver = clean.where("device_id <> 'D3'")  # what the watermark would have dropped
    linked = in_silver.where("device_id = 'D1'")  # D2 isn't on any patient
    got = silver.reconcile_readings(typed, in_silver, linked).first().asDict()
    assert got == {
        "bronze_rows": 5,
        "quarantined": 1,
        "duplicates": 1,
        "in_silver": 2,
        "missing_from_silver": 1,
        "missing_first_ts": T0 - dt.timedelta(days=2),
        "missing_last_ts": T0 - dt.timedelta(days=2),
        "linked": 1,
        "not_on_a_patient": 1,
    }


def test_outcomes_keep_the_first_copy_and_the_stays_site(spark, ward, landed):
    typed = silver.typed_outcomes(_bronze(spark, landed / "outcomes", silver.OUTCOME_FIELDS))
    twice = typed.unionByName(typed)  # the same file landed twice
    encounters = spark.createDataFrame(ward.activity.encounters[["encounter_id", "site_id"]])
    out = silver.outcomes(twice, encounters).toPandas()
    recorded = landing.recorded_outcomes(ward.truth.outcomes, ward.activity.end)
    assert len(out) == len(recorded) > 0
    assert out["site_id"].notna().all()
    assert silver.quarantine(typed, silver.OUTCOME_RULES).count() == 0


def test_every_source_table_has_a_history_type():
    assert set(silver.SCD_TYPE) == set(TABLES)
    assert set(silver.SCD_TYPE.values()) <= {1, 2}
