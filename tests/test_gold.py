"""Gold on local Spark, checked against the pandas references.

The same simulated ward goes through `features.build_patient_hours` and
`gold.device_health_daily` (pandas) and through the Spark Gold functions; the results must match
row for row, except where Gold differs by design (labels from recorded outcomes, age at
admission; see `heavden.pipelines.gold`).
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from test_source_db import HOURS, START, _population

from generator import activity, hospital, landing, truth
from heavden.analytics import gold as gold_pandas
from heavden.ml import features

pytest.importorskip("pyspark")
from heavden.pipelines import gold  # noqa: E402  (needs pyspark)

KEYS = ["encounter_id", "prediction_ts"]
# One active source description per flag, as Synthea words them.
DESCRIPTIONS = {
    "copd": "Chronic obstructive bronchitis (disorder)",
    "heart_failure": "Chronic congestive heart failure (disorder)",
    "diabetes": "Diabetes mellitus type 2 (disorder)",
    "ckd": "Chronic kidney disease stage 1 (disorder)",
    "hypertension": "Essential hypertension (disorder)",
    "atrial_fibrillation": "Atrial fibrillation (disorder)",
}


@pytest.fixture(scope="module")
def ward():
    pop = _population()
    inpatients = hospital.admit_inpatients(pop, n=120)
    plan = activity.schedule_activity(pop, inpatients, START, HOURS, seed=1)
    return truth.simulate_ward(inpatients, START, HOURS, seed=2, activity=plan)


def _source_history(patients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conditions and medications that give each patient exactly their profile flags, plus
    decoys that must not count: prediabetes, a cured condition, a future diagnosis."""
    conditions, meds = [], []
    for p in patients.itertuples():
        for flag, text in DESCRIPTIONS.items():
            if getattr(p, flag):
                conditions.append((p.patient_id, flag, text, dt.date(2015, 1, 1), None))
        if not p.diabetes:
            conditions.append(
                (p.patient_id, "pre", "Prediabetes (finding)", dt.date(2016, 1, 1), None)
            )
        if not p.heart_failure:
            cured = ("Heart failure (disorder)", dt.date(2010, 1, 1), dt.date(2011, 1, 1))
            future = ("Atrial fibrillation (disorder)", dt.date(2027, 1, 1), None)
            conditions.append((p.patient_id, "cured", *cured))
            if not p.atrial_fibrillation:
                conditions.append((p.patient_id, "future", *future))
        if p.on_beta_blocker:
            drug = "24 HR Metoprolol succinate 100 MG Extended Release Oral Tablet"
            meds.append(
                (p.patient_id, "866427", drug, pd.Timestamp("2020-01-01", tz="UTC"), pd.NaT)
            )
    conditions = pd.DataFrame(
        conditions, columns=["patient_id", "code", "description", "start_date", "stop_date"]
    )
    meds = pd.DataFrame(meds, columns=["patient_id", "code", "description", "start_ts", "stop_ts"])
    return conditions, meds


def _spark(spark, df: pd.DataFrame):
    """pandas -> Spark through Arrow. Without Arrow, Spark converts times through the local
    time zone (DST shifts them), so Arrow-backed string columns become plain objects first."""
    strings = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    out = spark.createDataFrame(df.astype(dict.fromkeys(strings, object)))
    return out


@pytest.fixture(scope="module")
def silver(spark, ward):
    """Silver-shaped Spark tables for the ward."""
    act = ward.activity
    linked = features.link_readings(ward.readings, act.device_assignments)
    enc = act.encounters[["encounter_id", "patient_id", "site_id", "admit_ts", "discharge_ts"]]
    units = pd.concat(
        [
            act.encounters[["encounter_id", "admit_unit_id", "admit_ts"]].set_axis(
                ["encounter_id", "unit_id", "changed_ts"], axis=1
            ),
            act.transfers[["encounter_id", "to_unit_id", "transfer_ts"]].set_axis(
                ["encounter_id", "unit_id", "changed_ts"], axis=1
            ),
        ]
    )
    conditions, meds = _source_history(act.patients)
    outcomes = landing.recorded_outcomes(ward.truth.outcomes, act.end)
    obs = ward.nurse_observations[["encounter_id", "obs_ts", "acvpu", "on_oxygen", "o2_flow_lpm"]]
    frames = {
        "vitals": linked,
        "readings": ward.readings,
        "encounters": enc,
        "units": units,
        "patients": act.patients[["patient_id", "birth_date", "sex"]],
        "conditions": conditions,
        "medications": meds,
        "observations": obs,
        "outcomes": outcomes[["encounter_id", "event_type", "event_ts", "recorded_ts"]],
        "assignments": act.device_assignments,
    }
    return {name: _spark(spark, df) for name, df in frames.items()}


@pytest.fixture(scope="module")
def built(silver):
    s = silver
    flags = gold.profile_flags(s["encounters"], s["conditions"], s["medications"])
    feats = gold.patient_hour_features(
        s["vitals"], s["encounters"], s["units"], s["patients"], flags, s["observations"]
    ).cache()
    window = gold.data_window(s["vitals"])
    return {"features": feats, "window": window, "flags": flags}


@pytest.fixture(scope="module")
def reference(ward):
    return features.build_patient_hours(
        ward.readings,
        ward.activity,
        ward.truth.outcomes,
        nurse_observations=ward.nurse_observations,
    )


def _pandas(df) -> pd.DataFrame:
    out = df.toPandas()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = pd.to_datetime(out[column], utc=True)
    return out


def test_data_window_is_the_simulation_window(ward, built):
    w = _pandas(built["window"]).iloc[0]  # collect() would give local times; toPandas gives UTC
    assert w["data_start"] == ward.activity.start
    assert w["as_of"] == ward.activity.end


def test_flags_follow_the_inclusion_rule(ward, built):
    """Exactly the profile flags: prediabetes, cured and future conditions don't count."""
    got = _pandas(built["flags"]).set_index("encounter_id")
    enc = ward.activity.encounters.set_index("encounter_id")
    profile = ward.activity.patients.set_index("patient_id")
    for flag in [*gold.CONDITION_FLAGS, "on_beta_blocker", "n_conditions"]:
        want = enc["patient_id"].map(profile[flag]).astype(int)
        assert (got[flag].astype(int).reindex(want.index) == want).all(), flag


def test_features_match_the_pandas_reference(ward, built, reference):
    got = _pandas(built["features"]).sort_values(KEYS, ignore_index=True)
    want = reference.drop(columns=["label", "label_known_at"]).sort_values(KEYS, ignore_index=True)
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want) > 1000

    # Age is taken at admission from the birth date (pandas: a profile column).
    birth = pd.to_datetime(
        got["patient_id"].map(ward.activity.patients.set_index("patient_id")["birth_date"])
    )
    admit = got["encounter_id"].map(ward.activity.encounters.set_index("encounter_id")["admit_ts"])
    days = (admit.dt.tz_localize(None).dt.normalize() - birth).dt.days
    assert (got["age"] == np.floor(days / 365.25)).all()

    columns = [c for c in want.columns if c != "age"]
    pd.testing.assert_frame_equal(got[columns], want[columns], check_dtype=False)


def test_labels_use_only_recorded_outcomes(ward, built, silver, reference):
    """Where Gold has a label it equals the pandas label; it has one exactly when the
    outcome is recorded or t + 12 h has passed the data's end."""
    labelled = gold.training_set(built["features"], silver["outcomes"], built["window"])
    got = _pandas(labelled.select(*KEYS, "label", "label_known_at"))
    ref = reference[[*KEYS, "label"]]
    end = ward.activity.end
    recorded = landing.recorded_outcomes(ward.truth.outcomes, end).set_index("encounter_id")
    event = ref["encounter_id"].map(recorded["event_ts"])
    t = ref["prediction_ts"]
    positive = (event > t) & (event <= t + pd.Timedelta(hours=6))
    settled = t + pd.Timedelta(hours=12) <= end
    want = ref[settled | positive]

    merged = want.merge(got, on=KEYS, how="outer", indicator=True, suffixes=("_want", "_got"))
    assert (merged["_merge"] == "both").all()
    assert (merged["label_want"] == merged["label_got"]).all()
    assert merged["label_got"].sum() > 0  # the test ward has escalations to find
    assert (got["label_known_at"] <= end).all()


def test_unit_comes_from_history_before_the_hour(spark):
    t = pd.Timestamp("2026-11-01 10:00", tz="UTC")
    grid = spark.createDataFrame(
        pd.DataFrame(
            {"encounter_id": ["E1", "E1", "E2"], "prediction_ts": [t, t + pd.Timedelta(hours=2), t]}
        )
    )
    units = spark.createDataFrame(
        pd.DataFrame(
            {
                "encounter_id": ["E1", "E1", "E2"],
                "unit_id": ["A-GENERAL", "A-STEP_DOWN", "B-RESPIRATORY"],
                # E1 moved at 11:00; E2 has only a version recorded later (one bulk load)
                "changed_ts": [
                    t - pd.Timedelta(hours=5),
                    t + pd.Timedelta(hours=1),
                    t + pd.Timedelta(hours=9),
                ],
            }
        )
    )
    got = _pandas(gold.unit_at(grid, units)).sort_values(KEYS)
    assert list(got["unit_id"]) == ["A-GENERAL", "A-STEP_DOWN", "B-RESPIRATORY"]


def test_unit_history_reads_the_scd2_start(spark):
    scd2 = spark.sql(
        "SELECT 'E1' AS encounter_id, 'A-GENERAL' AS unit_id, "
        "named_struct('last_updated', TIMESTAMP'2026-11-01 08:00:00', '_change_version', 1L, "
        "'_ingest_ts', TIMESTAMP'2026-11-02 00:00:00') AS __START_AT"
    )
    row = _pandas(gold.unit_history(scd2)).iloc[0]
    assert row["changed_ts"] == pd.Timestamp("2026-11-01 08:00", tz="UTC")


def test_condition_reference_lists_what_each_flag_matched(silver):
    ref = _pandas(gold.condition_flag_reference(silver["conditions"], silver["medications"]))
    diabetes = ref[ref["flag"] == "diabetes"]["description"]
    assert "Prediabetes (finding)" not in set(diabetes)
    assert set(ref["flag"]) <= {*gold.CONDITION_FLAGS, "on_beta_blocker"}
    assert (ref["patients"] > 0).all()


def test_device_health_matches_the_pandas_reference(ward, silver, built):
    act = ward.activity
    want = gold_pandas.device_health_daily(
        ward.readings, act.device_assignments, act.start, act.end
    )
    got = _pandas(
        gold.device_health_daily(silver["readings"], silver["assignments"], built["window"])
    )
    key = ["device_id", "date"]
    got = got.sort_values(key, ignore_index=True)
    want = want.sort_values(key, ignore_index=True)
    assert list(got.columns) == list(want.columns)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
