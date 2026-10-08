import numpy as np
import pandas as pd
import pytest

from generator import truth
from generator.truth import TruthConfig, hourly_onset_probability, simulate_ward

START = pd.Timestamp("2026-11-01", tz="UTC")
FLAGS = (
    "copd",
    "heart_failure",
    "diabetes",
    "ckd",
    "hypertension",
    "atrial_fibrillation",
    "on_beta_blocker",
)


def _inpatients(n: int = 120, age: int = 75) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "encounter_id": [f"E-{i}" for i in range(n)],
            "device_id": [f"DEV-A-{i:04d}" for i in range(n)],
            "age": age,
            "n_conditions": 1,
        }
    )
    for f in FLAGS:
        df[f] = False
    return df


@pytest.fixture(scope="module")
def ward():
    # A high base rate gives plenty of trajectories to test against.
    return simulate_ward(
        _inpatients(), START, hours=48, seed=3, truth_config=TruthConfig(base_rate_scale=8)
    )


def test_same_seed_same_hospital():
    a = simulate_ward(_inpatients(30), START, hours=12, seed=5)
    b = simulate_ward(_inpatients(30), START, hours=12, seed=5)
    pd.testing.assert_frame_equal(a.readings, b.readings)
    pd.testing.assert_frame_equal(a.truth.trajectories, b.truth.trajectories)


def test_outcomes_are_exactly_the_escalated_trajectories(ward):
    escalated = ward.truth.trajectories.query("outcome == 'escalated'")
    assert len(escalated) > 10
    assert sorted(ward.truth.outcomes["patient_id"]) == sorted(escalated["patient_id"])
    assert set(ward.truth.outcomes["event_type"]) <= {"rapid_response", "icu_transfer"}


def test_monitor_stops_when_patient_leaves_ward(ward):
    for _, event in ward.truth.outcomes.iterrows():
        device = f"DEV-A-{int(event['patient_id'][1:]):04d}"
        last = ward.readings.loc[ward.readings["device_id"] == device, "ts"].max()
        assert last <= event["event_ts"]


def test_at_most_one_escalation_per_patient(ward):
    assert ward.truth.outcomes["patient_id"].is_unique


def test_vitals_worsen_before_escalation(ward):
    resp = ward.truth.trajectories.query("outcome == 'escalated' and kind == 'respiratory'")
    row = resp.iloc[0]
    i = int(row["patient_id"][1:])
    step = ward.timestamps.get_loc(row["event_ts"])
    start = ward.timestamps.get_loc(row["start_ts"])
    spo2 = ward.physiology["spo2"][i]
    rr = ward.physiology["resp_rate"][i]
    assert spo2[step] < spo2[start] - 3
    assert rr[step] > rr[start] + 3


def test_patients_without_trajectories_are_untouched(ward):
    touched = set(ward.truth.trajectories["patient_id"])
    calm = [i for i in range(120) if f"p{i}" not in touched]
    for offsets in ward.truth.offsets.values():
        assert not offsets[calm].any()


def test_hidden_rule_rises_with_abnormal_vitals_and_age():
    patients = _inpatients(2)
    patients.loc[1, "age"] = 90
    normal = {"heart_rate": 75, "resp_rate": 16, "spo2": 97, "temp_c": 36.8, "sbp": 125}
    hourly = {v: np.full((2, 1), x, dtype=float) for v, x in normal.items()}
    base = hourly_onset_probability(patients, hourly, np.zeros(2), TruthConfig())
    assert base[1, 0] > base[0, 0]  # older patient

    hourly["spo2"][:] = 85
    hypoxic = hourly_onset_probability(patients, hourly, np.zeros(2), TruthConfig())
    assert hypoxic[0, 0] > 5 * base[0, 0]


def test_earlier_escalation_protocol_means_more_and_earlier_events():
    patients = _inpatients()
    counts = {}
    for threshold in (1.0, 0.7):
        cfg = TruthConfig(base_rate_scale=8, escalation_threshold=threshold)
        traj = [
            simulate_ward(patients, START, hours=48, seed=s, truth_config=cfg).truth.trajectories
            for s in range(3)
        ]
        counts[threshold] = sum((t["outcome"] == "escalated").sum() for t in traj)
        recovered = sum((t["outcome"] == "recovered").sum() for t in traj)
        counts[f"recovered_{threshold}"] = recovered
    assert counts[0.7] > counts[1.0]
    assert counts["recovered_0.7"] < counts["recovered_1.0"]


def test_decompensation_profiles_point_the_right_way():
    assert truth.DECOMPENSATION["respiratory"]["spo2"] < 0
    assert truth.DECOMPENSATION["sepsis"]["temp_c"] > 0
    assert truth.DECOMPENSATION["cardiac"]["sbp"] < 0
