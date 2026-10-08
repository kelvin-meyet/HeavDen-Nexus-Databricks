import numpy as np
import pandas as pd
import pytest

from generator.vitals import (
    LIMITS,
    VITALS,
    VitalsConfig,
    make_baselines,
    simulate_physiology,
    simulate_vitals,
)

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
CLEAN = VitalsConfig(wander=False, circadian=False, noise=False, artefacts=False, faults=False)


def _inpatients(n: int = 400, **flags: bool) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "device_id": [f"DEV-A-{i:04d}" for i in range(n)],
            "age": 60,
        }
    )
    for f in FLAGS:
        df[f] = flags.get(f, False)
    return df


def test_same_seed_same_readings():
    a = simulate_vitals(_inpatients(20), START, hours=6, seed=1).readings
    b = simulate_vitals(_inpatients(20), START, hours=6, seed=1).readings
    pd.testing.assert_frame_equal(a, b)


def test_copd_lowers_spo2_and_beta_blockers_lower_heart_rate():
    rng = np.random.default_rng(0)
    healthy = make_baselines(_inpatients(), rng)
    copd = make_baselines(_inpatients(copd=True), rng)
    blocked = make_baselines(_inpatients(on_beta_blocker=True), rng)
    assert copd["spo2"].mean() < healthy["spo2"].mean() - 4
    assert blocked["heart_rate"].mean() < healthy["heart_rate"].mean() - 7


def test_fever_raises_heart_and_breathing_rate():
    patients = _inpatients(50)
    baselines = make_baselines(patients, np.random.default_rng(0))
    timestamps = pd.date_range(START, periods=12, freq="5min")
    fever = {"temp_c": np.full((50, 12), 1.5)}
    calm = simulate_physiology(baselines, timestamps, np.random.default_rng(1), CLEAN)
    hot = simulate_physiology(baselines, timestamps, np.random.default_rng(1), CLEAN, fever)
    assert np.allclose(hot["heart_rate"] - calm["heart_rate"], 15)
    assert np.allclose(hot["resp_rate"] - calm["resp_rate"], 3)


def test_clean_config_sends_every_reading_exactly():
    result = simulate_vitals(_inpatients(10), START, hours=2, config=CLEAN)
    assert len(result.readings) == 10 * 24
    first = result.readings[result.readings["device_id"] == "DEV-A-0000"]["heart_rate"]
    assert (first == round(result.baselines.loc[0, "heart_rate"])).all()


def test_faults_drop_some_messages_but_not_most():
    result = simulate_vitals(_inpatients(100), START, hours=24)
    expected = 100 * 288
    assert 0.9 * expected < len(result.readings) < expected


def test_readings_stay_in_physical_limits():
    readings = simulate_vitals(_inpatients(100), START, hours=24).readings
    for v in VITALS:
        lo, hi = LIMITS[v]
        assert readings[v].between(lo, hi).all(), v
    assert readings["spo2"].max() <= 100


def test_measurement_offset_changes_readings_not_physiology():
    patients = _inpatients(30)
    bias = {"spo2": np.full((30, 72), -3.0)}
    plain = simulate_vitals(patients, START, hours=6, config=CLEAN)
    biased = simulate_vitals(patients, START, hours=6, config=CLEAN, measurement_offsets=bias)
    assert np.array_equal(plain.physiology["spo2"], biased.physiology["spo2"])
    assert biased.readings["spo2"].mean() == pytest.approx(plain.readings["spo2"].mean() - 3, 0.01)


def test_readings_carry_device_not_patient():
    readings = simulate_vitals(_inpatients(5), START, hours=1).readings
    assert "device_id" in readings and "patient_id" not in readings
