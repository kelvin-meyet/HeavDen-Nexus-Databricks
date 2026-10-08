import numpy as np
import pandas as pd
import pytest

from generator import drift, hospital
from generator.synthea import CONDITION_KEYWORDS

START = hospital.SIM_START


def _population(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "age": rng.integers(18, 95, size=n),
            "sex": rng.choice(["F", "M"], size=n),
        }
    )
    for flag in [*CONDITION_KEYWORDS, "on_beta_blocker"]:
        df[flag] = rng.random(n) < 0.1
    df["n_conditions"] = df[list(CONDITION_KEYWORDS)].sum(axis=1)
    return df


@pytest.fixture(scope="module")
def hospital_setup():
    pop = _population()
    return pop, hospital.admit_inpatients(pop, n=300)


def _episode(hospital_setup, *scenarios, hours=48):
    pop, inpatients = hospital_setup
    return drift.simulate_episode(pop, inpatients, START, hours, scenarios, seed=7)


@pytest.fixture(scope="module")
def normal(hospital_setup):
    return _episode(hospital_setup)


def test_config_has_the_four_planned_scenarios():
    assert set(drift.load_config()) == {
        "spo2_sensor_bias",
        "respiratory_outbreak",
        "protocol_change",
        "firmware_schema_change",
    }


def test_unknown_scenario_is_rejected(hospital_setup):
    with pytest.raises(ValueError, match="unknown"):
        _episode(hospital_setup, "alien_invasion", hours=6)


def test_sensor_bias_lowers_site_b_readings_but_not_true_health(hospital_setup, normal):
    biased = _episode(hospital_setup, "spo2_sensor_bias")
    late = normal.readings["ts"] >= START + pd.Timedelta(hours=36)

    def site_b_mean(ep):
        r = ep.readings
        mask = r["device_id"].str.startswith("DEV-B") & (r["ts"] >= START + pd.Timedelta(hours=36))
        return r.loc[mask, "spo2"].mean()

    assert site_b_mean(biased) < site_b_mean(normal) - 0.3
    assert np.allclose(
        np.nan_to_num(biased.ward.physiology["spo2"]), np.nan_to_num(normal.ward.physiology["spo2"])
    )
    other = normal.readings.loc[late & ~normal.readings["device_id"].str.startswith("DEV-B")]
    other_b = biased.readings.loc[
        (biased.readings["ts"] >= START + pd.Timedelta(hours=36))
        & ~biased.readings["device_id"].str.startswith("DEV-B")
    ]
    assert other["spo2"].mean() == pytest.approx(other_b["spo2"].mean(), abs=0.05)


def test_outbreak_brings_pneumonia_admissions_only_after_it_starts(hospital_setup):
    ep = _episode(hospital_setup, "respiratory_outbreak")
    enc = ep.plan.encounters
    starts = ep.windows.set_index("scenario").loc["respiratory_outbreak", "starts_at"]
    pneumonia = enc["admission_reason"] == "pneumonia"
    assert pneumonia.any()
    assert (enc.loc[pneumonia, "admit_ts"] >= starts).all()
    stays = ep.ward.stays
    assert (
        stays.loc[stays["pneumonia"], "resp_rate"].mean()
        > stays.loc[~stays["pneumonia"], "resp_rate"].mean() + 2
    )


def test_protocol_change_gives_more_escalations(hospital_setup):
    pop, inpatients = hospital_setup
    totals = {"normal": 0, "protocol": 0}
    for seed in range(3):
        for name, scen in {"normal": (), "protocol": ("protocol_change",)}.items():
            ep = drift.simulate_episode(pop, inpatients, START, 48, scen, seed=seed)
            totals[name] += len(ep.ward.truth.outcomes)
    assert totals["protocol"] > totals["normal"]


def test_protocol_reviews_only_happen_after_the_change(hospital_setup):
    ep = _episode(hospital_setup, "protocol_change")
    starts = ep.windows["starts_at"].iloc[0]
    reviews = ep.ward.truth.trajectories.query("kind == 'protocol_review'")
    assert len(reviews) > 0
    assert (reviews["event_ts"] >= starts).all()
    assert set(reviews["encounter_id"]) <= set(ep.ward.truth.outcomes["encounter_id"])


def test_firmware_change_swaps_fields_from_its_start(hospital_setup):
    ep = _episode(hospital_setup, "firmware_schema_change", hours=12)
    starts = ep.windows["starts_at"].iloc[0]
    r = ep.readings
    before, after = r[r["ts"] < starts], r[r["ts"] >= starts]
    assert before["motion"].notna().all() and before["etco2"].isna().all()
    assert after["motion"].isna().all() and after["etco2"].notna().all()
    assert set(after["firmware"]) == {"3.2.0"}
    assert after["etco2"].between(15, 60).all()


def test_escalations_before_a_protocol_change_are_unchanged(hospital_setup, normal):
    ep = _episode(hospital_setup, "protocol_change")
    starts = ep.windows["starts_at"].iloc[0]
    cols = ["encounter_id", "event_ts", "event_type"]
    n_out, p_out = normal.ward.truth.outcomes, ep.ward.truth.outcomes
    before = n_out[n_out["event_ts"] < starts][cols].reset_index(drop=True)
    after = p_out[p_out["event_ts"] < starts][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_scenarios_do_not_change_the_hospital_before_they_start(hospital_setup, normal):
    ep = _episode(hospital_setup, "protocol_change", "spo2_sensor_bias")
    early = START + pd.Timedelta(hours=12)
    a = normal.readings[normal.readings["ts"] < early].reset_index(drop=True)
    b = ep.readings[ep.readings["ts"] < early].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
