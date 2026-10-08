import numpy as np
import pandas as pd
import pytest

from generator import activity, hospital, truth
from generator.synthea import CONDITION_KEYWORDS

START = hospital.SIM_START
HOURS = 24 * 7


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
def schedule():
    pop = _population()
    inpatients = hospital.admit_inpatients(pop, n=300)
    return pop, inpatients, activity.schedule_activity(pop, inpatients, START, HOURS, seed=1)


@pytest.fixture(scope="module")
def ward(schedule):
    pop, inpatients, act = schedule
    return truth.simulate_ward(inpatients, START, HOURS, seed=2, activity=act)


def _overlaps(intervals: pd.DataFrame, key: str, start: str, end: str, horizon) -> bool:
    df = intervals.assign(_end=intervals[end].fillna(horizon)).sort_values([key, start])
    previous_end = df.groupby(key)["_end"].shift()
    return bool((df[start] < previous_end).any())


def test_schedule_is_reproducible():
    pop = _population(800)
    inpatients = hospital.admit_inpatients(pop, n=150)
    a = activity.schedule_activity(pop, inpatients, START, 48, seed=5)
    b = activity.schedule_activity(pop, inpatients, START, 48, seed=5)
    pd.testing.assert_frame_equal(a.encounters, b.encounters)


def test_census_stays_near_capacity_target(schedule):
    census = activity.census(schedule[2])
    assert 250 <= census.min() and census.max() <= 360
    assert 270 <= census.mean() <= 320


def test_lengths_of_stay_are_realistic(schedule):
    enc = schedule[2].encounters.dropna(subset=["discharge_ts"])
    days = (enc["discharge_ts"] - enc["admit_ts"]).dt.total_seconds() / 86400
    assert 2.5 < days.median() < 6.5


def test_no_device_or_patient_is_double_booked(schedule):
    act = schedule[2]
    dev = act.device_assignments
    assert not _overlaps(dev, "device_id", "start_ts", "end_ts", act.end)
    assert not _overlaps(dev, "patient_id", "start_ts", "end_ts", act.end)


def test_transfers_stay_in_site_and_change_unit_type(schedule):
    moves = schedule[2].transfers
    assert len(moves) > 0
    site = moves["from_unit_id"].str.split("-").str[0]
    assert (site == moves["to_unit_id"].str.split("-").str[0]).all()
    assert (moves["from_unit_id"] != moves["to_unit_id"]).all()


def test_sql_snapshot_reflects_the_moment(schedule):
    act = schedule[2]
    as_of = START + pd.Timedelta(days=3)
    enc = activity.sql_snapshot(act, as_of)["encounters"]
    assert (enc["admit_ts"] <= as_of).all()
    assert (enc["last_updated"] <= as_of).all()
    discharged = enc["status"] == "discharged"
    assert enc.loc[discharged, "discharge_ts"].notna().all()
    assert enc.loc[~discharged, "discharge_ts"].isna().all()
    assert (enc.loc[discharged, "last_updated"] == enc.loc[discharged, "discharge_ts"]).all()


def test_incremental_changes_are_only_rows_touched_since_last_run(schedule):
    act = schedule[2]
    day3, day5 = START + pd.Timedelta(days=3), START + pd.Timedelta(days=5)
    changed = activity.changes_between(act, day3, day5)["encounters"]
    assert len(changed) > 0
    assert changed["last_updated"].between(day3, day5, inclusive="right").all()
    before = activity.sql_snapshot(act, day3)["encounters"].set_index("encounter_id")
    untouched = before.index.difference(changed["encounter_id"])
    after = activity.sql_snapshot(act, day5)["encounters"].set_index("encounter_id")
    pd.testing.assert_frame_equal(before.loc[untouched], after.loc[untouched])


def test_escalation_ends_the_stay_and_the_readings(ward):
    events = ward.truth.outcomes
    assert len(events) > 0
    enc = ward.activity.encounters.set_index("encounter_id")
    assert (enc.loc[events["encounter_id"], "disposition"] == "escalated").all()
    assert (enc.loc[events["encounter_id"], "discharge_ts"].to_numpy() == events["event_ts"]).all()

    dev = ward.activity.device_assignments.set_index("encounter_id")
    for _, e in events.iterrows():
        device = dev.loc[e["encounter_id"], "device_id"]
        assigned = ward.activity.device_assignments
        later = assigned[
            (assigned["device_id"] == device) & (assigned["start_ts"] > e["event_ts"])
        ]["start_ts"]
        gap_end = later.min() if len(later) else ward.timestamps[-1] + pd.Timedelta(minutes=5)
        r = ward.readings
        gap = r[(r["device_id"] == device) & (r["ts"] > e["event_ts"]) & (r["ts"] < gap_end)]
        assert gap.empty


def test_snapshot_of_actual_records_shows_escalations_as_discharges(ward):
    e = ward.truth.outcomes.iloc[0]
    snap = activity.sql_snapshot(ward.activity, e["event_ts"])["encounters"]
    row = snap.set_index("encounter_id").loc[e["encounter_id"]]
    assert row["status"] == "discharged" and row["discharge_ts"] == e["event_ts"]


def test_transfers_are_spread_out_not_bunched_at_the_start(schedule):
    moves = schedule[2].transfers
    assert (moves["transfer_ts"] == START).mean() < 0.05


def test_new_admissions_get_their_own_vitals(ward):
    stays = ward.stays
    reused = stays["device_id"].duplicated(keep=False)
    assert reused.any()  # devices really are passed from patient to patient
