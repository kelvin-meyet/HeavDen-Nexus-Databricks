import json

import pandas as pd
import pytest
from test_source_db import REFERENCE, _population, _synthea

from generator import drift, episode, hospital, landing, source_db, vitals

EPISODES = [
    episode.Episode("sensor-bias", 24, ["spo2_sensor_bias"], seed=11),
    episode.Episode("normal", 24, [], seed=12),
]


@pytest.fixture(scope="module")
def periods():
    pop = _population()
    inpatients = hospital.admit_inpatients(pop, n=120)
    periods = [episode.baseline(pop, inpatients, days=2)]
    for e in EPISODES:
        periods.append(episode.continue_hospital(pop, periods[-1], e))
    return periods


def _tables(period):
    ids = list(period.actual.encounters.sort_values("admit_ts")["patient_id"].unique())
    return episode.source_tables(_synthea(ids), REFERENCE, period)


def _overlapping(df: pd.DataFrame, key: str, start: str, end: str, horizon) -> int:
    df = df.assign(_end=df[end].fillna(horizon)).sort_values([key, start])
    return int((df.groupby(key)[start].shift(-1) < df["_end"]).sum())


def test_an_episode_continues_the_stays_in_bed(periods):
    before, after = periods[0], periods[1]
    old = before.actual.encounters
    in_bed = old[old["discharge_ts"].isna()].set_index("encounter_id")
    new = after.actual.encounters.set_index("encounter_id")
    assert len(in_bed) > 100
    for column in ("patient_id", "admit_ts", "admit_unit_id", "admit_bed_id", "device_id"):
        pd.testing.assert_series_equal(new.loc[in_bed.index, column], in_bed[column])
    numbers = new.index.str[2:].astype(int)
    admitted = new["admit_ts"] >= after.start
    assert admitted.any()
    assert numbers[admitted].min() > old["encounter_id"].str[2:].astype(int).max()
    assert after.start == before.end and after.end == before.end + pd.Timedelta(hours=24)


def test_history_never_double_books_a_device_or_a_patient(periods):
    final = periods[-1]
    devices = final.actual.device_assignments
    stays = final.actual.encounters
    assert stays["encounter_id"].is_unique
    assert _overlapping(devices, "device_id", "start_ts", "end_ts", final.end) == 0
    assert _overlapping(stays, "patient_id", "admit_ts", "discharge_ts", final.end) == 0


def test_every_new_reading_belongs_to_one_stay(periods):
    for period in periods[1:]:
        readings = period.readings[["device_id", "ts"]]
        assert readings["ts"].min() >= period.start and readings["ts"].max() < period.end
        devices = period.actual.device_assignments
        linked = readings.merge(devices, on="device_id")
        linked = linked[
            (linked["start_ts"] <= linked["ts"])
            & (linked["end_ts"].isna() | (linked["ts"] < linked["end_ts"]))
        ]
        per_reading = linked.groupby(["device_id", "ts"]).size()
        assert (per_reading == 1).all()
        assert len(per_reading) >= 0.999 * len(readings)  # rest: exactly at a stay's end


def test_stays_keep_their_personal_normal_values(periods):
    before, after = periods[0], periods[1]
    old = before.actual.encounters
    carried = old.loc[old["discharge_ts"].isna(), "encounter_id"]
    pd.testing.assert_frame_equal(after.baselines.loc[carried], before.baselines.loc[carried])
    assert list(after.baselines.columns) == [*vitals.VITALS, "hr_wander_scale"]


def test_the_source_database_only_gets_inserts_and_updates(periods):
    for before, after in zip(periods, periods[1:], strict=False):
        changes = source_db.table_changes(_tables(before), _tables(after))
        updated = {table for table, (_, updates) in changes.items() if len(updates)}
        assert updated <= {"encounters", "device_assignments"}
        assert len(changes["encounters"][0]) and len(changes["nurse_observations"][0])
        old_ids = set(before.actual.encounters["encounter_id"])
        assert set(changes["encounters"][1]["encounter_id"]) <= old_ids


def test_landing_files_are_new_and_include_outcomes_recorded_late(periods):
    base, first = periods[0], periods[1]
    old = landing.landing_files("dev", base.readings, base.outcomes, base.end)
    new = episode.new_files("dev", base, first)
    assert not set(old) & set(new)
    outcomes = pd.DataFrame(
        [
            json.loads(line)
            for path, text in new.items()
            if "/outcomes/" in path
            for line in text.splitlines()
        ]
    )
    recorded = pd.to_datetime(outcomes["recorded_ts"], utc=True)
    assert recorded.gt(base.end).all() and recorded.le(first.end).all()
    # escalations near the end of the baseline are recorded during the episode
    late = base.outcomes[base.outcomes["event_ts"] + landing.LABEL_DELAY > base.end]
    assert len(late) and set(late["encounter_id"]) <= set(outcomes["encounter_id"])


def test_scenarios_start_inside_their_episode(periods):
    first = periods[1]
    starts = first.windows.set_index("scenario")["starts_at"]
    hours = drift.load_config()["spo2_sensor_bias"]["start_after_hours"]
    assert starts["spo2_sensor_bias"] == first.start + pd.Timedelta(hours=hours)
    assert periods[2].windows.empty


def test_settings_override_the_scenario_config():
    settings = {"protocol_change": {"start_after_hours": 0}}
    e = episode.Episode("x", 48, ["protocol_change"], 1, settings)
    assert episode.scenario_config(e)["protocol_change"]["start_after_hours"] == 0
    assert drift.load_config()["protocol_change"]["start_after_hours"] == 12  # file unchanged


@pytest.mark.parametrize("env", source_db.ENVIRONMENTS)
def test_the_manifest_is_valid(env):
    episodes = episode.load_episodes(env)
    config = drift.load_config()
    assert len({e.name for e in episodes}) == len(episodes)
    assert len({e.seed for e in episodes}) == len(episodes)
    for e in episodes:
        assert set(e.scenarios) <= set(config) and set(e.settings) <= set(e.scenarios)
        assert e.hours > 0
