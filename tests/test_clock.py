import datetime as dt

import pandas as pd
import pytest

from generator import clock, hospital, synthea

BY = pd.Timedelta(days=-42)


def test_every_date_and_time_moves_and_nothing_else():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "birth_date": [dt.date(1950, 3, 1), dt.date(2000, 2, 29)],
            "stop_date": [dt.date(2026, 11, 1), None],
            "admit_ts": pd.to_datetime(["2026-11-01 08:30", "2026-11-02 23:59"], utc=True),
            "discharge_ts": pd.to_datetime(["2026-11-03 10:00", None], utc=True),
            "beds": [3, 4],
        }
    )
    moved = clock.move(df, BY)
    assert moved["birth_date"].tolist() == [dt.date(1950, 1, 18), dt.date(2000, 1, 18)]
    assert moved["stop_date"].tolist() == [dt.date(2026, 9, 20), None]
    assert moved["admit_ts"].tolist() == list(df["admit_ts"] + BY)
    assert moved["admit_ts"].dt.hour.tolist() == [8, 23]  # the hour of day is the hospital's
    assert pd.isna(moved["discharge_ts"].iloc[1])
    assert moved[["patient_id", "beds"]].equals(df[["patient_id", "beds"]])
    assert df["admit_ts"].iloc[0] == pd.Timestamp("2026-11-01 08:30", tz="UTC")  # a copy
    pd.testing.assert_frame_equal(clock.move(moved, -BY), df)
    assert clock.move(df, pd.Timedelta(0)) is df


def test_patient_histories_end_on_the_hospitals_first_day():
    """One calendar: the histories' "today" is the day the hospital starts."""
    run = synthea.SyntheaRun()
    assert pd.Timestamp(run.reference_date, tz="UTC") == hospital.SIM_START
    moved_by = hospital.SIM_START - pd.Timestamp(run.generated_for, tz="UTC")
    assert moved_by == moved_by.floor("D")  # whole days: ages and the hour of day are kept


def test_loaded_histories_are_moved_onto_the_calendar(tmp_path):
    for name in synthea.TABLES:
        (tmp_path / f"{name}.csv").write_text(
            {
                "patients": "Id,BIRTHDATE,DEATHDATE\np1,1950-03-01,\n",
                "conditions": "PATIENT,START,STOP\np1,2026-10-31,\n",
                "medications": "PATIENT,START,STOP\np1,2026-01-01T08:00:00Z,2026-11-01T08:00:00Z\n",
                "encounters": "PATIENT,START,STOP\np1,2026-10-01T00:00:00Z,2026-10-02T00:00:00Z\n",
            }[name],
            encoding="utf-8",
        )
    as_generated = synthea.SyntheaRun(reference_date="20261101")  # no move
    plain = synthea.load_tables(tmp_path, as_generated)
    moved = synthea.load_tables(tmp_path)
    assert plain["patients"]["BIRTHDATE"].iloc[0] == pd.Timestamp("1950-03-01", tz="UTC")
    assert moved["patients"]["BIRTHDATE"].iloc[0] == pd.Timestamp("1950-01-18", tz="UTC")
    # the newest condition was a day before the histories' "today", and still is
    newest = moved["conditions"]["START"].iloc[0]
    assert hospital.SIM_START - newest == pd.Timedelta(days=1)
    assert moved["medications"]["STOP"].iloc[0] == hospital.SIM_START + pd.Timedelta(hours=8)
    assert pd.isna(moved["patients"]["DEATHDATE"].iloc[0])


def test_a_load_that_ends_in_the_future_is_refused():
    now = pd.Timestamp("2026-10-14 09:00", tz="UTC")
    clock.require_past(pd.Timestamp("2026-10-14 00:00", tz="UTC"), now)
    with pytest.raises(SystemExit, match="still in the future"):
        clock.require_past(pd.Timestamp("2026-10-16 00:00", tz="UTC"), now)
