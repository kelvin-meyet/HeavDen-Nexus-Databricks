import numpy as np
import pandas as pd
import pytest

from generator.activity import HospitalActivity
from heavden.ml import features, news2

T0 = pd.Timestamp("2026-11-01", tz="UTC")


def h(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


# --- NEWS2 -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "values", "expected"),
    [
        (news2.resp_rate_points, [8, 9, 11, 12, 20, 21, 24, 25], [3, 1, 1, 0, 0, 2, 2, 3]),
        (news2.sbp_points, [90, 91, 100, 101, 110, 111, 219, 220], [3, 2, 2, 1, 1, 0, 0, 3]),
        (
            news2.heart_rate_points,
            [40, 41, 50, 51, 90, 91, 110, 111, 130, 131],
            [3, 1, 1, 0, 0, 1, 1, 2, 2, 3],
        ),
        (
            news2.temp_points,
            [35.0, 35.1, 36.0, 36.1, 38.0, 38.1, 39.0, 39.1],
            [3, 1, 1, 0, 0, 1, 1, 2],
        ),
    ],
)
def test_news2_bands(fn, values, expected):
    assert fn(np.array(values, dtype=float)).tolist() == expected


def test_news2_spo2_uses_scale_2_for_copd():
    spo2 = np.array([91, 93, 95, 96, 87, 88], dtype=float)
    assert news2.spo2_points(spo2).tolist() == [3, 2, 1, 0, 3, 3]
    copd = np.array([True] * 6)
    assert news2.spo2_points(spo2, copd).tolist() == [0, 0, 0, 0, 1, 0]


def test_news2_oxygen_and_consciousness():
    assert news2.oxygen_points(np.array([True, False])).tolist() == [2, 0]
    assert news2.consciousness_points(
        np.array(["A", "C", "V", "P", "U", None], dtype=object)
    ).tolist() == [
        0,
        3,
        3,
        3,
        3,
        0,
    ]


def test_news2_scale_2_penalises_high_spo2_on_oxygen():
    spo2 = np.array([93, 95, 97, 97], dtype=float)
    copd = np.array([True] * 4)
    on_o2 = np.array([True, True, True, False])
    assert news2.spo2_points(spo2, copd, on_o2).tolist() == [1, 2, 3, 0]


def test_news2_total_includes_all_seven_parameters():
    frame = pd.DataFrame(
        {
            "resp_rate": [16],
            "spo2": [97],
            "sbp": [120],
            "heart_rate": [70],
            "temp_c": [37.0],
            "on_oxygen": [True],
            "acvpu": ["C"],
        }
    )
    assert news2.news2(frame).loc[0, "news2_total"] == 2 + 3


def test_news2_total_is_missing_when_a_vital_is_missing():
    frame = pd.DataFrame(
        {
            "resp_rate": [26, 16],
            "spo2": [90, np.nan],
            "sbp": [95, 120],
            "heart_rate": [120, 70],
            "temp_c": [38.5, 37.0],
        }
    )
    out = news2.news2(frame)
    assert out.loc[0, "news2_total"] == 3 + 3 + 2 + 2 + 1
    assert np.isnan(out.loc[1, "news2_total"])


# --- a tiny hospital ------------------------------------------------------------------------


def _activity() -> HospitalActivity:
    """Device D1 worn by encounter E1 (0-10h), then E2 (12h-); E1 transfers at 5h."""
    encounters = pd.DataFrame(
        {
            "encounter_id": ["E1", "E2"],
            "patient_id": ["P1", "P2"],
            "site_id": "SITE_A",
            "admit_unit_id": ["SITE_A-GENERAL", "SITE_A-RESPIRATORY"],
            "admit_ts": [h(0), h(12)],
            "discharge_ts": [h(10), pd.NaT],
        }
    )
    assignments = pd.DataFrame(
        {
            "device_id": "D1",
            "encounter_id": ["E1", "E2"],
            "patient_id": ["P1", "P2"],
            "start_ts": [h(0), h(12)],
            "end_ts": [h(10), pd.NaT],
        }
    )
    transfers = pd.DataFrame(
        {
            "encounter_id": ["E1"],
            "transfer_ts": [h(5)],
            "from_unit_id": ["SITE_A-GENERAL"],
            "to_unit_id": ["SITE_A-STEP_DOWN"],
            "from_bed_id": ["b1"],
            "to_bed_id": ["b2"],
        }
    )
    patients = pd.DataFrame(
        {"patient_id": ["P1", "P2"], "age": [70, 80], "sex": ["F", "M"], "n_conditions": [1, 0]}
    )
    for flag in features.PROFILE_FLAGS:
        patients[flag] = False
    for df in (encounters, assignments):
        for c in df.columns:
            if c.endswith("_ts"):
                df[c] = pd.to_datetime(df[c], utc=True)
    return HospitalActivity(encounters, assignments, transfers, patients, h(0), h(24))


def _readings(hr_fn) -> pd.DataFrame:
    ts = pd.date_range(h(0), h(24), freq="5min", inclusive="left")
    df = pd.DataFrame({"device_id": "D1", "ts": ts})
    for v in features.VITALS:
        df[v] = {
            "heart_rate": 70,
            "resp_rate": 16,
            "spo2": 97,
            "temp_c": 36.8,
            "sbp": 120,
            "dbp": 75,
        }[v]
    df["heart_rate"] = [hr_fn(t) for t in ts]
    return df


def test_readings_are_linked_by_device_and_time_window():
    linked = features.link_readings(_readings(lambda t: 70), _activity().device_assignments)
    assert set(linked.loc[linked["ts"] < h(10), "encounter_id"]) == {"E1"}
    assert set(linked.loc[linked["ts"] >= h(12), "encounter_id"]) == {"E2"}
    assert not linked["ts"].between(h(10), h(12), inclusive="left").any()  # device unused


def test_features_never_use_readings_from_or_after_the_prediction_time():
    spike = h(6)  # absurd heart rate from 06:00 onwards
    table = features.build_patient_hours(
        _readings(lambda t: 200 if t >= spike else 70),
        _activity(),
        pd.DataFrame(columns=["encounter_id", "event_ts"]),
    ).set_index(["encounter_id", "prediction_ts"])
    assert table.loc[("E1", spike), "heart_rate_max_1h"] == 70
    assert table.loc[("E1", spike + pd.Timedelta(hours=1)), "heart_rate_max_1h"] == 200


def test_prediction_times_only_while_on_the_ward():
    table = features.build_patient_hours(
        _readings(lambda t: 70), _activity(), pd.DataFrame(columns=["encounter_id", "event_ts"])
    )
    e1 = table[table["encounter_id"] == "E1"]["prediction_ts"]
    assert e1.min() == h(1) and e1.max() == h(9)
    assert (table[table["encounter_id"] == "E2"]["prediction_ts"] >= h(13)).all()


def test_unit_follows_transfers_that_happened_before_t():
    table = features.build_patient_hours(
        _readings(lambda t: 70), _activity(), pd.DataFrame(columns=["encounter_id", "event_ts"])
    ).set_index(["encounter_id", "prediction_ts"])
    assert table.loc[("E1", h(5)), "unit_type"] == "general"  # transfer at exactly 05:00
    assert table.loc[("E1", h(6)), "unit_type"] == "step_down"


def test_labels_look_six_hours_ahead_and_stay_unknown_until_the_window_closes():
    outcomes = pd.DataFrame({"encounter_id": ["E1"], "event_ts": [h(9.5)]})
    act = _activity()
    act.encounters.loc[0, "discharge_ts"] = h(9.5)
    table = features.build_patient_hours(_readings(lambda t: 70), act, outcomes).set_index(
        ["encounter_id", "prediction_ts"]
    )
    assert table.loc[("E1", h(3)), "label"] == 0  # event 6.5 h later
    assert table.loc[("E1", h(4)), "label"] == 1  # event within (4h, 10h]
    assert table.loc[("E1", h(9)), "label"] == 1
    e2 = table.loc["E2"]
    assert e2.loc[h(18), "label"] == 0  # window closes exactly at the data end
    assert np.isnan(e2.loc[h(19), "label"])  # still open at the end of the data
    assert e2.loc[h(19), "label_known_at"] == h(25)


def test_feature_columns_exclude_ids_time_and_label():
    cols = features.feature_columns(
        features.build_patient_hours(
            _readings(lambda t: 70), _activity(), pd.DataFrame(columns=["encounter_id", "event_ts"])
        )
    )
    for forbidden in ("label", "label_known_at", "prediction_ts", "encounter_id", "patient_id"):
        assert forbidden not in cols
    assert "news2_total" in cols and "spo2_trend_3h" in cols


def test_latest_nurse_observation_is_strictly_before_t():
    observations = pd.DataFrame(
        {
            "encounter_id": ["E1", "E1"],
            "patient_id": ["P1", "P1"],
            "obs_ts": [h(2), h(5)],
            "acvpu": ["A", "C"],
            "on_oxygen": [False, True],
            "o2_flow_lpm": [0.0, 2.0],
        }
    )
    table = features.build_patient_hours(
        _readings(lambda t: 70),
        _activity(),
        pd.DataFrame(columns=["encounter_id", "event_ts"]),
        nurse_observations=observations,
    ).set_index(["encounter_id", "prediction_ts"])
    assert table.loc[("E1", h(2)), "hours_since_obs"] != table.loc[("E1", h(2)), "hours_since_obs"]
    assert table.loc[("E1", h(5)), "acvpu"] == "A"  # the 05:00 obs is not yet visible at 05:00
    row = table.loc[("E1", h(6))]
    assert row["new_confusion"] and row["on_oxygen"] and row["hours_since_obs"] == 1
    assert row["news2_consciousness"] == 3 and row["news2_oxygen"] == 2
