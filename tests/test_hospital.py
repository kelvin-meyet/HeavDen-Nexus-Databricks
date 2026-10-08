import numpy as np
import pandas as pd
import pytest

from generator.hospital import admission_weights, admit_inpatients, units_frame
from generator.synthea import CONDITION_KEYWORDS


def _population(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "age": rng.integers(18, 95, size=n),
            "sex": rng.choice(["F", "M"], size=n),
        }
    )
    for flag in CONDITION_KEYWORDS:
        df[flag] = rng.random(n) < 0.1
    df["n_conditions"] = df[list(CONDITION_KEYWORDS)].sum(axis=1)
    return df


def test_admission_is_reproducible():
    pop = _population()
    a = admit_inpatients(pop, n=100, seed=3)
    b = admit_inpatients(pop, n=100, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_no_unit_exceeds_its_beds():
    admitted = admit_inpatients(_population(), n=300)
    occupancy = admitted.groupby("unit_id").size()
    beds = units_frame().set_index("unit_id")["beds"]
    assert (occupancy <= beds.reindex(occupancy.index)).all()


def test_beds_and_devices_are_unique():
    admitted = admit_inpatients(_population(), n=300)
    assert admitted["bed_id"].is_unique
    assert admitted["device_id"].is_unique
    assert admitted["patient_id"].is_unique


def test_older_and_sicker_people_are_more_likely_to_be_admitted():
    pop = pd.DataFrame(
        {"age": [30, 80, 80], "n_conditions": [0, 0, 3], "copd": False, "heart_failure": False}
    )
    w = admission_weights(pop)
    assert w[0] < w[1] < w[2]


def test_children_are_never_admitted():
    pop = pd.DataFrame(
        {"age": [10, 60], "n_conditions": [5, 0], "copd": [True, False], "heart_failure": False}
    )
    assert admission_weights(pop)[0] == 0


def test_full_hospital_raises():
    with pytest.raises(ValueError, match="full"):
        admit_inpatients(_population(n=2000), n=400)
