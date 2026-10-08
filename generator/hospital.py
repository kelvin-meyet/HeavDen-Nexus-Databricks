"""The pretend hospital network: sites, units, beds, and which patients are currently admitted.

Synthea gives us a general population. Real inpatients are older and sicker than that, so
`admit_inpatients` samples from the population with extra weight on age and chronic illness,
then places each patient on a site, a unit and a bed with a monitoring device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Site:
    site_id: str
    name: str
    city: str
    beds: int


SITES: tuple[Site, ...] = (
    Site("SITE_A", "HeavDen General Hospital", "Boston", 160),
    Site("SITE_B", "HeavDen Northshore Medical Center", "Salem", 110),
    Site("SITE_C", "HeavDen Valley Community Hospital", "Worcester", 90),
)

# Start of simulated time: the moment the baseline history begins.
SIM_START = pd.Timestamp("2026-11-01", tz="UTC")

# Units every site has, with the share of the site's beds in each.
UNITS: dict[str, float] = {"general": 0.55, "step_down": 0.25, "respiratory": 0.20}


def sites_frame() -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in SITES])


def units_frame() -> pd.DataFrame:
    rows = []
    for site in SITES:
        for unit, share in UNITS.items():
            rows.append(
                {
                    "unit_id": f"{site.site_id}-{unit.upper()}",
                    "site_id": site.site_id,
                    "unit_type": unit,
                    "beds": round(site.beds * share),
                }
            )
    return pd.DataFrame(rows)


def admission_weights(profiles: pd.DataFrame) -> np.ndarray:
    """Relative chance of each person being an inpatient: rises with age and chronic illness."""
    age = profiles["age"].to_numpy()
    age_factor = np.exp((age - 50) / 40)
    illness_factor = 1 + 0.8 * profiles["n_conditions"].to_numpy()
    # COPD and heart failure are among the commonest reasons for admission.
    admission_reasons = (
        1 + 3.0 * profiles["copd"].to_numpy() + 3.0 * profiles["heart_failure"].to_numpy()
    )
    weights = age_factor * illness_factor * admission_reasons * (age >= 18)
    return weights / weights.sum()


def _unit_preference(row: pd.Series) -> np.ndarray:
    """Respiratory patients tend to go to the respiratory unit, cardiac ones to step-down."""
    if row["copd"]:
        return np.array([0.25, 0.15, 0.60])
    if row["heart_failure"] or row["atrial_fibrillation"]:
        return np.array([0.35, 0.55, 0.10])
    return np.array([0.60, 0.20, 0.20])


def _place_in_units(
    inpatients: pd.DataFrame, site_ids: np.ndarray, rng: np.random.Generator
) -> list[str]:
    """Pick a unit for each patient by preference, but only among units with a free bed."""
    free = units_frame().set_index("unit_id")["beds"].to_dict()
    unit_types = list(UNITS)
    placed = []
    for (_, row), site in zip(inpatients.iterrows(), site_ids, strict=True):
        candidates = [f"{site}-{u.upper()}" for u in unit_types]
        probs = _unit_preference(row) * np.array([free[c] > 0 for c in candidates])
        if probs.sum() == 0:
            raise ValueError(f"{site} is full; admit fewer patients or add beds")
        unit_id = str(rng.choice(candidates, p=probs / probs.sum()))
        free[unit_id] -= 1
        placed.append(unit_id)
    return placed


def admit_inpatients(
    profiles: pd.DataFrame,
    n: int = 300,
    seed: int = 42,
    sim_start: pd.Timestamp = SIM_START,
) -> pd.DataFrame:
    """Pick `n` current inpatients and place them on a site, unit, bed and device.

    Admission times fall in the 5 days before `sim_start`, so patients have been in for a while.
    """
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(profiles), size=n, replace=False, p=admission_weights(profiles))
    inpatients = profiles.iloc[np.sort(chosen_idx)].reset_index(drop=True)

    site_share = np.array([s.beds for s in SITES], dtype=float)
    site_ids = rng.choice([s.site_id for s in SITES], size=n, p=site_share / site_share.sum())
    unit_ids = _place_in_units(inpatients, site_ids, rng)
    hours_before = rng.uniform(1, 5 * 24, size=n)

    placement = pd.DataFrame(
        {
            "encounter_id": [f"E-{100000 + i}" for i in range(n)],
            "patient_id": inpatients["patient_id"],
            "site_id": site_ids,
            "unit_id": unit_ids,
            "admit_ts": sim_start - pd.to_timedelta(hours_before, unit="h"),
        }
    )
    placement["admit_ts"] = placement["admit_ts"].dt.floor("min")

    # Bed and device numbers are sequential within each unit / site.
    placement["bed_id"] = (
        placement["unit_id"]
        + "-"
        + (placement.groupby("unit_id").cumcount() + 1).astype(str).str.zfill(3)
    )
    site_letter = placement["site_id"].str[-1]
    device_no = (placement.groupby("site_id").cumcount() + 1).astype(str).str.zfill(4)
    placement["device_id"] = "DEV-" + site_letter + "-" + device_no
    return placement.merge(inpatients, on="patient_id")
