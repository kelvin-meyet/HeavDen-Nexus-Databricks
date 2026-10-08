"""Bedside care: supplemental oxygen and nurse observations (ACVPU).

Two parts of NEWS2 can't come from a wearable monitor, so the simulation adds what ward nurses do:

* **Supplemental oxygen.** Each hour a nurse checks the patient's SpO2. Below the target range
  (92-96%, or 88-92% for COPD) they start or turn up oxygen; well above it they wean it.
  Oxygen raises the patient's *true* SpO2, which masks a falling saturation, exactly as in a
  real ward. That's why NEWS2 adds 2 points for anyone on oxygen.
* **Nurse observations.** Nurses chart a set of observations every 4 hours (more often when the
  patient scores higher): consciousness on the **ACVPU** scale (Alert, new Confusion, responds
  to Voice, to Pain, Unresponsive) and current oxygen. New confusion becomes likely late in a
  deterioration, especially sepsis.

Simplification: oxygen treats the symptom only. Who deteriorates and when was already decided by
the truth generator, so oxygen doesn't change outcomes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Oxygen steps a nurse titrates through (litres/minute) and the SpO2 points each one adds.
O2_FLOWS = np.array([0, 2, 4, 6, 10])
O2_BOOST = np.array([0.0, 3.0, 6.0, 8.0, 11.0])

# Chance of non-alert consciousness at full deterioration, by kind (respiratory/sepsis/cardiac).
CONFUSION_AT_FULL = {"respiratory": 0.35, "sepsis": 0.65, "cardiac": 0.30}
BACKGROUND_CONFUSION = 0.002  # occasional new confusion unrelated to deterioration


@dataclass(frozen=True)
class CareConfig:
    start_o2_prob: float = 0.8  # chance a nurse acts within the hour when SpO2 is low
    wean_prob: float = 0.5
    routine_obs_hours: int = 4  # NEWS2 0-2
    medium_obs_hours: int = 2  # NEWS2 3-4
    urgent_obs_hours: int = 1  # NEWS2 >= 5


def _targets(copd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.where(copd, 88.0, 92.0), np.where(copd, 92.0, 96.0)


def apply_oxygen(
    spo2_on_air: np.ndarray,
    copd: np.ndarray,
    occupant: np.ndarray,
    steps_per_hour: int,
    rng: np.random.Generator,
    config: CareConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hourly oxygen titration. Returns (true SpO2 with oxygen, oxygen flow L/min), both
    (slots, steps).

    `spo2_on_air` is the true SpO2 the patient would have without oxygen. `copd` is per slot and
    step (the wearer's COPD status). A new wearer starts on room air.
    """
    config = config or CareConfig()
    n, steps = spo2_on_air.shape
    level = np.zeros(n, dtype=int)
    flow_level = np.zeros((n, steps), dtype=int)
    for h0 in range(0, steps, steps_per_hour):
        h1 = min(h0 + steps_per_hour, steps)
        new_wearer = occupant[:, h0] != (occupant[:, h0 - 1] if h0 else -2)
        level[new_wearer | (occupant[:, h0] < 0)] = 0
        flow_level[:, h0:h1] = level[:, None]
        # the nurse reviews at the end of the hour
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            on_o2 = np.nanmean(
                np.minimum(100, spo2_on_air[:, h0:h1] + O2_BOOST[level][:, None]), axis=1
            )
        low, high = _targets(copd[:, h0])
        draws = rng.random((n, 2))
        present = occupant[:, h0] >= 0
        up = present & (on_o2 < low) & (draws[:, 0] < config.start_o2_prob)
        down = present & (level > 0) & (on_o2 > high + 1) & (draws[:, 1] < config.wean_prob)
        level = np.clip(level + up - down, 0, len(O2_FLOWS) - 1)
    spo2 = np.minimum(100, spo2_on_air + O2_BOOST[flow_level])
    return spo2, O2_FLOWS[flow_level].astype(float)


def _vitals_news2(hourly: dict[str, np.ndarray], copd: np.ndarray) -> np.ndarray:
    """NEWS2 from vitals only, as the nurse would tally it (used to pick observation frequency)."""
    from heavden.ml import news2

    points = (
        news2.resp_rate_points(hourly["resp_rate"])
        + news2.spo2_points(hourly["spo2"], copd)
        + news2.sbp_points(hourly["sbp"])
        + news2.heart_rate_points(hourly["heart_rate"])
        + news2.temp_points(hourly["temp_c"])
    )
    return np.nan_to_num(points, nan=0.0)


def nurse_observations(
    physiology: dict[str, np.ndarray],
    o2_flow: np.ndarray,
    progress: np.ndarray,
    kind_code: np.ndarray,
    kinds: tuple[str, ...],
    occupant: np.ndarray,
    present: np.ndarray,
    stays: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    seed: int,
    config: CareConfig | None = None,
) -> pd.DataFrame:
    """Charted observations: one row per nurse visit with ACVPU and oxygen.

    Each device slot has its own random stream (seeded from `seed` and the slot number), so
    changes at one bed never alter another bed's charting.
    """
    config = config or CareConfig()
    n, steps = occupant.shape
    sph = int(pd.Timedelta(hours=1) / (timestamps[1] - timestamps[0]))
    hours = steps // sph
    copd_per_stay = stays["copd"].to_numpy(bool)
    confusion_at_full = np.array([CONFUSION_AT_FULL[k] for k in kinds])

    # Vitals-only NEWS2 per slot-hour (the wearer at the start of each hour).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty slot-hours are all-NaN
        hourly = {
            v: np.nanmean(arr[:, : hours * sph].reshape(n, hours, sph), axis=2)
            for v, arr in physiology.items()
        }
    wearer = occupant[:, : hours * sph : sph]
    copd_grid = np.where(wearer >= 0, copd_per_stay[np.where(wearer >= 0, wearer, 0)], False)
    vitals_score = _vitals_news2(hourly, copd_grid)

    rows = []
    for slot in range(n):
        rng = np.random.default_rng([seed, slot])
        next_due = None
        current = -1
        for h in range(hours):
            s0 = h * sph
            stay = occupant[slot, s0]
            if stay != current:  # new wearer: first observations on arrival
                current, next_due = stay, h
            if stay < 0 or h < next_due:
                continue
            step = s0 + int(rng.integers(sph))
            if not present[slot, step]:
                continue
            score = vitals_score[slot, h] + (2 if o2_flow[slot, step] > 0 else 0)
            p = progress[slot, step]
            k = kind_code[slot, step]
            p_confused = BACKGROUND_CONFUSION + (confusion_at_full[k] * p**2 if k >= 0 else 0.0)
            if rng.random() < p_confused:
                acvpu = "C" if (p < 0.9 or rng.random() < 0.7) else rng.choice(["V", "P"])
            else:
                acvpu = "A"
            rows.append(
                {
                    "encounter_id": stays.iloc[stay]["encounter_id"],
                    "patient_id": stays.iloc[stay]["patient_id"],
                    "obs_ts": timestamps[step],
                    "acvpu": acvpu,
                    "on_oxygen": bool(o2_flow[slot, step] > 0),
                    "o2_flow_lpm": float(o2_flow[slot, step]),
                }
            )
            if score >= 5 or acvpu != "A":
                gap = config.urgent_obs_hours
            elif score >= 3:
                gap = config.medium_obs_hours
            else:
                gap = config.routine_obs_hours
            next_due = h + gap
    obs = pd.DataFrame(
        rows, columns=["encounter_id", "patient_id", "obs_ts", "acvpu", "on_oxygen", "o2_flow_lpm"]
    )
    obs["obs_ts"] = pd.to_datetime(obs["obs_ts"], utc=True)
    return obs.sort_values(["obs_ts", "encounter_id"], ignore_index=True)
