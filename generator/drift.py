"""Drift scenarios: deliberately change the hospital so monitoring has something to catch.

Each scenario in `drift_config.yaml` pulls a different lever, so each tests a different part of
the platform (Plan.md §6):

- spo2_sensor_bias: readings only, not health (`measurement_offsets` on Site B devices)
- respiratory_outbreak: who gets admitted (`ActivityConfig.outbreak_from`)
- protocol_change: outcomes for the same vitals (`TruthConfig.protocol_change_at`)
- firmware_schema_change: the shape of device messages (post-processing of readings)

`simulate_episode` runs one episode (normally 48 simulated hours) with any set of scenarios on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generator import activity, truth, vitals

CONFIG_PATH = Path(__file__).with_name("drift_config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, dict]:
    return yaml.safe_load(path.read_text())["scenarios"]


def sensor_bias_offsets(settings: dict, starts_at: pd.Timestamp) -> truth.MeasurementOffsetFn:
    """SpO2 offsets for one site's devices, growing linearly from `starts_at` up to a cap."""

    def offsets(slot_devices: np.ndarray, timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        site_letter = settings["site"][-1]
        on_site = np.array([d.split("-")[1] == site_letter for d in slot_devices])
        days = np.clip((timestamps - starts_at) / pd.Timedelta(days=1), 0, None)
        bias = np.maximum(settings["spo2_offset_per_day"] * days.to_numpy(), settings["max_offset"])
        return {"spo2": np.outer(on_site, bias)}

    return offsets


def apply_firmware_change(
    readings: pd.DataFrame, settings: dict, starts_at: pd.Timestamp, seed: int = 0
) -> pd.DataFrame:
    """From `starts_at` devices send the new payload: new fields appear, dropped fields vanish.

    Dropped fields become NaN (the JSON writer omits them). `etco2` (exhaled CO2, mmHg) is
    derived from breathing rate: faster breathing blows off more CO2.
    """
    out = readings.copy()
    new = out["ts"] >= starts_at
    out.loc[new, "firmware"] = settings["firmware"]
    for field in settings.get("drop_fields", []):
        out[field] = out[field].astype(float)
        out.loc[new, field] = np.nan
    if "etco2" in settings.get("add_fields", []):
        rng = np.random.default_rng(seed)
        etco2 = 40 - 0.6 * (out["resp_rate"] - 16) + rng.normal(0, 1.5, len(out))
        out["etco2"] = np.where(new, np.clip(etco2, 15, 60).round(), np.nan)
    return out


@dataclass
class EpisodeResult:
    scenarios: list[str]
    windows: pd.DataFrame  # scenario, type, starts_at
    plan: activity.HospitalActivity  # the planned schedule (before escalations)
    ward: truth.WardResult  # vitals, truth and actual records

    @property
    def readings(self) -> pd.DataFrame:
        return self.ward.readings


def simulate_episode(
    profiles: pd.DataFrame,
    inpatients: pd.DataFrame,
    start: pd.Timestamp,
    hours: float = 48,
    scenarios: list[str] | tuple[str, ...] = (),
    seed: int = 42,
    config: dict[str, dict] | None = None,
) -> EpisodeResult:
    """One episode of hospital life with the chosen drift scenarios switched on.

    The same seed with and without a scenario gives the same hospital up to the moment the
    scenario starts, so differences are caused by the scenario.
    """
    config = config or load_config()
    unknown = set(scenarios) - set(config)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")

    def starts(name: str) -> pd.Timestamp:
        return start + pd.Timedelta(hours=config[name]["start_after_hours"])

    activity_config = activity.ActivityConfig()
    truth_config = truth.TruthConfig()
    offsets = None
    if "respiratory_outbreak" in scenarios:
        activity_config = activity.ActivityConfig(
            outbreak_from=starts("respiratory_outbreak"),
            outbreak_share=config["respiratory_outbreak"]["admit_share_pneumonia"],
        )
    if "protocol_change" in scenarios:
        settings = config["protocol_change"]
        truth_config = truth.TruthConfig(
            protocol_change_at=starts("protocol_change"),
            new_escalation_threshold=settings["new_escalation_threshold"],
            review_prob_per_hour=settings.get("review_prob_per_hour", 0.0),
            review_warning_score=settings.get("review_warning_score", 1.0),
        )
    if "spo2_sensor_bias" in scenarios:
        offsets = sensor_bias_offsets(config["spo2_sensor_bias"], starts("spo2_sensor_bias"))

    plan = activity.schedule_activity(profiles, inpatients, start, hours, seed, activity_config)
    ward = truth.simulate_ward(
        inpatients,
        start,
        hours,
        seed,
        vitals_config=vitals.VitalsConfig(),
        truth_config=truth_config,
        measurement_offsets=offsets,
        activity=plan,
    )
    if "firmware_schema_change" in scenarios:
        ward.readings = apply_firmware_change(
            ward.readings, config["firmware_schema_change"], starts("firmware_schema_change"), seed
        )

    windows = pd.DataFrame(
        [{"scenario": s, "type": config[s]["type"], "starts_at": starts(s)} for s in scenarios],
        columns=["scenario", "type", "starts_at"],
    )
    return EpisodeResult(list(scenarios), windows, plan, ward)
