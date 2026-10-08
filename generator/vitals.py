"""Vital-sign simulator: what each inpatient's monitor reports every 5 minutes.

Two layers are kept separate on purpose:

* **Physiology**: what the patient's body is really doing. Built from a personal baseline
  (age, illnesses, medicines), slow random wander, a day/night rhythm and fever coupling.
  The truth generator decides outcomes from this layer.
* **Measurement**: what the device reports. Physiology plus sensor noise, motion artefacts,
  stuck values, dropped messages and battery outages. The data platform only sees this layer.

`offsets` push the physiology (e.g. a deterioration trajectory or a fever). `measurement_offsets`
push only the readings (e.g. a miscalibrated SpO2 sensor), which is how a sensor-fault drift
scenario changes the data without changing anyone's health.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VITALS = ("heart_rate", "resp_rate", "spo2", "temp_c", "sbp", "dbp")
INTEGER_VITALS = ("heart_rate", "resp_rate", "spo2", "sbp", "dbp")

# Typical adult resting values: (mean, spread between patients).
POPULATION = {
    "heart_rate": (76, 8),
    "resp_rate": (16, 1.5),
    "spo2": (97, 1.0),
    "temp_c": (36.8, 0.2),
    "sbp": (124, 10),
    "dbp": (74, 7),
}

# Slow within-patient wander: (typical size, minutes it takes to "forget" a deviation).
WANDER = {
    "heart_rate": (4.0, 90),
    "resp_rate": (1.2, 60),
    "spo2": (0.6, 60),
    "temp_c": (0.15, 180),
    "sbp": (6.0, 120),
    "dbp": (4.0, 120),
}

# Day/night swing (peak around 16:00, lowest around 04:00).
CIRCADIAN_AMPLITUDE = {
    "heart_rate": 5.0,
    "resp_rate": 0.8,
    "spo2": 0.0,
    "temp_c": 0.3,
    "sbp": 7.0,
    "dbp": 4.0,
}
CIRCADIAN_PEAK_HOUR = 16

# Change per +1 °C of fever above the patient's normal temperature.
FEVER_COUPLING = {"heart_rate": 10.0, "resp_rate": 2.0}

# Device measurement noise (standard deviation per reading).
NOISE_SD = {
    "heart_rate": 1.5,
    "resp_rate": 0.8,
    "spo2": 0.5,
    "temp_c": 0.05,
    "sbp": 3.0,
    "dbp": 2.0,
}

# Plausible *resting* range for a patient's personal baseline (sickness can go beyond these).
RESTING_RANGE = {
    "heart_rate": (50, 110),
    "resp_rate": (12, 24),
    "spo2": (86, 100),
    "temp_c": (36.2, 37.4),
    "sbp": (95, 180),
    "dbp": (55, 105),
}

# Physically possible ranges; values are clipped to these.
LIMITS = {
    "heart_rate": (20, 250),
    "resp_rate": (4, 60),
    "spo2": (50, 100),
    "temp_c": (32.0, 43.0),
    "sbp": (50, 250),
    "dbp": (25, 150),
}


@dataclass(frozen=True)
class VitalsConfig:
    """Switches and rates for each realism layer. Turning layers off is useful for learning."""

    step_minutes: int = 5
    wander: bool = True
    circadian: bool = True
    fever_coupling: bool = True
    noise: bool = True
    artefacts: bool = True
    faults: bool = True
    dropout_prob: float = 0.015  # a single message lost
    stuck_prob_per_hour: float = 0.01  # a sensor repeats its last value for 30-60 min
    battery_drain_pct_per_hour: tuple[float, float] = (1.0, 2.0)
    battery_swap_minutes: tuple[int, int] = (30, 90)  # offline time when a battery dies
    firmware: str = "3.1.4"


@dataclass
class VitalsResult:
    baselines: pd.DataFrame
    timestamps: pd.DatetimeIndex
    physiology: dict[str, np.ndarray]  # vital -> (patients, steps) true values
    readings: pd.DataFrame  # one row per message the devices actually sent
    motion: np.ndarray = field(repr=False)


# Profile flags (from generator.synthea.build_profiles) that change a patient's baseline.
_FLAGS = (
    "copd",
    "heart_failure",
    "diabetes",
    "ckd",
    "hypertension",
    "atrial_fibrillation",
    "on_beta_blocker",
    "pneumonia",  # admission reason during the respiratory-outbreak drift scenario
)

# Baseline change for patients admitted with pneumonia.
PNEUMONIA_SHIFT = {"resp_rate": 6.0, "temp_c": 0.9, "spo2": -3.0, "heart_rate": 10.0}


def make_timestamps(start: pd.Timestamp, hours: float, step_minutes: int = 5) -> pd.DatetimeIndex:
    steps = int(hours * 60 // step_minutes)
    return pd.date_range(start, periods=steps, freq=f"{step_minutes}min")


def make_baselines(inpatients: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each patient's personal normal values, shaped by age, illnesses and medicines."""
    n = len(inpatients)
    age = inpatients["age"].to_numpy()
    flag = {
        c: inpatients[c].to_numpy(dtype=bool) if c in inpatients else np.zeros(n, dtype=bool)
        for c in _FLAGS
    }
    base = {v: rng.normal(mean, sd, n) for v, (mean, sd) in POPULATION.items()}

    over_50 = np.clip(age - 50, 0, None)
    base["sbp"] += 0.4 * over_50  # arteries stiffen with age
    base["heart_rate"] -= 0.1 * over_50

    base["spo2"] = np.where(flag["copd"], rng.normal(91, 1.5, n), base["spo2"])
    base["resp_rate"] += 3 * flag["copd"]

    base["heart_rate"] += 8 * flag["heart_failure"] + 10 * flag["atrial_fibrillation"]
    base["heart_rate"] -= 10 * flag["on_beta_blocker"]

    base["sbp"] += 14 * flag["hypertension"] + 5 * (flag["diabetes"] | flag["ckd"])
    base["dbp"] += 8 * flag["hypertension"]

    for v, shift in PNEUMONIA_SHIFT.items():
        base[v] += shift * flag["pneumonia"]

    out = inpatients[["patient_id", "device_id"]].reset_index(drop=True).copy()
    for v in VITALS:
        out[v] = np.clip(base[v], *RESTING_RANGE[v])
    # Atrial fibrillation makes heart rate more irregular.
    out["hr_wander_scale"] = 1 + 0.4 * flag["atrial_fibrillation"]
    return out


def _ou_process(
    rng: np.random.Generator,
    n: int,
    steps: int,
    size: float,
    memory_min: float,
    dt: float,
    reset: np.ndarray | None = None,
) -> np.ndarray:
    """Mean-reverting random wander (Ornstein-Uhlenbeck) with stationary sd = `size`.

    Where `reset` is True (a new patient on the device) the wander starts afresh.
    """
    phi = np.exp(-dt / memory_min)
    eps = rng.standard_normal((n, steps))
    x = np.empty((n, steps))
    x[:, 0] = size * eps[:, 0]
    step_sd = size * np.sqrt(1 - phi**2)
    for t in range(1, steps):
        x[:, t] = phi * x[:, t - 1] + step_sd * eps[:, t]
        if reset is not None:
            x[reset[:, t], t] = size * eps[reset[:, t], t]
    return x


def _circadian(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour + timestamps.minute / 60
    return np.cos(2 * np.pi * (hour.to_numpy() - CIRCADIAN_PEAK_HOUR) / 24)


def simulate_physiology(
    baselines: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    config: VitalsConfig | None = None,
    offsets: dict[str, np.ndarray] | None = None,
    occupant: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """True vital signs, shape (slots, steps) per vital.

    By default slot i is baseline row i for the whole period. With hospital activity,
    `occupant[slot, step]` gives the baseline row of whoever wears that device at that step
    (-1 = nobody; those cells are NaN).
    """
    config = config or VitalsConfig()
    steps = len(timestamps)
    if occupant is None:
        occupant = np.broadcast_to(np.arange(len(baselines))[:, None], (len(baselines), steps))
    n = occupant.shape[0]
    occupied = occupant >= 0
    row = np.where(occupied, occupant, 0)
    reset = np.zeros_like(occupied)
    reset[:, 1:] = occupant[:, 1:] != occupant[:, :-1]
    offsets = offsets or {}
    rhythm = _circadian(timestamps) if config.circadian else np.zeros(steps)

    def per_slot(column: str) -> np.ndarray:
        return np.where(occupied, baselines[column].to_numpy()[row], np.nan)

    deviation = {}
    for v in VITALS:
        size, memory = WANDER[v]
        dev = (
            _ou_process(rng, n, steps, size, memory, config.step_minutes, reset)
            if config.wander
            else np.zeros((n, steps))
        )
        if v == "heart_rate":
            dev *= np.nan_to_num(per_slot("hr_wander_scale"), nan=1.0)
        deviation[v] = dev + CIRCADIAN_AMPLITUDE[v] * rhythm + offsets.get(v, 0.0)

    if config.wander:
        deviation["dbp"] += 0.5 * (deviation["sbp"] - CIRCADIAN_AMPLITUDE["sbp"] * rhythm)
    if config.fever_coupling:
        fever = deviation["temp_c"] - CIRCADIAN_AMPLITUDE["temp_c"] * rhythm
        for v, per_degree in FEVER_COUPLING.items():
            deviation[v] += per_degree * fever

    return {v: np.clip(per_slot(v) + deviation[v], *LIMITS[v]) for v in VITALS}


def _motion(rng: np.random.Generator, timestamps: pd.DatetimeIndex, n: int, dt: int) -> np.ndarray:
    """Patient movement 0-1: low at night, some daytime activity, occasional bursts."""
    hour = timestamps.hour.to_numpy()
    awake = (hour >= 7) & (hour < 22)
    level = np.where(awake, 0.2, 0.04)
    wiggle = 0.08 * _ou_process(rng, n, len(timestamps), 1.0, 30, dt)
    bursts = (rng.random((n, len(timestamps))) < np.where(awake, 0.04, 0.005)) * rng.uniform(
        0.5, 0.9, (n, len(timestamps))
    )
    return np.clip(level + wiggle + bursts, 0, 1)


def _battery(
    rng: np.random.Generator, n: int, steps: int, config: VitalsConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Battery % per step and whether the device is online (dead batteries take time to swap)."""
    dt_h = config.step_minutes / 60
    drain = rng.uniform(*config.battery_drain_pct_per_hour, n) * dt_h
    level = rng.uniform(40, 100, n)
    offline_left = np.zeros(n, dtype=int)
    battery = np.empty((n, steps))
    online = np.ones((n, steps), dtype=bool)
    lo, hi = config.battery_swap_minutes
    for t in range(steps):
        dead = (level <= 0) & (offline_left == 0)
        offline_left[dead] = rng.integers(lo, hi + 1, dead.sum()) // config.step_minutes
        swapping = offline_left > 0
        online[:, t] = ~swapping
        offline_left[swapping] -= 1
        level[swapping & (offline_left == 0)] = 100.0
        battery[:, t] = np.clip(level, 0, 100)
        level[~swapping] -= drain[~swapping]
    return battery, online


def measure(
    physiology: dict[str, np.ndarray],
    baselines: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    config: VitalsConfig | None = None,
    measurement_offsets: dict[str, np.ndarray] | None = None,
    slot_devices: np.ndarray | None = None,
    occupied: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn true physiology into the messages devices send. Returns (readings, motion).

    `slot_devices` names the device of each physiology row (default: baselines' device_id).
    Only cells where `occupied` is True (a patient wears the device) produce messages.
    """
    config = config or VitalsConfig()
    n, steps = physiology["heart_rate"].shape
    if slot_devices is None:
        slot_devices = baselines["device_id"].to_numpy()
    if occupied is None:
        occupied = ~np.isnan(physiology["heart_rate"])
    measurement_offsets = measurement_offsets or {}
    motion = (
        _motion(rng, timestamps, n, config.step_minutes)
        if config.artefacts
        else np.zeros((n, steps))
    )

    measured = {}
    for v in VITALS:
        value = physiology[v] + measurement_offsets.get(v, 0.0)
        if config.noise:
            value = value + rng.normal(0, NOISE_SD[v], (n, steps))
        measured[v] = value

    if config.artefacts:
        # Movement confuses optical sensors: heart rate jumps, SpO2 reads falsely low.
        moving = (motion > 0.6) & (rng.random((n, steps)) < 0.5)
        measured["heart_rate"] += moving * rng.uniform(5, 25, (n, steps))
        measured["spo2"] -= moving * rng.uniform(2, 8, (n, steps))

    for v in VITALS:
        measured[v] = np.clip(np.nan_to_num(measured[v], nan=LIMITS[v][0]), *LIMITS[v])
        measured[v] = (
            np.round(measured[v]).astype(int) if v in INTEGER_VITALS else np.round(measured[v], 1)
        )

    sent = occupied.copy()
    battery = np.full((n, steps), 100.0)
    if config.faults:
        _stick_values(measured, rng, config)
        battery, online = _battery(rng, n, steps, config)
        sent &= online & (rng.random((n, steps)) >= config.dropout_prob)

    patient_idx, step_idx = np.nonzero(sent)
    readings = pd.DataFrame(
        {
            "device_id": slot_devices[patient_idx],
            "ts": timestamps[step_idx],
            **{v: measured[v][patient_idx, step_idx] for v in VITALS},
            "motion": np.round(motion[patient_idx, step_idx], 2),
            "battery_pct": np.round(battery[patient_idx, step_idx]).astype(int),
            "firmware": config.firmware,
        }
    )
    return readings.sort_values(["ts", "device_id"], ignore_index=True), motion


def _stick_values(
    measured: dict[str, np.ndarray], rng: np.random.Generator, config: VitalsConfig
) -> None:
    """Occasionally a sensor freezes and repeats one value for 30-60 minutes."""
    n, steps = measured["heart_rate"].shape
    p_step = config.stuck_prob_per_hour * config.step_minutes / 60
    starts = np.argwhere(rng.random((n, steps)) < p_step)
    for patient, t in starts:
        v = VITALS[rng.integers(len(VITALS))]
        length = rng.integers(30, 61) // config.step_minutes
        measured[v][patient, t : t + length] = measured[v][patient, t]


def simulate_vitals(
    inpatients: pd.DataFrame,
    start: pd.Timestamp,
    hours: float,
    seed: int = 42,
    config: VitalsConfig | None = None,
    offsets: dict[str, np.ndarray] | None = None,
    measurement_offsets: dict[str, np.ndarray] | None = None,
) -> VitalsResult:
    """Baselines → physiology → measured readings for every inpatient, reproducibly."""
    config = config or VitalsConfig()
    baseline_rng, physiology_rng, measure_rng = np.random.default_rng(seed).spawn(3)
    timestamps = make_timestamps(start, hours, config.step_minutes)
    baselines = make_baselines(inpatients, baseline_rng)
    physiology = simulate_physiology(baselines, timestamps, physiology_rng, config, offsets)
    readings, motion = measure(
        physiology, baselines, timestamps, measure_rng, config, measurement_offsets
    )
    return VitalsResult(baselines, timestamps, physiology, readings, motion)
