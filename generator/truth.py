"""Truth generator: decides which patients deteriorate, when, and what that does to their vitals.

How it works (plain-language version in ml_model.md §5):

1. A **hidden rule** gives every patient, every hour, a small chance of starting to deteriorate.
   The chance rises with abnormal *true* vitals, age, chronic illness and an unobserved personal
   frailty. Its weights live only here, never in the data platform, so the model has to learn
   the pattern from data.
2. A started deterioration is a **trajectory**: over 2-12 hours (sometimes under 2) the patient's
   physiology is pushed toward a respiratory, septic or cardiac decompensation profile.
3. Most trajectories end in an **escalation event** (rapid-response call or ICU transfer), the
   moment progress reaches the escalation threshold. About a quarter **recover** on their own.
   After an escalation the patient leaves the ward, so their monitor stops.
4. Escalations are written as **outcome events**. Nobody knows at prediction time whether one is
   coming; the label "escalated within 6 h" can only be built once 6 hours have passed.

`TruthConfig.escalation_threshold` is the concept-drift lever: lowering it (a new protocol that
escalates earlier) changes outcomes for the *same* vitals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from generator import care, vitals
from generator.activity import HospitalActivity

# Physiology change at full severity, per deterioration kind.
DECOMPENSATION = {
    "respiratory": {"resp_rate": 10, "spo2": -9, "heart_rate": 22, "temp_c": 0.3, "sbp": -5},
    "sepsis": {"temp_c": 1.8, "heart_rate": 12, "resp_rate": 5, "spo2": -3, "sbp": -30, "dbp": -18},
    "cardiac": {"heart_rate": 28, "sbp": -35, "dbp": -15, "resp_rate": 6, "spo2": -4},
}
KINDS = tuple(DECOMPENSATION)

# The hidden rule. Weights on hourly "abnormality" features (see `_risk_features`).
_HIDDEN_WEIGHTS = {
    "hr_abn": 0.8,
    "rr_abn": 0.9,
    "spo2_abn": 1.0,
    "temp_abn": 0.6,
    "sbp_abn": 0.9,
    "age_z": 0.35,
    "n_conditions": 0.25,
    "copd": 0.4,
    "heart_failure": 0.6,
    "pneumonia": 0.8,  # active infection (respiratory-outbreak scenario)
}
_HIDDEN_INTERCEPT = -7.2
_FRAILTY_SD = 0.5  # unobserved patient-to-patient variation in risk


@dataclass(frozen=True)
class TruthConfig:
    escalation_threshold: float = 1.0  # progress at which staff escalate (lower = earlier)
    # Concept drift: from this moment a new protocol escalates at `new_escalation_threshold`.
    protocol_change_at: pd.Timestamp | None = None
    new_escalation_threshold: float = 0.7
    # The new protocol also reviews (and escalates) stable patients whose vitals score at least
    # `review_warning_score`, with this chance per hour. Same vitals, different outcome.
    review_prob_per_hour: float = 0.0
    review_warning_score: float = 1.0
    base_rate_scale: float = 1.0  # multiplies everyone's hourly odds of deteriorating
    recovery_prob: float = 0.25  # share of trajectories that resolve without escalation
    sudden_prob: float = 0.10  # share of events with under 2 hours of warning
    trajectory_hours: tuple[float, float] = (2.0, 12.0)
    sudden_hours: tuple[float, float] = (0.5, 1.5)
    icu_share: float = 0.3  # remaining escalations are rapid-response calls


@dataclass
class TruthResult:
    trajectories: pd.DataFrame  # ground-truth audit, one row per trajectory (never for the model)
    outcomes: pd.DataFrame  # outcome events, as the hospital would record them
    offsets: dict[str, np.ndarray]  # physiology offsets to apply, (slots, steps)
    stay_end_step: np.ndarray  # per stay: step at which the patient left (escalation or plan)
    progress: np.ndarray  # (slots, steps) how far into a deterioration (0 = none, 1 = full)
    kind_code: np.ndarray  # (slots, steps) index into KINDS of the ongoing deterioration, -1 none


def _relu(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, None)


def _as_grid(values, shape: tuple[int, int]) -> np.ndarray:
    """Per-patient values (n,) or per patient-hour values (n, hours) as an (n, hours) grid."""
    values = np.asarray(values, dtype=float)
    return np.broadcast_to(values[:, None] if values.ndim == 1 else values, shape)


def _risk_features(attributes, hourly: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Abnormality scores (0 = normal) per patient-hour, from true hourly vitals.

    `attributes` holds age, n_conditions, copd and heart_failure, either one value per row
    (a DataFrame) or one value per row-hour (a dict of arrays).
    """
    shape = hourly["heart_rate"].shape
    copd = _as_grid(attributes["copd"], shape)
    spo2_target = np.where(copd > 0, 88.0, 92.0)
    return {
        "hr_abn": _relu(hourly["heart_rate"] - 95) / 10 + _relu(55 - hourly["heart_rate"]) / 10,
        "rr_abn": _relu(hourly["resp_rate"] - 20) / 3,
        "spo2_abn": _relu(spo2_target - hourly["spo2"]) / 2,
        "temp_abn": _relu(hourly["temp_c"] - 37.8) / 0.5 + _relu(36.0 - hourly["temp_c"]) / 0.5,
        "sbp_abn": _relu(105 - hourly["sbp"]) / 10,
        "age_z": (_as_grid(attributes["age"], shape) - 65) / 15,
        "n_conditions": _as_grid(attributes["n_conditions"], shape),
        "copd": copd,
        "heart_failure": _as_grid(attributes["heart_failure"], shape),
        "pneumonia": _as_grid(attributes["pneumonia"], shape)
        if "pneumonia" in attributes
        else np.zeros(shape),
    }


# Features that make up the bedside "warning score" a review protocol reacts to.
_WARNING_FEATURES = ("hr_abn", "rr_abn", "spo2_abn", "temp_abn", "sbp_abn")


def hourly_onset_probability(
    attributes, hourly: dict[str, np.ndarray], frailty: np.ndarray, config: TruthConfig
) -> np.ndarray:
    """Chance per patient-hour that a deterioration trajectory starts (the hidden rule)."""
    features = _risk_features(attributes, hourly)
    shape = hourly["heart_rate"].shape
    logit = _HIDDEN_INTERCEPT + _as_grid(frailty, shape) + np.log(config.base_rate_scale)
    for name, weight in _HIDDEN_WEIGHTS.items():
        logit = logit + weight * features[name]
    with np.errstate(over="ignore", invalid="ignore"):
        p = 1 / (1 + np.exp(-logit))
    return np.nan_to_num(p, nan=0.0)  # empty devices (NaN vitals) never deteriorate


def _kind_probabilities(row: pd.Series) -> list[float]:
    """Which way a patient tends to deteriorate: respiratory / sepsis / cardiac."""
    if row.get("pneumonia", False):
        return [0.70, 0.25, 0.05]
    if row["copd"]:
        return [0.60, 0.25, 0.15]
    if row["heart_failure"] or row["atrial_fibrillation"]:
        return [0.20, 0.25, 0.55]
    return [0.30, 0.45, 0.25]


def _to_hourly(physiology: dict[str, np.ndarray], steps_per_hour: int) -> dict[str, np.ndarray]:
    n, steps = physiology["heart_rate"].shape
    hours = steps // steps_per_hour
    return {
        v: arr[:, : hours * steps_per_hour].reshape(n, hours, steps_per_hour).mean(axis=2)
        for v, arr in physiology.items()
    }


# Patients this close to their planned discharge are stable enough to go home.
_NO_ONSET_BEFORE_DISCHARGE_HOURS = 12
_NEVER = np.iinfo(np.int64).max // 2


def generate_truth(
    stays: pd.DataFrame,
    physiology: dict[str, np.ndarray],
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
    config: TruthConfig | None = None,
    occupant: np.ndarray | None = None,
    planned_end_step: np.ndarray | None = None,
) -> TruthResult:
    """Decide trajectories and outcomes from *undisturbed* true physiology.

    Rows of `physiology` are device slots. `occupant[slot, step]` is the row of `stays` wearing
    that device (-1 = empty); by default slot i is stay i for the whole period.
    `planned_end_step[stay]` is when the stay would end without deterioration.
    """
    config = config or TruthConfig()
    steps = len(timestamps)
    n_stays = len(stays)
    if occupant is None:
        occupant = np.broadcast_to(np.arange(n_stays)[:, None], (n_stays, steps))
    if planned_end_step is None:
        planned_end_step = np.full(n_stays, _NEVER)
    n_slots = occupant.shape[0]
    step_min = int((timestamps[1] - timestamps[0]) / pd.Timedelta(minutes=1))
    steps_per_hour = 60 // step_min
    quiet_steps = _NO_ONSET_BEFORE_DISCHARGE_HOURS * steps_per_hour

    frailty = rng.normal(0, _FRAILTY_SD, n_stays)
    hourly = _to_hourly(physiology, steps_per_hour)
    n_hours = hourly["heart_rate"].shape[1]
    occ_hour = occupant[:, : n_hours * steps_per_hour : steps_per_hour]
    row = np.where(occ_hour >= 0, occ_hour, 0)
    attributes = {
        c: np.where(occ_hour >= 0, stays[c].to_numpy(float)[row], np.nan)
        for c in ("age", "n_conditions", "copd", "heart_failure", "pneumonia")
        if c in stays
    }
    stay_frailty = np.where(occ_hour >= 0, frailty[row], 0.0)
    p_onset = hourly_onset_probability(attributes, hourly, stay_frailty, config)
    onset_draws = rng.random(p_onset.shape) < p_onset
    review_step = _protocol_reviews(
        config,
        _risk_features(attributes, hourly),
        occupant,
        occ_hour,
        planned_end_step,
        timestamps,
        steps_per_hour,
        np.random.default_rng(int(rng.integers(2**32))),  # own stream: trajectories unchanged
    )

    offsets = {v: np.zeros((n_slots, steps)) for v in vitals.VITALS}
    progress_grid = np.zeros((n_slots, steps), dtype=np.float32)
    kind_grid = np.full((n_slots, steps), -1, dtype=np.int8)
    actual_end = np.array(planned_end_step, dtype=np.int64)
    escalated_stays: set[int] = set()
    rows = []
    # Each device gets its own random stream, so a change at one device (e.g. a protocol review
    # cutting a stay short) can't alter what happens at any other device.
    trajectory_seed = int(rng.integers(2**32))
    for slot in range(n_slots):
        slot_rng = np.random.default_rng([trajectory_seed, slot])
        busy_until = 0
        for hour in np.flatnonzero(onset_draws[slot]):
            start = hour * steps_per_hour + int(slot_rng.integers(steps_per_hour))
            stay = int(occupant[slot, start]) if start < steps else -1
            if (
                stay < 0
                or stay in escalated_stays
                or start < busy_until
                or start >= planned_end_step[stay] - quiet_steps
                or start >= review_step[stay]
            ):
                continue
            info = stays.iloc[stay]
            kind = KINDS[slot_rng.choice(len(KINDS), p=_kind_probabilities(info))]
            severity = slot_rng.uniform(0.7, 1.2)
            sudden = slot_rng.random() < config.sudden_prob
            hours_range = config.sudden_hours if sudden else config.trajectory_hours
            length_h = slot_rng.uniform(*hours_range)
            length = max(1, round(length_h * steps_per_hour))
            recovers = (not sudden) and slot_rng.random() < config.recovery_prob
            icu = slot_rng.random() < config.icu_share

            progress = _progress_curve(length, recovers, slot_rng, steps_per_hour)
            threshold = _thresholds(timestamps, start, len(progress), config)
            event_step = _first_reaching(progress, threshold)
            if event_step is not None:
                progress = progress[: event_step + 1]
            limit = int(min(steps, review_step[stay]))  # a protocol review takes them first
            end = int(min(start + len(progress), limit, planned_end_step[stay]))
            for v, delta in DECOMPENSATION[kind].items():
                offsets[v][slot, start:end] += severity * delta * progress[: end - start]
            progress_grid[slot, start:end] = progress[: end - start]
            kind_grid[slot, start:end] = KINDS.index(kind)

            escalated = event_step is not None and start + event_step < limit
            if escalated:
                escalated_stays.add(stay)
                actual_end[stay] = start + event_step + 1
            rows.append(
                {
                    "patient_id": info["patient_id"],
                    "encounter_id": info["encounter_id"],
                    "device_id": info["device_id"],
                    "kind": kind,
                    "severity": round(severity, 2),
                    "sudden": sudden,
                    "start_ts": timestamps[start],
                    "planned_hours": round(length_h, 2),
                    "outcome": "escalated"
                    if escalated
                    else (
                        "recovered"
                        if event_step is None
                        else ("ongoing" if limit == steps else "pre-empted by review")
                    ),
                    "event_ts": timestamps[start + event_step] if escalated else pd.NaT,
                    "event_type": ("icu_transfer" if icu else "rapid_response")
                    if escalated
                    else None,
                }
            )
            busy_until = start + len(progress)

    for stay in np.flatnonzero(review_step < _NEVER):
        if stay in escalated_stays or review_step[stay] >= actual_end[stay]:
            continue
        info = stays.iloc[stay]
        event_ts = timestamps[review_step[stay]]
        actual_end[stay] = review_step[stay] + 1
        rows.append(
            {
                "patient_id": info["patient_id"],
                "encounter_id": info["encounter_id"],
                "device_id": info["device_id"],
                "kind": "protocol_review",
                "severity": 0.0,
                "sudden": False,
                "start_ts": event_ts,
                "planned_hours": 0.0,
                "outcome": "escalated",
                "event_ts": event_ts,
                "event_type": "rapid_response",  # reviews are rapid-response calls
            }
        )

    trajectories = pd.DataFrame(rows, columns=_TRAJECTORY_COLUMNS)
    for column in ("start_ts", "event_ts"):  # keep datetime dtype even when there are no rows
        trajectories[column] = pd.to_datetime(trajectories[column], utc=True)
    events = trajectories[trajectories["outcome"] == "escalated"]
    outcomes = pd.DataFrame(
        {
            "patient_id": events["patient_id"],
            "encounter_id": events["encounter_id"],
            "event_type": events["event_type"],
            "event_ts": events["event_ts"],
        }
    ).reset_index(drop=True)
    return TruthResult(trajectories, outcomes, offsets, actual_end, progress_grid, kind_grid)


_TRAJECTORY_COLUMNS = [
    "patient_id",
    "encounter_id",
    "device_id",
    "kind",
    "severity",
    "sudden",
    "start_ts",
    "planned_hours",
    "outcome",
    "event_ts",
    "event_type",
]


def _protocol_reviews(
    config: TruthConfig,
    features: dict[str, np.ndarray],
    occupant: np.ndarray,
    occ_hour: np.ndarray,
    planned_end_step: np.ndarray,
    timestamps: pd.DatetimeIndex,
    steps_per_hour: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Step at which the new protocol escalates each stay (_NEVER if it doesn't)."""
    review = np.full(len(planned_end_step), _NEVER, dtype=np.int64)
    if config.protocol_change_at is None or config.review_prob_per_hour <= 0:
        return review
    warning = sum(features[name] for name in _WARNING_FEATURES)
    first_hour = np.ceil((config.protocol_change_at - timestamps[0]) / pd.Timedelta(hours=1))
    hour_idx = np.arange(occ_hour.shape[1])[None, :]
    eligible = (occ_hour >= 0) & (warning >= config.review_warning_score) & (hour_idx >= first_hour)
    hits = eligible & (rng.random(eligible.shape) < config.review_prob_per_hour)
    for slot, hour in np.argwhere(hits):
        stay = occ_hour[slot, hour]
        step = hour * steps_per_hour + int(rng.integers(steps_per_hour))
        if (
            step < len(timestamps)
            and occupant[slot, step] == stay
            and step < planned_end_step[stay]
        ):
            review[stay] = min(review[stay], step)
    return review


def _progress_curve(
    length: int, recovers: bool, rng: np.random.Generator, steps_per_hour: int
) -> np.ndarray:
    """0 → 1 over `length` steps, accelerating. Recoveries peak below 1 then fade back to 0."""
    t = np.arange(1, length + 1) / length
    if not recovers:
        return t**1.4
    peak = rng.uniform(0.4, 0.85)
    rise = peak * t**1.4
    fall_steps = round(rng.uniform(2, 4) * steps_per_hour)
    fall = peak * (1 - np.arange(1, fall_steps + 1) / fall_steps)
    return np.concatenate([rise, fall])


def _thresholds(
    timestamps: pd.DatetimeIndex, start: int, length: int, config: TruthConfig
) -> np.ndarray:
    """Escalation threshold at each step of a trajectory (changes if a new protocol starts)."""
    threshold = np.full(length, config.escalation_threshold)
    if config.protocol_change_at is not None:
        ts = timestamps[start : start + length]
        after = np.zeros(length, dtype=bool)
        after[: len(ts)] = ts >= config.protocol_change_at
        after[len(ts) :] = after[len(ts) - 1] if len(ts) else False
        threshold[after] = config.new_escalation_threshold
    return threshold


def _first_reaching(progress: np.ndarray, threshold) -> int | None:
    hits = np.flatnonzero(progress >= threshold - 1e-9)
    return int(hits[0]) if len(hits) else None


MeasurementOffsetFn = Callable[[np.ndarray, pd.DatetimeIndex], dict[str, np.ndarray]]


@dataclass
class WardResult:
    stays: pd.DataFrame  # one row per patient-on-a-device stay (profile + baseline vitals)
    timestamps: pd.DatetimeIndex
    slot_devices: np.ndarray  # device_id of each physiology row
    occupant: np.ndarray  # (slots, steps) row of `stays` wearing the device, -1 = empty
    physiology: dict[str, np.ndarray]  # true vitals *including* deteriorations, (slots, steps)
    readings: pd.DataFrame  # what devices sent (stops when a patient leaves)
    truth: TruthResult
    # Actual hospital records (escalations applied). Only with an activity schedule.
    activity: HospitalActivity | None = None
    spo2_on_air: np.ndarray | None = None  # true SpO2 the patient would have without oxygen
    o2_flow: np.ndarray | None = None  # (slots, steps) supplemental oxygen, L/min
    nurse_observations: pd.DataFrame | None = None  # charted ACVPU + oxygen


def _stays_from_activity(activity: HospitalActivity, timestamps: pd.DatetimeIndex):
    """Stays, slot devices, occupancy grid and planned end steps from a hospital schedule."""
    profile_cols = [c for c in activity.patients.columns if c != "patient_id"]
    stays = activity.device_assignments.merge(
        activity.patients[["patient_id", *profile_cols]], on="patient_id"
    ).reset_index(drop=True)
    if "admission_reason" in activity.encounters:
        reason = stays["encounter_id"].map(
            activity.encounters.set_index("encounter_id")["admission_reason"]
        )
        stays["pneumonia"] = reason.eq("pneumonia").to_numpy()
    slot_devices = np.array(sorted(stays["device_id"].unique()))
    slot_of = {d: i for i, d in enumerate(slot_devices)}
    steps = len(timestamps)
    step = timestamps[1] - timestamps[0]

    def to_step(ts) -> int:
        if pd.isna(ts):
            return steps
        return int(np.clip(np.ceil((ts - timestamps[0]) / step), 0, steps))

    occupant = np.full((len(slot_devices), steps), -1)
    planned_end = np.empty(len(stays), dtype=np.int64)
    for i, s in stays.iterrows():
        a, b = to_step(s["start_ts"]), to_step(s["end_ts"])
        occupant[slot_of[s["device_id"]], a:b] = i
        planned_end[i] = b if pd.notna(s["end_ts"]) else _NEVER
    return stays, slot_devices, occupant, planned_end


def simulate_ward(
    inpatients: pd.DataFrame,
    start: pd.Timestamp,
    hours: float,
    seed: int = 42,
    vitals_config: vitals.VitalsConfig | None = None,
    truth_config: TruthConfig | None = None,
    measurement_offsets: dict[str, np.ndarray] | MeasurementOffsetFn | None = None,
    activity: HospitalActivity | None = None,
    bedside_care: bool = True,
) -> WardResult:
    """Vitals + truth together: undisturbed physiology → trajectories → disturbed physiology →
    readings. The same seed always gives the same hospital.

    `measurement_offsets` can also be a function of (slot_devices, timestamps), which is
    handy when devices are only known once the schedule has been laid out.

    Without `activity` the ward is static: `inpatients` stay for the whole period (one device
    each). With an `activity` schedule (generator.activity) patients come and go, devices are
    reused, and escalations end stays early.
    """
    vitals_config = vitals_config or vitals.VitalsConfig()
    base_seed, phys_seed, truth_seed, measure_seed, care_seed = np.random.SeedSequence(seed).spawn(
        5
    )
    timestamps = vitals.make_timestamps(start, hours, vitals_config.step_minutes)

    if activity is None:
        stays = inpatients.reset_index(drop=True)
        slot_devices = stays["device_id"].to_numpy()
        occupant = np.broadcast_to(np.arange(len(stays))[:, None], (len(stays), len(timestamps)))
        planned_end = None
    else:
        stays, slot_devices, occupant, planned_end = _stays_from_activity(activity, timestamps)

    baselines = vitals.make_baselines(stays, np.random.default_rng(base_seed))
    stays = stays.join(baselines.drop(columns=["patient_id", "device_id"]))

    def physiology(offsets=None):  # same seed → same wander, so only the offsets differ
        return vitals.simulate_physiology(
            stays, timestamps, np.random.default_rng(phys_seed), vitals_config, offsets, occupant
        )

    truth = generate_truth(
        stays,
        physiology(),
        timestamps,
        np.random.default_rng(truth_seed),
        truth_config,
        occupant,
        planned_end,
    )
    disturbed = physiology(truth.offsets)

    step_idx = np.arange(len(timestamps))[None, :]
    occ = np.where(occupant >= 0, occupant, 0)
    present = (occupant >= 0) & (step_idx < truth.stay_end_step[occ])

    spo2_on_air, o2_flow, observations = disturbed["spo2"], None, None
    if bedside_care:  # nurses give oxygen (raising true SpO2) and chart ACVPU
        steps_per_hour = int(pd.Timedelta(hours=1) / (timestamps[1] - timestamps[0]))
        copd_grid = np.where(occupant >= 0, stays["copd"].to_numpy(bool)[occ], False)
        care_rng, obs_seed = np.random.default_rng(care_seed), int(care_seed.generate_state(1)[0])
        disturbed["spo2"], o2_flow = care.apply_oxygen(
            spo2_on_air, copd_grid, occupant, steps_per_hour, care_rng
        )
        observations = care.nurse_observations(
            disturbed,
            o2_flow,
            truth.progress,
            truth.kind_code,
            KINDS,
            occupant,
            present,
            stays,
            timestamps,
            obs_seed,
        )

    if callable(measurement_offsets):
        measurement_offsets = measurement_offsets(slot_devices, timestamps)
    readings, _ = vitals.measure(
        disturbed,
        stays,
        timestamps,
        np.random.default_rng(measure_seed),
        vitals_config,
        measurement_offsets,
        slot_devices,
        present,
    )
    ward = WardResult(stays, timestamps, slot_devices, occupant, disturbed, readings, truth)
    ward.spo2_on_air, ward.o2_flow, ward.nurse_observations = spo2_on_air, o2_flow, observations
    if activity is not None:
        _apply_escalations(ward, activity)
    return ward


def _apply_escalations(ward: WardResult, activity: HospitalActivity) -> None:
    """Actual hospital records: escalated stays end at the event; later transfers never happen."""
    event = ward.truth.outcomes.set_index("encounter_id")["event_ts"]
    enc = activity.encounters.copy()
    hit = enc["encounter_id"].isin(event.index)
    if hit.any():
        enc.loc[hit, "discharge_ts"] = enc.loc[hit, "encounter_id"].map(event)
    enc["disposition"] = np.where(
        hit, "escalated", np.where(enc["discharge_ts"].notna(), "home", "in_hospital")
    )
    dev = activity.device_assignments.copy()
    dev_hit = dev["encounter_id"].isin(event.index)
    if dev_hit.any():
        dev.loc[dev_hit, "end_ts"] = dev.loc[dev_hit, "encounter_id"].map(event)
    moves = activity.transfers.merge(enc[["encounter_id", "discharge_ts"]], on="encounter_id")
    moves = moves[moves["discharge_ts"].isna() | (moves["transfer_ts"] < moves["discharge_ts"])]
    ward.activity = HospitalActivity(
        encounters=enc,
        device_assignments=dev,
        transfers=moves.drop(columns="discharge_ts").reset_index(drop=True),
        patients=activity.patients,
        start=activity.start,
        end=activity.end,
    )
