"""Hospital activity: admissions, discharges and transfers over simulated time.

The ward isn't static. Patients go home after a realistic length of stay, the freed bed is
cleaned and refilled, and some patients move between units. This module plans that schedule
(before anyone deteriorates). `generator.truth.simulate_ward` then ends stays early for
patients who escalate.

Devices are **wearables**: a patient gets a free monitor at admission, keeps it through any
transfer, and returns it at discharge, after which it is reused by someone else. A device is
therefore a "slot" that hosts a sequence of patients.

`sql_snapshot` shows what the hospital's Azure SQL tables look like at any moment, including
each row's `last_updated`, which is how Lakeflow Connect finds what changed since its last run.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from generator import hospital

UNIT_TYPES = list(hospital.UNITS)


@dataclass(frozen=True)
class ActivityConfig:
    los_median_days: float = 4.5  # typical medical-ward length of stay
    los_sigma: float = 0.6  # spread of the (log-normal) length of stay
    min_los_hours: float = 12.0
    # Time a bed stays empty (cleaning + waiting for the next admission). Sized so the
    # hospital runs about 83% full, like the ~300 of 360 beds at the start.
    bed_idle_hours: tuple[float, float] = (4.0, 48.0)
    transfer_prob: float = 0.15  # share of stays with one move to another unit
    # Respiratory-outbreak drift: from this moment, this share of admissions have pneumonia.
    outbreak_from: pd.Timestamp | None = None
    outbreak_share: float = 0.35


@dataclass
class HospitalActivity:
    encounters: pd.DataFrame  # one row per stay; discharge_ts NaT = still in hospital at the end
    device_assignments: pd.DataFrame  # which device each stay wore, and when
    transfers: pd.DataFrame  # unit/bed moves (in SQL these are UPDATEs of encounters)
    patients: pd.DataFrame  # profiles of everyone admitted during the period
    start: pd.Timestamp
    end: pd.Timestamp


def _all_beds() -> pd.DataFrame:
    units = hospital.units_frame()
    rows = [
        {"bed_id": f"{u.unit_id}-{k:03d}", "unit_id": u.unit_id, "site_id": u.site_id}
        for u in units.itertuples()
        for k in range(1, u.beds + 1)
    ]
    return pd.DataFrame(rows)


def _unit_type(unit_id: str) -> str:
    return unit_id.split("-")[-1].lower()


def _preference_matrix(profiles: pd.DataFrame) -> np.ndarray:
    """Preference of every profile for each unit type (rows sum to 1)."""
    return np.stack([hospital._unit_preference(row) for _, row in profiles.iterrows()])


def schedule_activity(
    profiles: pd.DataFrame,
    inpatients: pd.DataFrame,
    start: pd.Timestamp,
    hours: float,
    seed: int = 42,
    config: ActivityConfig | None = None,
) -> HospitalActivity:
    """Plan admissions, discharges and transfers from `start` for `hours`.

    `inpatients` (from `hospital.admit_inpatients`) are the patients already in hospital at
    `start`; `profiles` is the population new admissions are drawn from.
    """
    config = config or ActivityConfig()
    rng = np.random.default_rng(seed)
    end = start + pd.Timedelta(hours=hours)

    profiles = profiles.reset_index(drop=True)
    weights = hospital.admission_weights(profiles)
    preference = _preference_matrix(profiles)
    profile_row = {pid: i for i, pid in enumerate(profiles["patient_id"])}

    beds = _all_beds()
    bed_unit = dict(zip(beds["bed_id"], beds["unit_id"], strict=True))
    bed_site = dict(zip(beds["bed_id"], beds["site_id"], strict=True))
    occupant: dict[str, str | None] = dict.fromkeys(beds["bed_id"])
    free_devices: dict[str, list[str]] = {
        s.site_id: [f"DEV-{s.site_id[-1]}-{k:04d}" for k in range(s.beds, 0, -1)]
        for s in hospital.SITES
    }
    in_hospital: set[str] = set()

    encounters: dict[str, dict] = {}
    assignments: dict[str, dict] = {}  # by encounter_id
    transfers: list[dict] = []
    events: list[tuple] = []  # (time, tiebreak, kind, payload)
    tiebreak = itertools.count()
    encounter_no = itertools.count(100000)

    def los() -> pd.Timedelta:
        days = rng.lognormal(np.log(config.los_median_days), config.los_sigma)
        return pd.Timedelta(hours=max(config.min_los_hours, days * 24))

    def push(time: pd.Timestamp, kind: str, payload: dict) -> None:
        if time < end:
            heapq.heappush(events, (time, next(tiebreak), kind, payload))

    def open_stay(patient_id, bed_id, admit_ts, device_id, planned_discharge, reason=None):
        encounter_id = f"E-{next(encounter_no)}"
        encounters[encounter_id] = {
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "site_id": bed_site[bed_id],
            "admit_unit_id": bed_unit[bed_id],
            "admit_bed_id": bed_id,
            "unit_id": bed_unit[bed_id],
            "bed_id": bed_id,
            "admit_ts": admit_ts,
            "planned_discharge_ts": planned_discharge,
            "discharge_ts": pd.NaT,
            "device_id": device_id,
            "admission_reason": reason,
        }
        occupant[bed_id] = encounter_id
        in_hospital.add(patient_id)
        assignments[encounter_id] = {
            "device_id": device_id,
            "encounter_id": encounter_id,
            "patient_id": patient_id,
            "start_ts": admit_ts,
            "end_ts": pd.NaT,
        }
        push(planned_discharge, "discharge", {"encounter_id": encounter_id})
        if rng.random() < config.transfer_prob:
            # Patients already in hospital at the start can only move from now on.
            earliest = max(admit_ts, start)
            when = earliest + (planned_discharge - earliest) * rng.uniform(0.2, 0.8)
            push(when, "transfer", {"encounter_id": encounter_id})

    # Patients already in hospital: keep their bed and device, plan their remaining stay.
    for row in inpatients.itertuples():
        planned = row.admit_ts + los()
        if planned <= start + pd.Timedelta(hours=2):
            planned = start + pd.Timedelta(hours=rng.uniform(2, 48))
        free_devices[row.site_id].remove(row.device_id)
        open_stay(row.patient_id, row.bed_id, row.admit_ts, row.device_id, planned.floor("min"))

    # Beds empty at the start get an admission after a short delay.
    for bed_id, enc in occupant.items():
        if enc is None:
            push(
                start + pd.Timedelta(hours=rng.uniform(0, config.bed_idle_hours[1])),
                "admit",
                {"bed_id": bed_id},
            )

    def free_bed_later(bed_id: str, now: pd.Timestamp) -> None:
        occupant[bed_id] = None
        push(
            now + pd.Timedelta(hours=rng.uniform(*config.bed_idle_hours)),
            "admit",
            {"bed_id": bed_id},
        )

    while events:
        now, _, kind, payload = heapq.heappop(events)
        now = now.floor("min")
        if kind == "discharge":
            enc = encounters[payload["encounter_id"]]
            enc["discharge_ts"] = now
            in_hospital.discard(enc["patient_id"])
            free_devices[enc["site_id"]].append(enc["device_id"])
            assignments[enc["encounter_id"]]["end_ts"] = now
            free_bed_later(enc["bed_id"], now)

        elif kind == "admit":
            bed_id = payload["bed_id"]
            if occupant[bed_id] is not None:
                continue
            unit_idx = UNIT_TYPES.index(_unit_type(bed_unit[bed_id]))
            p = weights * preference[:, unit_idx]
            busy = [profile_row[pid] for pid in in_hospital if pid in profile_row]
            p[busy] = 0
            patient_id = profiles.loc[rng.choice(len(p), p=p / p.sum()), "patient_id"]
            device_id = free_devices[bed_site[bed_id]].pop()
            outbreak = config.outbreak_from is not None and now >= config.outbreak_from
            reason = "pneumonia" if outbreak and rng.random() < config.outbreak_share else None
            open_stay(patient_id, bed_id, now, device_id, (now + los()).floor("min"), reason)

        elif kind == "transfer":
            enc = encounters[payload["encounter_id"]]
            if pd.notna(enc["discharge_ts"]):
                continue
            current = _unit_type(enc["unit_id"])
            targets = [
                b
                for b, who in occupant.items()
                if who is None
                and bed_site[b] == enc["site_id"]
                and _unit_type(bed_unit[b]) != current
            ]
            if not targets:
                continue
            new_bed = targets[rng.integers(len(targets))]
            transfers.append(
                {
                    "encounter_id": enc["encounter_id"],
                    "transfer_ts": now,
                    "from_unit_id": enc["unit_id"],
                    "to_unit_id": bed_unit[new_bed],
                    "from_bed_id": enc["bed_id"],
                    "to_bed_id": new_bed,
                }
            )
            old_bed = enc["bed_id"]
            enc["unit_id"], enc["bed_id"] = bed_unit[new_bed], new_bed
            occupant[new_bed] = enc["encounter_id"]
            free_bed_later(old_bed, now)

    encounters_df = _utc(
        pd.DataFrame(encounters.values()), "admit_ts", "planned_discharge_ts", "discharge_ts"
    )
    admitted = encounters_df["patient_id"].unique()
    return HospitalActivity(
        encounters=encounters_df,
        device_assignments=_utc(pd.DataFrame(assignments.values()), "start_ts", "end_ts"),
        transfers=_utc(pd.DataFrame(transfers, columns=_TRANSFER_COLUMNS), "transfer_ts"),
        patients=profiles[profiles["patient_id"].isin(admitted)].reset_index(drop=True),
        start=start,
        end=end,
    )


def _utc(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Make timestamp columns tz-aware UTC datetimes, even when every value is missing."""
    for c in columns:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


_TRANSFER_COLUMNS = [
    "encounter_id",
    "transfer_ts",
    "from_unit_id",
    "to_unit_id",
    "from_bed_id",
    "to_bed_id",
]


def census(activity: HospitalActivity, freq: str = "1h") -> pd.Series:
    """Number of patients in hospital over time."""
    times = pd.date_range(activity.start, activity.end, freq=freq, inclusive="left")
    enc = activity.encounters
    discharge = enc["discharge_ts"].fillna(activity.end)
    return pd.Series(
        [int(((enc["admit_ts"] <= t) & (discharge > t)).sum()) for t in times],
        index=times,
        name="patients",
    )


def sql_snapshot(activity: HospitalActivity, as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """The encounters and device_assignments tables as they'd look in Azure SQL at `as_of`.

    Pass the *actual* records (`WardResult.activity`, escalations applied) rather than the plan.

    `last_updated` is the time of the row's latest change (admission, transfer or discharge).
    Lakeflow Connect's query-based connector picks up rows with last_updated > its cursor.
    """
    enc = activity.encounters[activity.encounters["admit_ts"] <= as_of].copy()
    discharged = enc["discharge_ts"].notna() & (enc["discharge_ts"] <= as_of)
    enc.loc[~discharged, "discharge_ts"] = pd.NaT
    enc["status"] = np.where(discharged, "discharged", "admitted")

    # Unit and bed as of `as_of`: replay transfers that have happened by then.
    enc["unit_id"], enc["bed_id"] = enc["admit_unit_id"], enc["admit_bed_id"]
    enc["last_updated"] = enc["admit_ts"]
    moves = activity.transfers[activity.transfers["transfer_ts"] <= as_of]
    moves = moves.sort_values("transfer_ts").groupby("encounter_id").last()
    moved = enc["encounter_id"].isin(moves.index)
    enc.loc[moved, "unit_id"] = enc.loc[moved, "encounter_id"].map(moves["to_unit_id"])
    enc.loc[moved, "bed_id"] = enc.loc[moved, "encounter_id"].map(moves["to_bed_id"])
    enc.loc[moved, "last_updated"] = enc.loc[moved, "encounter_id"].map(moves["transfer_ts"])
    enc.loc[discharged, "last_updated"] = enc.loc[discharged, "discharge_ts"]

    dev = activity.device_assignments[activity.device_assignments["start_ts"] <= as_of].copy()
    ended = dev["end_ts"].notna() & (dev["end_ts"] <= as_of)
    dev.loc[~ended, "end_ts"] = pd.NaT
    dev["last_updated"] = dev["end_ts"].where(ended, dev["start_ts"])

    columns = [
        "encounter_id",
        "patient_id",
        "site_id",
        "unit_id",
        "bed_id",
        "admit_ts",
        "discharge_ts",
        "status",
        "last_updated",
    ]
    return {
        "encounters": enc[columns].reset_index(drop=True),
        "device_assignments": dev.reset_index(drop=True),
    }


def changes_between(
    activity: HospitalActivity, previous: pd.Timestamp, now: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Rows an incremental ingestion run at `now` would fetch, given its last run at `previous`."""
    snapshot = sql_snapshot(activity, now)
    return {
        name: table[table["last_updated"] > previous].reset_index(drop=True)
        for name, table in snapshot.items()
    }
