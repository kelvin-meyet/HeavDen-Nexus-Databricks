"""Patient-hour features and 6-hour labels (Plan.md §7.3 gold tables, ml_model.md §6-§7).

This is the pandas reference implementation, used locally, in tests and in demo mode. The
Databricks gold pipeline must produce the same columns with the same meaning.

The point-in-time rules:

* A **prediction time** `t` is on the hour, while the patient is on the ward (admitted <= t < left).
* Features at `t` use only readings with ts < t, grouped into hourly **buckets** [b, b + 1h).
  The bucket ending at t is "the last hour".
* The **label** at `t` is 1 if an escalation happens in (t, t + 6h]. It is only known at
  `t + 6h`; if that's after the end of the data, the label is missing (not 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heavden.ml import news2

VITALS = ("heart_rate", "resp_rate", "spo2", "temp_c", "sbp", "dbp")
HORIZON = pd.Timedelta(hours=6)
READINGS_PER_HOUR = 12
PROFILE_FLAGS = (
    "copd",
    "heart_failure",
    "diabetes",
    "ckd",
    "hypertension",
    "atrial_fibrillation",
    "on_beta_blocker",
)


def _ns(series: pd.Series) -> pd.Series:
    """Timestamps as UTC nanoseconds, so time-based joins line up whatever their source."""
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def link_readings(readings: pd.DataFrame, device_assignments: pd.DataFrame) -> pd.DataFrame:
    """Attach encounter_id and patient_id to each reading via device + time window.

    Devices are reused, so joining on device_id alone would mix patients up: a reading belongs
    to the assignment with start_ts <= ts < end_ts on that device. Readings outside any
    assignment are dropped.
    """
    left = readings.assign(ts=_ns(readings["ts"])).sort_values("ts")
    right = device_assignments.assign(
        start_ts=_ns(device_assignments["start_ts"]), end_ts=_ns(device_assignments["end_ts"])
    ).sort_values("start_ts")[["device_id", "start_ts", "end_ts", "encounter_id", "patient_id"]]
    linked = pd.merge_asof(
        left, right, left_on="ts", right_on="start_ts", by="device_id", direction="backward"
    )
    inside = linked["encounter_id"].notna() & (
        linked["end_ts"].isna() | (linked["ts"] < linked["end_ts"])
    )
    return linked[inside].drop(columns=["start_ts", "end_ts"]).reset_index(drop=True)


def hourly_buckets(linked: pd.DataFrame) -> pd.DataFrame:
    """Per encounter and hour bucket [b, b+1h): mean, median, min, max and count per vital."""
    df = linked.assign(bucket=linked["ts"].dt.floor("h"))
    agg = df.groupby(["encounter_id", "bucket"])[list(VITALS)].agg(["mean", "median", "min", "max"])
    agg.columns = [f"{v}_{stat}" for v, stat in agg.columns]
    agg["n_readings"] = df.groupby(["encounter_id", "bucket"]).size()
    return agg.reset_index()


def prediction_grid(encounters: pd.DataFrame, data_start, data_end) -> pd.DataFrame:
    """One row per encounter per hour it is on the ward, within [data_start, data_end)."""
    rows = []
    for e in encounters.itertuples():
        first = max(e.admit_ts, data_start).ceil("h")
        if first == max(e.admit_ts, data_start):  # need at least one bucket before t
            first += pd.Timedelta(hours=1)
        left = e.discharge_ts if pd.notna(e.discharge_ts) else data_end
        last = min(left, data_end)
        times = pd.date_range(first, last, freq="h", inclusive="left")
        times = times[times < last]  # date_range returns `first` when first == last
        rows.append(pd.DataFrame({"encounter_id": e.encounter_id, "prediction_ts": times}))
    return pd.concat(rows, ignore_index=True)


def _window_features(buckets: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Rolling 1/3/6 h features, aligned so the row for time t uses buckets < t only.

    Works like a SQL window over (encounter_id ORDER BY bucket): every encounter gets a complete
    hourly run of buckets (empty hours included), rolling sums run per encounter, and the row
    for bucket b becomes the feature row for prediction time b + 1h.
    """
    span = grid.groupby("encounter_id")["prediction_ts"].agg(["min", "max"])
    full = pd.DataFrame(
        {
            "encounter_id": span.index.repeat(
                ((span["max"] - span["min"]) / pd.Timedelta(hours=1)).astype(int) + 7
            )
        }
    )
    offset = full.groupby("encounter_id").cumcount()
    full["bucket"] = (
        full["encounter_id"].map(span["min"])
        - pd.Timedelta(hours=7)
        + pd.to_timedelta(offset + 1, unit="h")
    )
    b = full.merge(
        buckets.assign(bucket=_ns(buckets["bucket"])),
        on=["encounter_id", "bucket"],
        how="left",
    ).assign(bucket=lambda d: _ns(d["bucket"]))
    b = b.sort_values(["encounter_id", "bucket"], ignore_index=True)
    g = b.groupby("encounter_id", sort=False)
    n = b["n_readings"].fillna(0)

    def rolling_sum(series: pd.Series, hours: int) -> pd.Series:
        rolled = series.groupby(b["encounter_id"], sort=False).rolling(hours, min_periods=1).sum()
        return rolled.reset_index(level=0, drop=True).sort_index()

    f = pd.DataFrame(
        {"encounter_id": b["encounter_id"], "prediction_ts": b["bucket"] + pd.Timedelta(hours=1)}
    )
    n_3h, n_6h = rolling_sum(n, 3), rolling_sum(n, 6)
    for v in VITALS:
        mean = b[f"{v}_mean"]
        total = (mean * n).fillna(0)
        f[f"{v}_mean_1h"] = mean
        f[f"{v}_median_1h"] = b[f"{v}_median"]
        f[f"{v}_min_1h"] = b[f"{v}_min"]
        f[f"{v}_max_1h"] = b[f"{v}_max"]
        f[f"{v}_mean_3h"] = rolling_sum(total, 3) / n_3h.replace(0, np.nan)
        f[f"{v}_mean_6h"] = rolling_sum(total, 6) / n_6h.replace(0, np.nan)
        # change in the hourly mean over the last 3 hours (latest bucket vs 2 buckets earlier)
        f[f"{v}_trend_3h"] = mean - g[f"{v}_mean"].shift(2)
    f["readings_1h"] = n
    f["missing_1h"] = (READINGS_PER_HOUR - n).clip(lower=0)
    f["missing_6h"] = (6 * READINGS_PER_HOUR - n_6h).clip(lower=0)
    keys = grid.assign(prediction_ts=_ns(grid["prediction_ts"]))
    return keys.merge(f, on=["encounter_id", "prediction_ts"], how="left")


def _unit_at(grid: pd.DataFrame, encounters: pd.DataFrame, transfers: pd.DataFrame) -> pd.Series:
    """Unit each patient is on at the prediction time (transfers that happened before t)."""
    unit = grid["encounter_id"].map(encounters.set_index("encounter_id")["admit_unit_id"])
    if len(transfers):
        moves = (
            transfers.assign(transfer_ts=_ns(transfers["transfer_ts"]))
            .sort_values("transfer_ts")[["encounter_id", "transfer_ts", "to_unit_id"]]
            .astype({"encounter_id": object})
        )
        left = (
            grid[["encounter_id", "prediction_ts"]].reset_index().astype({"encounter_id": object})
        )
        at = pd.merge_asof(
            left.assign(prediction_ts=_ns(left["prediction_ts"])).sort_values("prediction_ts"),
            moves,
            left_on="prediction_ts",
            right_on="transfer_ts",
            by="encounter_id",
            direction="backward",
            allow_exact_matches=False,
        ).set_index("index")
        unit = at["to_unit_id"].reindex(grid.index).fillna(unit)
    return unit


def latest_observation(grid: pd.DataFrame, observations: pd.DataFrame | None) -> pd.DataFrame:
    """The last nurse observation charted strictly before each prediction time.

    No observation yet means alert and on room air (the NEWS2 default), with
    `hours_since_obs` missing.
    """
    out = pd.DataFrame(index=grid.index)
    out["acvpu"] = "A"
    out["on_oxygen"] = False
    out["o2_flow_lpm"] = 0.0
    out["hours_since_obs"] = np.nan
    if observations is not None and not observations.empty:
        obs = observations.assign(obs_ts=_ns(observations["obs_ts"])).astype(
            {"encounter_id": object}
        )
        left = (
            grid[["encounter_id", "prediction_ts"]].reset_index().astype({"encounter_id": object})
        )
        at = (
            pd.merge_asof(
                left.assign(prediction_ts=_ns(left["prediction_ts"])).sort_values("prediction_ts"),
                obs.sort_values("obs_ts")[
                    ["encounter_id", "obs_ts", "acvpu", "on_oxygen", "o2_flow_lpm"]
                ],
                left_on="prediction_ts",
                right_on="obs_ts",
                by="encounter_id",
                direction="backward",
                allow_exact_matches=False,
            )
            .set_index("index")
            .reindex(grid.index)
        )
        seen = at["obs_ts"].notna()
        out.loc[seen, "acvpu"] = at.loc[seen, "acvpu"]
        out.loc[seen, "on_oxygen"] = at.loc[seen, "on_oxygen"].astype(bool)
        out.loc[seen, "o2_flow_lpm"] = at.loc[seen, "o2_flow_lpm"]
        out["hours_since_obs"] = (at["prediction_ts"] - at["obs_ts"]) / pd.Timedelta(hours=1)
    out["on_oxygen"] = out["on_oxygen"].astype(bool)
    out["new_confusion"] = out["acvpu"] != "A"
    return out


def label(grid: pd.DataFrame, outcomes: pd.DataFrame, data_end) -> pd.DataFrame:
    """`label` = escalation in (t, t+6h]; NaN while the 6-hour window hasn't closed."""
    events = outcomes.set_index("encounter_id")["event_ts"]
    event = _ns(grid["encounter_id"].map(events))
    known_at = grid["prediction_ts"] + HORIZON
    positive = (event > grid["prediction_ts"]) & (event <= known_at)
    closed = (known_at <= data_end) | positive  # an event already seen settles the label
    return pd.DataFrame(
        {
            "label": np.where(closed, positive.astype(float), np.nan),
            "label_known_at": known_at,
        },
        index=grid.index,
    )


def build_patient_hours(
    readings: pd.DataFrame,
    activity,
    outcomes: pd.DataFrame,
    data_start=None,
    data_end=None,
    nurse_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The gold `patient_hour_features` + labels table, from readings and hospital records.

    `activity` is the *actual* hospital record (`WardResult.activity`): encounters,
    device_assignments, transfers and patient profiles. `nurse_observations` (ACVPU and
    oxygen, `WardResult.nurse_observations`) completes NEWS2; without them everyone is scored
    as alert and on room air.
    """
    data_start = data_start if data_start is not None else activity.start
    data_end = data_end if data_end is not None else activity.end
    encounters = activity.encounters

    linked = link_readings(readings, activity.device_assignments)
    buckets = hourly_buckets(linked)
    grid = prediction_grid(encounters, data_start, data_end)
    features = _window_features(buckets, grid)

    enc = encounters.set_index("encounter_id")
    features["patient_id"] = features["encounter_id"].map(enc["patient_id"])
    features["site_id"] = features["encounter_id"].map(enc["site_id"])
    features["unit_id"] = _unit_at(features, encounters, activity.transfers)
    features["unit_type"] = features["unit_id"].str.split("-").str[-1].str.lower()
    admit = features["encounter_id"].map(enc["admit_ts"])
    features["hours_since_admission"] = (features["prediction_ts"] - admit) / pd.Timedelta(hours=1)

    profile = activity.patients.set_index("patient_id")
    features["age"] = features["patient_id"].map(profile["age"])
    features["sex"] = features["patient_id"].map(profile["sex"])
    for flag in PROFILE_FLAGS:
        features[flag] = features["patient_id"].map(profile[flag]).astype(bool)
    features["n_conditions"] = features["patient_id"].map(profile["n_conditions"])

    features = pd.concat([features, latest_observation(features, nurse_observations)], axis=1)
    points = news2.news2(features, suffix="_median_1h")
    features = pd.concat([features, points, label(features, outcomes, data_end)], axis=1)
    first = ["encounter_id", "patient_id", "prediction_ts", "site_id", "unit_id", "unit_type"]
    rest = [c for c in features.columns if c not in first]
    return features[first + rest].sort_values(["prediction_ts", "encounter_id"], ignore_index=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Columns a model may use (everything except identifiers, time and label fields)."""
    excluded = {
        "acvpu",  # encoded as new_confusion
        "encounter_id",
        "patient_id",
        "prediction_ts",
        "site_id",
        "unit_id",
        "label",
        "label_known_at",
    }
    return [c for c in table.columns if c not in excluded]
