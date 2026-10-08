"""Gold analytics tables (Plan.md §7.3), pandas reference implementations.

The Spark gold pipeline must produce the same columns and meaning; the demo-mode backend serves
these tables from a snapshot. Column meanings are documented for users in
`docs/corpus/gold-data-dictionary.md`.

* `alerts_fact`: one row per High alert (a stay *entering* the High band), with re-alerts for
  the same stay suppressed for 6 hours, as ward alerting systems do to limit alarm fatigue.
  Same rule as `heavden.ml.evaluate.alert_onsets`, so model metrics and dashboards agree.
* `escalations_fact`: one row per escalation, with whether the patient was flagged High in the
  6 hours before and how early.
* `site_kpis_hourly`: census, risk bands, alerts and escalations per unit and hour.
* `device_health_daily`: completeness and fault indicators per device and day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heavden.ml.evaluate import SUPPRESS, alert_onsets

HORIZON = pd.Timedelta(hours=6)
STEP = pd.Timedelta(minutes=5)  # one VitalBand reading every 5 minutes
OUTAGE_GAP = pd.Timedelta(minutes=25)  # a gap this long between messages = an outage
STUCK_RUN = 6  # the same value 6+ readings in a row (30 min) = a frozen sensor
# Only these vary enough reading to reading for a 30-minute repeat to mean a fault: SpO2,
# temperature and breathing rate drift slowly and naturally repeat (2-12% of readings).
STUCK_VITALS = ("heart_rate", "sbp", "dbp")


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def alerts_fact(risk_scores: pd.DataFrame, outcomes: pd.DataFrame, as_of) -> pd.DataFrame:
    """High alerts: the hours where a stay enters the High band (not every High hour), unless
    the same stay already alerted within the previous 6 hours.

    `escalated_within_6h` is 1/0 once the 6-hour window has closed by `as_of` (or an escalation
    already happened), else null. `hours_to_escalation` is the warning time when escalated.
    """
    as_of = pd.Timestamp(as_of)
    alerts = risk_scores[alert_onsets(risk_scores, risk_scores["risk_band"].eq("High"), SUPPRESS)]

    event = _utc(alerts["encounter_id"].map(outcomes.set_index("encounter_id")["event_ts"]))
    alert_ts = _utc(alerts["prediction_ts"])
    escalated = (event > alert_ts) & (event <= alert_ts + HORIZON)
    known = (alert_ts + HORIZON <= as_of) | escalated
    out = pd.DataFrame(
        {
            "alert_id": alerts["encounter_id"] + "@" + alert_ts.dt.strftime("%Y%m%dT%H"),
            "encounter_id": alerts["encounter_id"],
            "patient_id": alerts["patient_id"],
            "site_id": alerts["site_id"],
            "unit_id": alerts["unit_id"],
            "alert_ts": alert_ts,
            "risk": alerts["risk"],
            "news2_total": alerts["news2_total"],
            "top_factors": alerts["top_factors"],
            "escalated_within_6h": np.where(known, escalated.astype(float), np.nan),
            "hours_to_escalation": np.where(
                escalated, (event - alert_ts) / pd.Timedelta(hours=1), np.nan
            ),
        }
    )
    return out.sort_values("alert_ts", ignore_index=True)


def escalations_fact(risk_scores: pd.DataFrame, outcomes: pd.DataFrame, as_of) -> pd.DataFrame:
    """One row per escalation up to `as_of`: where it happened, and whether the patient was in
    the High band at any hour in the 6 hours before (`flagged_6h_before`), first flagged how
    many hours ahead (`hours_flagged_before`)."""
    ev = outcomes.assign(event_ts=_utc(outcomes["event_ts"])).dropna(subset=["event_ts"])
    ev = ev[ev["event_ts"] <= pd.Timestamp(as_of)]
    r = risk_scores.assign(prediction_ts=_utc(risk_scores["prediction_ts"]))
    rows = []
    by_stay = dict(tuple(r.groupby("encounter_id")))
    for e in ev.itertuples():
        stay = by_stay.get(e.encounter_id)
        if stay is None:
            continue
        before = stay[stay["prediction_ts"] < e.event_ts]
        if before.empty:
            continue
        window = before[before["prediction_ts"] >= e.event_ts - HORIZON]
        flagged = window.loc[window["risk_band"] == "High", "prediction_ts"]
        last = before.iloc[-1]
        rows.append(
            {
                "encounter_id": e.encounter_id,
                "patient_id": last["patient_id"],
                "site_id": last["site_id"],
                "unit_id": last["unit_id"],
                "event_ts": e.event_ts,
                "event_type": e.event_type,
                "risk_before": last["risk"],
                "flagged_6h_before": not flagged.empty,
                "hours_flagged_before": (e.event_ts - flagged.min()) / pd.Timedelta(hours=1)
                if not flagged.empty
                else np.nan,
            }
        )
    columns = [
        "encounter_id", "patient_id", "site_id", "unit_id", "event_ts", "event_type",
        "risk_before", "flagged_6h_before", "hours_flagged_before",
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=columns).sort_values("event_ts", ignore_index=True)


def site_kpis_hourly(
    risk_scores: pd.DataFrame, alerts: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """One row per unit and hour: census, patients per band, new alerts, escalations, NEWS2."""
    r = risk_scores.assign(hour_ts=_utc(risk_scores["prediction_ts"]))
    keys = ["site_id", "unit_id", "hour_ts"]
    out = r.groupby(keys).agg(
        census=("encounter_id", "nunique"),
        n_low=("risk_band", lambda s: int((s == "Low").sum())),
        n_medium=("risk_band", lambda s: int((s == "Medium").sum())),
        n_high=("risk_band", lambda s: int((s == "High").sum())),
        mean_news2=("news2_total", "mean"),
    )
    raised = alerts.assign(hour_ts=_utc(alerts["alert_ts"])).groupby(keys).size()
    out["alerts_raised"] = raised.reindex(out.index, fill_value=0)

    # an escalation counts on the unit and hour where the patient last had a score before it
    ev = outcomes.assign(event_ts=_utc(outcomes["event_ts"])).dropna(subset=["event_ts"])
    ev = ev.assign(hour_ts=ev["event_ts"].dt.floor("h")).sort_values("hour_ts")
    located = pd.merge_asof(
        ev[["encounter_id", "hour_ts"]].astype({"encounter_id": object}),
        r.sort_values("hour_ts")[["encounter_id", *keys]].astype({"encounter_id": object}),
        on="hour_ts",
        by="encounter_id",
        direction="backward",
    ).dropna(subset=["unit_id"])
    out["escalations"] = located.groupby(keys).size().reindex(out.index, fill_value=0)
    return out.reset_index()


def device_health_daily(
    readings: pd.DataFrame, device_assignments: pd.DataFrame, start, end
) -> pd.DataFrame:
    """Per device and day: messages received vs expected while worn, outages, stuck values,
    battery, firmware and mean SpO2 (a site-wide drop can reveal a sensor fault)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rd = readings.assign(ts=_utc(readings["ts"])).sort_values(["device_id", "ts"])
    rd["date"] = rd["ts"].dt.floor("D")
    gap = rd.groupby("device_id")["ts"].diff()
    rd["outage"] = gap >= OUTAGE_GAP
    rd["stuck"] = False
    for v in STUCK_VITALS:
        # length of each run of identical consecutive values on the same device
        new_run = rd[v].ne(rd[v].shift()) | rd["device_id"].ne(rd["device_id"].shift())
        run_id = new_run.cumsum()
        rd["stuck"] |= run_id.map(run_id.value_counts()) >= STUCK_RUN

    daily = rd.groupby(["device_id", "date"]).agg(
        messages_received=("ts", "size"),
        battery_outages=("outage", "sum"),
        stuck_readings=("stuck", "sum"),
        min_battery_pct=("battery_pct", "min"),
        firmware=("firmware", lambda s: ",".join(sorted(set(s)))),
        mean_spo2=("spo2", "mean"),
    )
    daily["stuck_minutes"] = daily.pop("stuck_readings") * int(STEP / pd.Timedelta(minutes=1))

    expected = _expected_messages(device_assignments, start, end)
    daily = daily.join(expected, how="outer").fillna({"messages_received": 0})
    daily["messages_expected"] = daily["messages_expected"].fillna(0)
    worn = daily["messages_expected"] > 0
    daily["uptime_pct"] = np.where(
        worn,
        (100 * daily["messages_received"] / daily["messages_expected"]).clip(upper=100),
        np.nan,
    )
    daily = daily.reset_index()
    daily["site_id"] = "SITE_" + daily["device_id"].str.split("-").str[1]
    return daily[
        [
            "device_id",
            "site_id",
            "date",
            "messages_received",
            "messages_expected",
            "uptime_pct",
            "battery_outages",
            "stuck_minutes",
            "min_battery_pct",
            "firmware",
            "mean_spo2",
        ]
    ]


def _expected_messages(assignments: pd.DataFrame, start, end) -> pd.Series:
    """Readings each device should have sent per day while worn (one per 5 minutes)."""
    a = assignments.assign(
        start_ts=_utc(assignments["start_ts"]).clip(lower=start),
        end_ts=_utc(assignments["end_ts"]).fillna(end).clip(upper=end),
    )
    a = a[a["end_ts"] > a["start_ts"]]
    rows = []
    for device, s, e in a[["device_id", "start_ts", "end_ts"]].itertuples(index=False):
        day = s.floor("D")
        while day < e:
            nxt = day + pd.Timedelta(days=1)
            minutes = (min(e, nxt) - max(s, day)) / pd.Timedelta(minutes=1)
            rows.append((device, day, minutes / 5))
            day = nxt
    frame = pd.DataFrame(rows, columns=["device_id", "date", "messages_expected"])
    return frame.groupby(["device_id", "date"])["messages_expected"].sum().round()
