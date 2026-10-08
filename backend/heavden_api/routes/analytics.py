"""KPIs and chart series for the Analytics tile (drawn with Recharts in the web app)."""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from heavden.ml import evaluate
from heavden_api import queries
from heavden_api.deps import AppState, state
from heavden_api.snapshot import to_records

router = APIRouter(prefix="/analytics", tags=["analytics"])

SiteId = Literal["SITE_A", "SITE_B", "SITE_C"]


# Plain-language definitions the app shows next to each number (alert vs hour matters).
DEFINITIONS = {
    "census": "Patients on the ward now.",
    "high_risk": "Patients in the High band now (risk at or above the alert threshold).",
    "alerts_24h": "Alerts in the last 24 hours. An alert fires when a patient enters the High "
    "band; the same patient doesn't alert again for 6 hours.",
    "alerts_per_nurse_shift_24h": "Alerts per nurse per 12-hour shift (1 nurse per 5 patients). "
    "The budget is about 2.",
    "alert_precision_7d": "Per alert: share of last-7-day alerts followed by an escalation "
    "within 6 hours.",
    "escalations_flagged_7d": "Per escalation: share of last-7-day escalations where the "
    "patient was in the High band at some point in the 6 hours before.",
    "median_hours_flagged_before_7d": "How many hours before an escalation the patient was first "
    "flagged High (median, flagged escalations only).",
}


@router.get("/summary")
def summary(site_id: SiteId | None = None, s: AppState = Depends(state)) -> dict:
    """Headline KPIs now, with `definitions` for each. Alert metrics count **alerts** (a
    patient entering High), not hours spent in High."""
    budget = evaluate.AlertBudget()
    params = {
        "as_of": s.as_of,
        "site_id": site_id,
        "patients_per_nurse": budget.patients_per_nurse,
        "shift_hours": budget.shift_hours,
    }
    row = to_records(s.source.query(queries.SUMMARY, params))[0]
    return {
        "as_of": s.as_of.isoformat(),
        "site_id": site_id,
        **row,
        "alert_budget_per_nurse_shift": budget.alerts_per_nurse_per_shift,
        "definitions": DEFINITIONS,
    }


@router.get("/census")
def census(
    site_id: SiteId | None = None,
    hours: int = Query(72, ge=1, le=24 * 14),
    s: AppState = Depends(state),
) -> list[dict]:
    """Hourly census, risk bands, alerts and escalations per site."""
    params = {"as_of": s.as_of, "site_id": site_id, "hours": hours}
    return to_records(s.source.query(queries.CENSUS_HOURLY, params))


@router.get("/units")
def units(site_id: SiteId | None = None, s: AppState = Depends(state)) -> list[dict]:
    """Patients per risk band on each unit right now."""
    return to_records(s.source.query(queries.UNITS_NOW, {"as_of": s.as_of, "site_id": site_id}))


@router.get("/alerts")
def alerts(
    site_id: SiteId | None = None,
    days: int = Query(7, ge=1, le=14),
    s: AppState = Depends(state),
) -> list[dict]:
    """Alerts per day and site, split by outcome (escalated, not escalated, still pending)."""
    params = {"as_of": s.as_of, "site_id": site_id, "days": days}
    return to_records(s.source.query(queries.ALERTS_DAILY, params))


@router.get("/devices")
def devices(
    site_id: SiteId | None = None,
    days: int = Query(7, ge=1, le=14),
    s: AppState = Depends(state),
) -> list[dict]:
    """Device health per day and site: uptime, outages, stuck sensors, SpO2, firmware."""
    params = {"as_of": s.as_of, "site_id": site_id, "days": days}
    return to_records(s.source.query(queries.DEVICES_DAILY, params))
