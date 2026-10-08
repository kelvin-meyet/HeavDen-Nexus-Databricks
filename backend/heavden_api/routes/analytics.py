"""KPIs and chart series for the Analytics tile (drawn with Recharts in the web app)."""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from heavden_api import queries
from heavden_api.deps import AppState, state
from heavden_api.snapshot import to_records

router = APIRouter(prefix="/analytics", tags=["analytics"])

SiteId = Literal["SITE_A", "SITE_B", "SITE_C"]


@router.get("/summary")
def summary(site_id: SiteId | None = None, s: AppState = Depends(state)) -> dict:
    """Headline KPIs now: census, patients per band, last-24 h alerts and escalations, and the
    share of last-7-day alerts followed by an escalation (alert precision)."""
    row = to_records(s.source.query(queries.SUMMARY, {"as_of": s.as_of, "site_id": site_id}))[0]
    return {"as_of": s.as_of.isoformat(), "site_id": site_id, **row}


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
