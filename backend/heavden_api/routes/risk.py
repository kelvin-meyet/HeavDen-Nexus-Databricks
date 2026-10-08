"""The Patient Risk tile: ward board, patient detail and the what-if scorer."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from heavden.ml import scoring
from heavden_api import queries
from heavden_api.deps import AppState, state
from heavden_api.routes.analytics import SiteId
from heavden_api.snapshot import to_records

router = APIRouter(prefix="/risk", tags=["risk"])

Band = Literal["Low", "High"]


def _with_factors(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["top_factors"] = json.loads(row["top_factors"]) if row.get("top_factors") else []
    return rows


@router.get("/board")
def board(
    site_id: SiteId | None = None,
    unit_id: str | None = None,
    band: Band | None = None,
    limit: int = Query(50, ge=1, le=400),
    s: AppState = Depends(state),
) -> dict:
    """Patients on the ward now, highest risk first, with the change over the last 6 hours."""
    params = {"as_of": s.as_of, "site_id": site_id, "unit_id": unit_id, "band": band}
    rows = _with_factors(to_records(s.source.query(queries.BOARD, params | {"limit": limit})))
    return {"as_of": s.as_of.isoformat(), "bands": s.snapshot.manifest.bands, "patients": rows}


@router.get("/patients/{encounter_id}")
def patient(
    encounter_id: str,
    hours: int = Query(24, ge=1, le=24 * 14),
    s: AppState = Depends(state),
) -> dict:
    """One stay: who and where, current risk and reasons, hourly history and past alerts."""
    params = {"as_of": s.as_of, "encounter_id": encounter_id}
    found = to_records(s.source.query(queries.PATIENT, params))
    if not found:
        raise HTTPException(404, f"unknown encounter {encounter_id}")
    header = _with_factors(found)[0]
    header.pop("patient_id", None)  # internal id; the app shows patient_label
    series = to_records(s.source.query(queries.PATIENT_SERIES, params | {"hours": hours}))
    alerts = to_records(s.source.query(queries.PATIENT_ALERTS, params))
    return {"as_of": s.as_of.isoformat(), "patient": header, "series": series, "alerts": alerts}


class WhatIfChanges(BaseModel):
    """New values for the latest hour. Leave a field out to keep the measured value."""

    heart_rate: float | None = Field(None, ge=20, le=250)
    resp_rate: float | None = Field(None, ge=4, le=60)
    spo2: float | None = Field(None, ge=50, le=100)
    temp_c: float | None = Field(None, ge=32, le=43)
    sbp: float | None = Field(None, ge=50, le=250)
    dbp: float | None = Field(None, ge=25, le=150)
    on_oxygen: bool | None = None
    acvpu: Literal["A", "C", "V", "P", "U"] | None = None


class WhatIfRequest(BaseModel):
    encounter_id: str
    changes: WhatIfChanges


@router.post("/score")
def score(body: WhatIfRequest, s: AppState = Depends(state)) -> dict:
    """What-if: re-score a patient's latest hour with some vitals or observations changed."""
    params = {"as_of": s.as_of, "encounter_id": body.encounter_id}
    rows = s.source.query(queries.FEATURE_ROW, params)
    if rows.empty:
        raise HTTPException(404, f"{body.encounter_id} is not on the ward at {s.as_of}")
    changes = body.changes.model_dump(exclude_none=True)
    result = scoring.what_if(
        s.snapshot.model,
        rows.iloc[0],
        changes,
        s.snapshot.manifest.risk_bands,
        s.snapshot.background,
    )
    return {"encounter_id": body.encounter_id, "changes": changes, **result}
