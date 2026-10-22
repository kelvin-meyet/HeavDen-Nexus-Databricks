from fastapi import APIRouter, Depends

from heavden_api.deps import AppState, state

router = APIRouter(tags=["health"])


@router.get("/health")
def health(s: AppState = Depends(state)) -> dict:
    """Liveness plus what is being served. Every page calls this (its header shows "now"), so it
    never wakes the SQL warehouse: in live mode "now" is prod's last answer, or the snapshot's
    hour before any page has asked prod. The row counts, which describe the snapshot's tables,
    are left out in live mode. `live_data`: the pages read prod live right now (false in demo
    mode, and in live mode once the day's warehouse allowance is used up)."""
    m = s.snapshot.manifest
    as_of, data_start = s.known_now()
    return {
        "status": "ok",
        "mode": s.settings.mode,
        "live_data": s.live_data(),
        "synthetic": m.synthetic,
        "as_of": as_of.isoformat(),
        "data_start": data_start.isoformat(),
        "model": {"name": m.model_name, "version": m.model_version, "type": m.model_type},
        "risk_bands": m.bands,
        "documents": s.documents_dir is not None,
        "rows": m.rows if s.live is None else {},
        "snapshot": s.snapshot.root.name,  # the published snapshot's name on Container Apps
    }
