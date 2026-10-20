from fastapi import APIRouter, Depends

from heavden_api.deps import AppState, state

router = APIRouter(tags=["health"])


@router.get("/health")
def health(s: AppState = Depends(state)) -> dict:
    """Liveness plus what is being served. The web app calls this on page load to wake Render.
    In live mode "now" comes from prod (the first call may wake the SQL warehouse) and the row
    counts, which describe the snapshot's tables, are left out."""
    m = s.snapshot.manifest
    return {
        "status": "ok",
        "mode": s.settings.mode,
        "synthetic": m.synthetic,
        "as_of": s.as_of.isoformat(),
        "data_start": s.data_start.isoformat(),
        "model": {"name": m.model_name, "version": m.model_version, "type": m.model_type},
        "risk_bands": m.bands,
        "documents": s.documents_dir is not None,
        "rows": m.rows if s.live is None else {},
        "snapshot": s.snapshot.root.name,  # the published snapshot's name on Container Apps
    }
