from fastapi import APIRouter, Depends

from heavden_api.deps import AppState, state

router = APIRouter(tags=["health"])


@router.get("/health")
def health(s: AppState = Depends(state)) -> dict:
    """Liveness plus what is being served. The web app calls this on page load to wake Render."""
    m = s.snapshot.manifest
    return {
        "status": "ok",
        "mode": s.settings.mode,
        "synthetic": m.synthetic,
        "as_of": m.as_of,
        "data_start": m.data_start,
        "model": {"name": m.model_name, "version": m.model_version, "type": m.model_type},
        "risk_bands": m.bands,
        "documents": s.snapshot.rag_dir is not None,
        "rows": m.rows,
    }
