"""The HeavDen-Nexus API (FastAPI). Run locally:

    uv run uvicorn heavden_api.app:app --reload        # http://127.0.0.1:8000/docs

`MODE=demo` (default) serves the snapshot in `SNAPSHOT_DIR`. `MODE=live` arrives in Phase 5.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from heavden.agent.retrieval import Embedder
from heavden_api.config import Settings
from heavden_api.deps import build_state
from heavden_api.routes import analytics, documents, health, risk


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.heavden = build_state(settings, embedder)
        yield

    app = FastAPI(
        title="HeavDen-Nexus API",
        description="Synthetic data only. Deterioration risk, analytics and document search.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    for module in (health, analytics, risk, documents):
        app.include_router(module.router)
    return app


app = create_app()
