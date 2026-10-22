"""The HeavDen-Nexus API (FastAPI). Run locally:

    uv run uvicorn heavden_api.app:app --reload        # http://127.0.0.1:8000/docs

`MODE=demo` (default) serves the snapshot in `SNAPSHOT_DIR`. `MODE=live` reads prod on Databricks
(see `config.Settings`).
"""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from heavden.agent.retrieval import Embedder
from heavden_api.chat.agent import LLM
from heavden_api.config import ORIGIN_HEADER, Settings, load_dotenv
from heavden_api.deps import build_state
from heavden_api.routes import analytics, chat, documents, health, risk


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    api=None,
) -> FastAPI:
    """`api`: live mode's Databricks API client (tests pass a fake; by default the identity in
    the environment)."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.heavden = build_state(settings, embedder, llm, api)
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
    if settings.origin_secret:
        secret = settings.origin_secret.encode()

        @app.middleware("http")
        async def only_through_the_website(request: Request, call_next):
            """Refuse requests without the website's secret: the API's own URL is public, and
            callers who skip the website would skip its visitor address and its cache."""
            sent = request.headers.get(ORIGIN_HEADER, "").encode()
            if not hmac.compare_digest(sent, secret):
                return JSONResponse({"detail": "This API answers only the website."}, 403)
            return await call_next(request)

    for module in (health, analytics, risk, documents, chat):
        app.include_router(module.router)
    return app


load_dotenv()  # local development: OPENAI_API_KEY etc. from .env
app = create_app()
