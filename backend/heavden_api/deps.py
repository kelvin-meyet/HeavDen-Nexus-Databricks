"""Shared application state and the dependency that hands it to routes."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from fastapi import Request

from heavden.agent.retrieval import Embedder, FastEmbedder, Retriever
from heavden_api.config import Settings
from heavden_api.snapshot import Snapshot
from heavden_api.sources import DataSource, DemoSource


@dataclass
class AppState:
    settings: Settings
    snapshot: Snapshot
    source: DataSource
    embedder: Embedder | None = None  # loaded on the first document search
    _retriever: Retriever | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def as_of(self):
        return self.snapshot.manifest.as_of_ts

    def retriever(self) -> Retriever | None:
        """The document index, built on first use (loading the embedding model takes seconds)."""
        if self.snapshot.rag_dir is None:
            return None
        with self._lock:
            if self._retriever is None:
                embedder = self.embedder or FastEmbedder(cache_dir=self.settings.embed_cache_dir)
                self._retriever = Retriever.load(self.snapshot.rag_dir, embedder)
        return self._retriever


def build_state(settings: Settings, embedder: Embedder | None = None) -> AppState:
    if settings.mode == "live":
        raise RuntimeError(
            "MODE=live is not implemented yet (Phase 5: Databricks SQL warehouse, model and "
            "agent endpoints). Use MODE=demo."
        )
    snapshot = Snapshot.load(settings.snapshot_dir)
    return AppState(settings, snapshot, DemoSource(snapshot), embedder)


def state(request: Request) -> AppState:
    return request.app.state.heavden
