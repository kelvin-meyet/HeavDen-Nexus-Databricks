"""Shared application state and the dependency that hands it to routes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from fastapi import Request

from heavden.agent.genie import find_space
from heavden.agent.retrieval import Embedder, FastEmbedder, Retriever
from heavden_api.chat.agent import LLM, OpenAILLM, system_prompt
from heavden_api.chat.databricks_tools import DatabricksToolbox, Genie, Warehouse
from heavden_api.chat.guardrails import LeakDetector
from heavden_api.chat.limits import DailyBudget, RateLimiter, TableBudget
from heavden_api.chat.memory import ConversationStore
from heavden_api.chat.models import profile
from heavden_api.chat.recorded import Recordings
from heavden_api.chat.sandbox import SqlSandbox
from heavden_api.chat.tools import Toolbox
from heavden_api.config import Settings, databricks_api
from heavden_api.snapshot import Snapshot, fetch_snapshot
from heavden_api.sources import DataSource, DemoSource, LiveSource


@dataclass(frozen=True)
class LiveTools:
    """Live mode's Databricks side of the chat: prod's Genie space and its patient function."""

    catalog: str
    warehouse: Warehouse
    genie: Genie


@dataclass
class AppState:
    settings: Settings
    snapshot: Snapshot
    source: DataSource
    sandbox: SqlSandbox | None  # demo: the assistant's locked-down SQL engine
    recordings: Recordings
    rate_limiter: RateLimiter
    live_budget: DailyBudget | TableBudget
    leak_detector: LeakDetector
    conversations: ConversationStore
    llm: LLM | None = None  # None: /chat replays recordings
    embedder: Embedder | None = None  # loaded on the first document search
    live: LiveTools | None = None  # live mode only
    _retriever: Retriever | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def as_of(self) -> pd.Timestamp:
        """ "Now": the snapshot's hour, or prod's latest scored hour in live mode."""
        return self.source.now()[0]

    @property
    def data_start(self) -> pd.Timestamp:
        return self.source.now()[1]

    def toolbox(self) -> Toolbox:
        """The chat's tools for one turn: in live mode numbers come from Genie and patients from
        the UC function, both on prod; documents are the snapshot's own index in both modes."""
        if self.live is None:
            return Toolbox(self, self.sandbox)
        live = self.live
        return DatabricksToolbox(live.catalog, live.warehouse, live.genie, self.retriever())

    def system_prompt(self) -> str:
        canary = self.leak_detector.canary
        start = self.data_start.isoformat()
        if self.live is None:
            m = self.snapshot.manifest
            return system_prompt(m.as_of, m.bands, self.sandbox.schema_text(), start, canary)
        return system_prompt(
            self.as_of.isoformat(), None, data_start=start, canary=canary, data_tool="genie"
        )

    @property
    def documents_dir(self) -> Path | None:
        """The document index: the image's own (RAG_DIR), else the snapshot's (the demo)."""
        return self.settings.rag_dir or self.snapshot.rag_dir

    def retriever(self) -> Retriever | None:
        """The document index, loaded on first use (loading the embedding model takes seconds)."""
        folder = self.documents_dir
        if folder is None:
            return None
        with self._lock:
            if self._retriever is None:
                embedder = self.embedder or FastEmbedder(cache_dir=self.settings.embed_cache_dir)
                self._retriever = Retriever.load(folder, embedder)
        return self._retriever


def connect_live(settings: Settings, api) -> tuple[LiveSource, LiveTools]:
    """Prod's Gold on the SQL warehouse, and the chat's Databricks tools, through `api`."""
    space = find_space(api, settings.genie_title)
    if space is None:
        raise RuntimeError(f"MODE=live: no Genie space titled {settings.genie_title!r}")
    warehouse = Warehouse(api, settings.warehouse_id, catalog=settings.live_catalog)
    # 180 s: a cold Genie question has taken ~120 s
    tools = LiveTools(settings.live_catalog, warehouse, Genie(api, space, timeout=180.0))
    return LiveSource(warehouse), tools


def build_state(
    settings: Settings,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    api=None,
    read_snapshot: Callable[[str], bytes] | None = None,
) -> AppState:
    """`api`: the Databricks API client for live mode; `read_snapshot`: reads the published
    snapshot's files (with SNAPSHOT_URL). Tests pass fakes; by default both use the website's
    managed identity (Databricks: or the identity in the environment)."""
    folder = settings.snapshot_dir
    if settings.snapshot_url:  # Blob Storage: starting never needs Databricks
        read = read_snapshot or settings.snapshot_reader()
        folder = fetch_snapshot(read, settings.snapshot_dir)
        print(f"snapshot {folder.name} from {settings.snapshot_url}", flush=True)
    snapshot = Snapshot.load(folder)
    if settings.mode == "live":
        source, live = connect_live(settings, api or databricks_api(settings.managed_identity))
        sandbox = None
    else:
        source, live, sandbox = DemoSource(snapshot), None, SqlSandbox(snapshot)
    connection = settings.llm_connection()
    if llm is None and connection:
        model = profile(settings.llm_model, settings.llm_provider)  # refuses unlisted models
        effort = settings.llm_reasoning_effort
        llm = OpenAILLM(
            connection.api_key,
            model.name,
            model.reasoning_effort if effort is None else (effort.strip() or None),
            connection.base_url,
            model.token_limit,
            **({"max_tokens": model.max_tokens} if model.max_tokens else {}),
        )
    return AppState(
        settings=settings,
        snapshot=snapshot,
        source=source,
        sandbox=sandbox,
        recordings=Recordings.load(),
        rate_limiter=RateLimiter(settings.chat_requests_per_hour),
        live_budget=settings.chat_budget(),
        leak_detector=LeakDetector.create(),
        conversations=ConversationStore(),
        llm=llm,
        embedder=embedder,
        live=live,
    )


def state(request: Request) -> AppState:
    return request.app.state.heavden
