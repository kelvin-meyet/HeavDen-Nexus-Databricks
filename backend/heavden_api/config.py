"""Settings from environment variables (Render sets these; locally the defaults work)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from heavden_api.chat.models import PROVIDERS as LLM_PROVIDERS

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Local development only: read KEY=value lines from `.env` (git-ignored) without
    overriding variables that are already set. On Render, variables come from the dashboard."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _origins() -> tuple[str, ...]:
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    return tuple(o.strip() for o in raw.split(",") if o.strip())


def azure_base_url(endpoint: str) -> str:
    """The OpenAI-compatible (v1) address of an Azure OpenAI / Foundry resource. Only the host
    of `endpoint` is used, so whatever the portal shows can be pasted as it is (with or without
    a path, a deployment or an api-version)."""
    host = urlsplit(endpoint if "://" in endpoint else f"https://{endpoint}").netloc
    return f"https://{host}/openai/v1/"


def databricks_token() -> str:
    """A current token for the Databricks identity in the environment: a service principal
    (DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET) on Render, a CLI profile
    (DATABRICKS_CONFIG_PROFILE) locally. The SDK refreshes it before it expires."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient().config.authenticate()["Authorization"].removeprefix("Bearer ")


@dataclass(frozen=True)
class LLMConnection:
    api_key: str | Callable[[], str] = field(repr=False)
    base_url: str | None  # None: the OpenAI API


@dataclass(frozen=True)
class Settings:
    # demo: snapshot files (DuckDB, FAISS, local model). live: Databricks (Phase 5).
    mode: str = field(default_factory=lambda: os.environ.get("MODE", "demo"))
    snapshot_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SNAPSHOT_DIR", ROOT / "data" / "demo_snapshot")
        )
    )
    # where fastembed caches the embedding model (~130 MB, downloaded on first search)
    embed_cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("EMBED_CACHE_DIR", ROOT / "data" / "models"))
    )
    # browsers calling the API directly (local dev). In production Vercel proxies /api/*.
    allowed_origins: tuple[str, ...] = field(default_factory=_origins)
    # below this best-match similarity, document search reports "not covered"
    min_similarity: float = 0.55
    # the assistant: without a key, /chat replays recorded answers
    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None, repr=False
    )
    # which service answers: "openai" (OPENAI_API_KEY), "azure" (Azure OpenAI in Foundry:
    # AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY) or "databricks" (a serving endpoint in the
    # workspace at DATABRICKS_HOST, called with a Databricks identity). All speak the same API.
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    )
    azure_openai_endpoint: str | None = field(
        default_factory=lambda: os.environ.get("AZURE_OPENAI_ENDPOINT") or None
    )
    azure_openai_key: str | None = field(
        default_factory=lambda: (
            os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY") or None
        ),
        repr=False,
    )
    databricks_host: str | None = field(
        default_factory=lambda: os.environ.get("DATABRICKS_HOST") or None
    )
    # OpenAI: the model name. Azure: the name of YOUR deployment. Databricks: the serving
    # endpoint. It must be listed in chat/models.yaml, with its settings.
    # gpt-5.5 answered judgement questions reliably in testing; gpt-5.4-mini is cheaper but weaker
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-5.5"))
    # overrides the model's reasoning_effort from models.yaml when set (empty: leave it out)
    llm_reasoning_effort: str | None = field(
        default_factory=lambda: os.environ.get("LLM_REASONING_EFFORT")
    )
    chat_requests_per_hour: int = field(
        default_factory=lambda: int(os.environ.get("CHAT_REQUESTS_PER_HOUR", "30"))
    )
    # live (LLM) answers per UTC day across all visitors; past it, recordings are replayed
    chat_live_answers_per_day: int = field(
        default_factory=lambda: int(os.environ.get("CHAT_LIVE_ANSWERS_PER_DAY", "300"))
    )
    # proxies we control in front of the API: 0 local, 1 Render, 2 Vercel -> Render.
    # The visitor's address is that many entries from the right of X-Forwarded-For.
    trusted_proxy_hops: int = field(
        default_factory=lambda: int(os.environ.get("TRUSTED_PROXY_HOPS", "0"))
    )

    def __post_init__(self):
        if self.mode not in ("demo", "live"):
            raise ValueError(f"MODE must be 'demo' or 'live', not {self.mode!r}")

    def llm_connection(self) -> LLMConnection | None:
        """Where the assistant's LLM calls go, or None when the chosen provider has no key (the
        assistant then replays recorded answers)."""
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {LLM_PROVIDERS}, not {self.llm_provider!r}"
            )
        if self.llm_provider == "openai":
            return LLMConnection(self.openai_api_key, None) if self.openai_api_key else None
        if self.llm_provider == "databricks":
            if not self.databricks_host:
                raise ValueError("LLM_PROVIDER=databricks needs DATABRICKS_HOST")
            host = self.databricks_host.rstrip("/")
            host = host if "://" in host else f"https://{host}"
            return LLMConnection(databricks_token, f"{host}/serving-endpoints")
        if not self.azure_openai_key:
            return None
        if not self.azure_openai_endpoint:
            raise ValueError("LLM_PROVIDER=azure needs AZURE_OPENAI_ENDPOINT as well as the key")
        return LLMConnection(self.azure_openai_key, azure_base_url(self.azure_openai_endpoint))
