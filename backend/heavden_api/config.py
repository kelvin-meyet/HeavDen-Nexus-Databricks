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


# What the website's managed identity asks Entra ID tokens for: Azure Databricks (its first-party
# application id), Azure OpenAI's v1 API and Azure Storage (the snapshot)
DATABRICKS_SCOPE = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"
AZURE_OPENAI_SCOPE = "https://ai.azure.com/.default"
STORAGE_SCOPE = "https://storage.azure.com/.default"

# The header that carries Settings.origin_secret (also in web/lib/backend.ts and smoke_api.py)
ORIGIN_HEADER = "x-origin-secret"


def managed_identity_token(client_id: str, scope: str) -> Callable[[], str]:
    """A function giving a current token for `scope` as the user-assigned managed identity
    `client_id` (Container Apps signs it in: no secret exists). azure-identity keeps the token
    and asks for a new one shortly before it expires."""
    from azure.identity import ManagedIdentityCredential

    credential = ManagedIdentityCredential(client_id=client_id)
    return lambda: credential.get_token(scope).token


def _bearer(token: Callable[[], str]):
    """`token` as a Databricks SDK sign-in: a bearer header, asked for on every request. (The
    Python SDK has no managed-identity sign-in of its own.)"""
    from databricks.sdk.credentials_provider import CredentialsStrategy

    class Bearer(CredentialsStrategy):
        def auth_type(self) -> str:
            return "azure-managed-identity"

        def __call__(self, cfg):
            return lambda: {"Authorization": f"Bearer {token()}"}

    return Bearer()


def databricks_api(managed_identity: str | None = None):
    """The Databricks API client for live mode: as the website's managed identity on Container
    Apps (`managed_identity` = its client id; a service principal in the workspace), otherwise as
    the identity in the environment (see `databricks_token`). A request may take 150 s: a cold
    Genie call has hung ~120 s, and the SDK's default 60 s would give up and send it again,
    starting a second conversation."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

    strategy = (
        _bearer(managed_identity_token(managed_identity, DATABRICKS_SCOPE))
        if managed_identity
        else None
    )
    config = Config(http_timeout_seconds=150, credentials_strategy=strategy)
    return WorkspaceClient(config=config).api_client


def databricks_token() -> str:
    """A current token for the Databricks identity in the environment: a service principal
    (DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET) or a CLI profile (DATABRICKS_CONFIG_PROFILE).
    The SDK refreshes it before it expires."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient().config.authenticate()["Authorization"].removeprefix("Bearer ")


@dataclass(frozen=True)
class LLMConnection:
    api_key: str | Callable[[], str] = field(repr=False)
    base_url: str | None  # None: the OpenAI API


@dataclass(frozen=True)
class Settings:
    # demo: snapshot files (DuckDB, FAISS, local model). live: prod's Gold on the SQL warehouse,
    # the chat's numbers from prod's Genie space; the model, documents and recordings still come
    # from the snapshot.
    mode: str = field(default_factory=lambda: os.environ.get("MODE", "demo"))
    # Container Apps: the client id of the website's user-assigned managed identity. When set, the
    # API signs in to Databricks (live mode) and Azure OpenAI as it, and no secret exists. Unset
    # (locally): the CLI profile (DATABRICKS_CONFIG_PROFILE) and the key in .env.
    managed_identity: str | None = field(
        default_factory=lambda: os.environ.get("MANAGED_IDENTITY_CLIENT_ID") or None
    )
    # live mode reads prod at DATABRICKS_HOST
    live_catalog: str = field(
        default_factory=lambda: os.environ.get("LIVE_CATALOG", "heavden_prod")
    )
    warehouse_id: str = field(
        default_factory=lambda: os.environ.get("WAREHOUSE_ID", "0a14d0d391ce8af6")
    )
    genie_title: str = field(
        default_factory=lambda: os.environ.get("GENIE_TITLE", "HeavDen Ward Data (prod)")
    )
    # the snapshot's folder; with SNAPSHOT_URL set, where it is downloaded to at start-up
    snapshot_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SNAPSHOT_DIR", ROOT / "data" / "demo_snapshot")
        )
    )
    # Container Apps: where the snapshot is published in Azure Blob Storage (its pointer,
    # current-v<format>.json, names it), read as the managed identity, e.g. https://<account>.blob.core.windows.net/
    # website/prod. Unset (locally): SNAPSHOT_DIR as it is.
    snapshot_url: str | None = field(default_factory=lambda: os.environ.get("SNAPSHOT_URL") or None)
    # the document index built into the image from the documents in git (heavden_api.documents);
    # unset (locally): the snapshot's own rag/ folder
    rag_dir: Path | None = field(
        default_factory=lambda: Path(os.environ["RAG_DIR"]) if os.environ.get("RAG_DIR") else None
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
    # live (LLM) answers per UTC day across all visitors; past it, recordings are replayed.
    # 100 (20 Oct): at most ~$4 a day even on gpt-5.5 (~4 cents an answer)
    chat_live_answers_per_day: int = field(
        default_factory=lambda: int(os.environ.get("CHAT_LIVE_ANSWERS_PER_DAY", "100"))
    )
    # Container Apps: the Azure Table that keeps the website's daily counts across restarts (that
    # one and the warehouse time below), written as the managed identity, e.g.
    # https://<account>.table.core.windows.net/websitecounters. Unset (locally): in memory.
    chat_budget_table_url: str | None = field(
        default_factory=lambda: os.environ.get("CHAT_BUDGET_TABLE_URL") or None
    )
    # live mode: SQL warehouse time per UTC day for the whole website, pages and chat and every
    # visitor together; past it, pages come from the snapshot and the chat answers from the
    # documents (numbers and patient details paused until midnight UTC).
    # 7200 (22 Oct): 2 hours, at most ~$5.60 a day at the warehouse's $2.80 an hour
    warehouse_seconds_per_day: int = field(
        default_factory=lambda: int(os.environ.get("WAREHOUSE_SECONDS_PER_DAY", "7200"))
    )
    # proxies we control in front of the API: 0 local, 1 Container Apps' ingress alone,
    # 2 Vercel -> Container Apps. The visitor's address is that many entries from the right of
    # X-Forwarded-For.
    trusted_proxy_hops: int = field(
        default_factory=lambda: int(os.environ.get("TRUSTED_PROXY_HOPS", "0"))
    )
    # Container Apps: the API's own URL is public, so it answers only requests that carry this
    # in the ORIGIN_HEADER header: the website's proxy (web/lib/backend.ts) and CI's smoke test.
    # Unset (locally, tests): every request is answered.
    origin_secret: str | None = field(
        default_factory=lambda: os.environ.get("ORIGIN_SECRET") or None, repr=False
    )

    def __post_init__(self):
        if self.mode not in ("demo", "live"):
            raise ValueError(f"MODE must be 'demo' or 'live', not {self.mode!r}")
        if self.chat_budget_table_url and not self.managed_identity:
            raise ValueError(
                "CHAT_BUDGET_TABLE_URL is written as the managed identity: "
                "set MANAGED_IDENTITY_CLIENT_ID"
            )
        if self.snapshot_url and not self.managed_identity:
            raise ValueError(
                "SNAPSHOT_URL is read as the managed identity: set MANAGED_IDENTITY_CLIENT_ID"
            )

    def chat_budget(self):
        """The daily cap on live answers: in the Azure Table on Container Apps, else in memory."""
        from heavden_api.chat.limits import DailyBudget, TableBudget

        if not self.chat_budget_table_url:
            return DailyBudget(self.chat_live_answers_per_day)
        token = managed_identity_token(self.managed_identity, STORAGE_SCOPE)
        return TableBudget(self.chat_live_answers_per_day, self.chat_budget_table_url, token)

    def warehouse_allowance(self):
        """The day's SQL warehouse time (live mode): in the Azure Table on Container Apps, else
        in memory."""
        from heavden_api.chat.limits import AzureTable, MemoryTable, WarehouseAllowance

        if not self.chat_budget_table_url:
            return WarehouseAllowance(self.warehouse_seconds_per_day, MemoryTable())
        token = managed_identity_token(self.managed_identity, STORAGE_SCOPE)
        table = AzureTable(self.chat_budget_table_url, token)
        return WarehouseAllowance(self.warehouse_seconds_per_day, table)

    def snapshot_reader(self) -> Callable[[str], bytes]:
        """Reads the published snapshot's files from Blob Storage as the managed identity."""
        from heavden_api.snapshot import blob_reader

        token = managed_identity_token(self.managed_identity, STORAGE_SCOPE)
        return blob_reader(self.snapshot_url, token)

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
        # the key when there is one (locally), otherwise the managed identity (Container Apps)
        if self.azure_openai_key:
            key = self.azure_openai_key
        elif self.managed_identity:
            key = managed_identity_token(self.managed_identity, AZURE_OPENAI_SCOPE)
        else:
            return None
        if not self.azure_openai_endpoint:
            raise ValueError("LLM_PROVIDER=azure needs AZURE_OPENAI_ENDPOINT as well")
        return LLMConnection(key, azure_base_url(self.azure_openai_endpoint))
