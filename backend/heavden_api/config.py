"""Settings from environment variables (Render sets these; locally the defaults work)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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
    # gpt-5.5 answered judgement questions reliably in testing; gpt-5.4-mini is cheaper but weaker
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-5.5"))
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
