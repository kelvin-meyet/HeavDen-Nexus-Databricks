"""Settings from environment variables (Render sets these; locally the defaults work)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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

    def __post_init__(self):
        if self.mode not in ("demo", "live"):
            raise ValueError(f"MODE must be 'demo' or 'live', not {self.mode!r}")
