"""Request limits for the public chat endpoint (they protect the LLM budget).

Three layers, from finest to coarsest:

1. `RateLimiter`: at most N questions per visitor per hour.
2. `DailyBudget`: at most M live (LLM) answers per day across **all** visitors, so even someone
   rotating addresses can't run up the bill; past it, `/chat` replays recordings.
3. A monthly spending cap on the OpenAI key itself (set in OpenAI's dashboard): the backstop,
   because these in-memory counters reset whenever the server restarts.

Identifying a visitor: `X-Forwarded-For` is a list that each proxy appends to, so only the
entries added by **our own** proxies can be trusted; everything to their left was written by
the caller and can be forged. `client_address` therefore counts `trusted_hops` entries from the
right: 0 = ignore the header (local), 1 = the entry Render adds, 2 = the entry Vercel adds when
it proxies `/api/*` to Render (the real visitor).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime

MAX_TRACKED_CLIENTS = 10_000


def client_address(forwarded_for: str | None, peer: str | None, trusted_hops: int) -> str:
    """The visitor's address as seen by the outermost proxy we control."""
    if trusted_hops <= 0:
        return peer or "unknown"
    entries = [e.strip() for e in (forwarded_for or "").split(",") if e.strip()]
    if len(entries) >= trusted_hops:
        return entries[-trusted_hops]
    # fewer entries than expected: a proxy is missing, so fall back to the direct peer
    return peer or "unknown"


class RateLimiter:
    """At most `limit` requests per client per `window_seconds` (in memory)."""

    def __init__(self, limit: int, window_seconds: float = 3600, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self.seen: dict[str, deque] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = self.clock()
        if len(self.seen) > MAX_TRACKED_CLIENTS:
            self._forget_idle(now)
        window = self.seen[client]
        while window and now - window[0] > self.window:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(now)
        return True

    def _forget_idle(self, now: float) -> None:
        for client in [c for c, w in self.seen.items() if not w or now - w[-1] > self.window]:
            del self.seen[client]


class DailyBudget:
    """At most `limit` uses per UTC day across all clients."""

    def __init__(self, limit: int, clock=lambda: datetime.now(UTC)):
        self.limit = limit
        self.clock = clock
        self.day = None
        self.used = 0

    def allow(self) -> bool:
        today = self.clock().date()
        if today != self.day:
            self.day, self.used = today, 0
        if self.used >= self.limit:
            return False
        self.used += 1
        return True
