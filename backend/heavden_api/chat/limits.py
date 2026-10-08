"""Per-client request limits for the public chat endpoint (protects the LLM budget)."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """At most `limit` requests per client per hour (in memory; one Render instance)."""

    def __init__(self, limit: int):
        self.limit = limit
        self.seen: dict[str, deque] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        window = self.seen[client]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(now)
        return True
