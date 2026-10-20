"""Request limits for the public chat endpoint (they protect the LLM budget).

Three layers, from finest to coarsest:

1. `RateLimiter`: at most N questions per visitor per hour (in memory: no visitor data is
   stored; it restarts when the server does).
2. `DailyBudget` / `TableBudget`: at most M live (LLM) answers per day across **all** visitors,
   so even someone rotating addresses can't run up the bill; past it, `/chat` replays
   recordings. On Container Apps the count is kept in an Azure Table (`TableBudget`), because
   the API is stopped whenever nobody visits and an in-memory count would start again at zero.
3. A limit at the model provider itself (tokens per minute on the Azure OpenAI deployment): the
   backstop, which holds even if the counters fail.

Identifying a visitor: `X-Forwarded-For` is a list that each proxy appends to, so only the
entries added by **our own** proxies can be trusted; everything to their left was written by
the caller and can be forged. `client_address` therefore counts `trusted_hops` entries from the
right: 0 = ignore the header (local), 1 = the entry Container Apps' ingress adds, 2 = the entry
Vercel adds when it proxies `/api/*` to the API (the real visitor).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable
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


class TableBudget:
    """`DailyBudget` kept in an Azure Table: one row per UTC day (PartitionKey "chat-answers",
    RowKey the date, Answers the count), so the count survives the API being stopped and
    started. A row is written back only if nobody changed it since it was read (its ETag), so
    two answers at once can't both take the last place. If the table can't be reached, no live
    answer is allowed: the spend guard fails closed and visitors get recorded answers."""

    PARTITION = "chat-answers"

    def __init__(
        self,
        limit: int,
        table_url: str,
        token: Callable[[], str],
        clock=lambda: datetime.now(UTC),
        attempts: int = 5,
    ):
        self.limit, self.url, self.token = limit, table_url.rstrip("/"), token
        self.clock, self.attempts = clock, attempts

    def allow(self) -> bool:
        day = self.clock().date().isoformat()
        try:
            for _ in range(self.attempts):
                used, etag = self._read(day)
                if used >= self.limit:
                    return False
                if self._write(day, used + 1, etag):
                    return True
            return False  # changed under us every time: refuse rather than over-count
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"chat budget table unreachable, live answers refused: {e}", flush=True)
            return False

    def _row(self, day: str) -> str:
        return f"{self.url}(PartitionKey='{self.PARTITION}',RowKey='{day}')"

    def _request(self, method: str, url: str, body: dict | None = None, **headers):
        request = urllib.request.Request(
            url,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.token()}",
                "x-ms-version": "2021-08-06",
                "Accept": "application/json;odata=nometadata",
                "Content-Type": "application/json",
                **headers,
            },
        )
        return urllib.request.urlopen(request, timeout=10)

    def _read(self, day: str) -> tuple[int, str | None]:
        """Today's count and the row's ETag (0 and None before the first answer of the day)."""
        try:
            with self._request("GET", self._row(day)) as response:
                return int(json.loads(response.read())["Answers"]), response.headers["ETag"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0, None
            raise

    def _write(self, day: str, answers: int, etag: str | None) -> bool:
        """Insert today's row, or replace it if it still has `etag`. False: someone was first."""
        row = {"PartitionKey": self.PARTITION, "RowKey": day, "Answers": answers}
        try:
            if etag is None:
                self._request("POST", self.url, row, Prefer="return-no-content").close()
            else:
                self._request("PUT", self._row(day), row, **{"If-Match": etag}).close()
            return True
        except urllib.error.HTTPError as e:
            if e.code in (409, 412):  # inserted / changed by another answer meanwhile
                return False
            raise
