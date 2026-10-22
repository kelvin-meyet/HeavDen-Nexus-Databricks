"""Request limits for the public website (they protect the LLM budget and the SQL warehouse).

For the chat, three layers, from finest to coarsest:

1. `RateLimiter`: at most N questions per visitor per hour (in memory: no visitor data is
   stored; it restarts when the server does).
2. `DailyBudget` / `TableBudget`: at most M live (LLM) answers per day across **all** visitors,
   so even someone rotating addresses can't run up the bill; past it, `/chat` replays
   recordings. On Container Apps the count is kept in an Azure Table (`TableBudget`), because
   the API is stopped whenever nobody visits and an in-memory count would start again at zero.
3. A limit at the model provider itself (tokens per minute on the Azure OpenAI deployment): the
   backstop, which holds even if the counters fail.

In live mode, `WarehouseAllowance` caps the SQL warehouse time of the pages and the chat
together, per day across all visitors (in the same table).

Identifying a visitor: `X-Forwarded-For` is a list that each proxy appends to, so only the
entries added by **our own** proxies can be trusted; everything to their left was written by
the caller and can be forged. `client_address` therefore counts `trusted_hops` entries from the
right: 0 = ignore the header (local), 1 = the entry Container Apps' ingress adds, 2 = the entry
Vercel adds when it proxies `/api/*` to the API (the real visitor).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime

# the SQL warehouse stops this many seconds after its last query (its auto-stop, 5 minutes)
WAREHOUSE_AUTO_STOP = 300.0
TABLE_ERRORS = (urllib.error.URLError, TimeoutError, ValueError, KeyError)

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


class AzureTable:
    """The Azure Table calls the website's daily counts need, as the managed identity: read a
    row with its ETag, insert it, or replace it only if nobody changed it since it was read."""

    def __init__(self, url: str, token: Callable[[], str]):
        self.url, self.token = url.rstrip("/"), token

    def _row(self, partition: str, key: str) -> str:
        return f"{self.url}(PartitionKey='{partition}',RowKey='{key}')"

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

    def read(self, partition: str, key: str) -> tuple[dict | None, str | None]:
        """The row and its ETag (None and None if there is no row yet)."""
        try:
            with self._request("GET", self._row(partition, key)) as response:
                return json.loads(response.read()), response.headers["ETag"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, None
            raise

    def write(self, row: dict, etag: str | None) -> bool:
        """Insert the row, or replace it if it still has `etag`. False: someone was first."""
        try:
            if etag is None:
                self._request("POST", self.url, row, Prefer="return-no-content").close()
            else:
                url = self._row(row["PartitionKey"], row["RowKey"])
                self._request("PUT", url, row, **{"If-Match": etag}).close()
            return True
        except urllib.error.HTTPError as e:
            if e.code in (409, 412):  # inserted / changed by another request meanwhile
                return False
            raise


class MemoryTable:
    """`AzureTable` in memory (locally): the same reads and conditional writes."""

    def __init__(self):
        self.rows: dict[tuple[str, str], tuple[dict, int]] = {}

    def read(self, partition: str, key: str) -> tuple[dict | None, str | None]:
        found = self.rows.get((partition, key))
        return (dict(found[0]), str(found[1])) if found else (None, None)

    def write(self, row: dict, etag: str | None) -> bool:
        key = (row["PartitionKey"], row["RowKey"])
        current = self.rows.get(key)
        if (etag is None) != (current is None) or (current and str(current[1]) != etag):
            return False
        self.rows[key] = (dict(row), current[1] + 1 if current else 1)
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
        self.limit, self.table = limit, AzureTable(table_url, token)
        self.clock, self.attempts = clock, attempts

    def allow(self) -> bool:
        day = self.clock().date().isoformat()
        try:
            for _ in range(self.attempts):
                row, etag = self.table.read(self.PARTITION, day)
                used = int(row["Answers"]) if row else 0
                if used >= self.limit:
                    return False
                counted = {"PartitionKey": self.PARTITION, "RowKey": day, "Answers": used + 1}
                if self.table.write(counted, etag):
                    return True
            return False  # changed under us every time: refuse rather than over-count
        except TABLE_ERRORS as e:
            print(f"chat budget table unreachable, live answers refused: {e}", flush=True)
            return False


class AllowanceUsed(RuntimeError):
    """Today's warehouse allowance is used up: answer without the SQL warehouse."""


class WarehouseAllowance:
    """At most `seconds` a day (UTC) of SQL warehouse time for the website in live mode: the
    pages and the chat together, shared by every visitor.

    The warehouse bills while it is on and stops 5 minutes after its last query, so its cost
    depends on how often queries come, not on how many: one every 4 minutes keeps it on all day,
    far below any per-visitor limit. So this counts time. Before a query, the API reserves the
    time that query keeps the warehouse on: the query itself (`work` seconds at most) and the 5
    minutes after it, rounded up by `step` so a burst of queries makes one reservation. Time
    already reserved isn't counted again. Once the day's `seconds` are reserved, `reserve` says
    no until midnight UTC: pages are answered from the snapshot, and the chat's data and patient
    tools refuse (it still answers from the documents).

    It counts the time the website could have kept the warehouse on: an upper bound, as other
    users (CI, jobs) may keep it on anyway. The day's total and the end of the last reservation
    are kept in `table` (one row per UTC day, PartitionKey "warehouse-seconds"): an `AzureTable`
    on Container Apps, where the API is stopped whenever nobody visits. It fails closed: if the
    table can't be reached, the warehouse isn't used."""

    PARTITION = "warehouse-seconds"

    def __init__(
        self,
        seconds: float,
        table: AzureTable | MemoryTable,
        clock=time.time,
        step: float = 30.0,
        attempts: int = 5,
    ):
        self.seconds, self.table, self.clock = seconds, table, clock
        self.step, self.attempts = step, attempts
        self._until = 0.0  # the end of the latest reservation this process made or saw
        self._lock = threading.Lock()

    def _day(self, now: float) -> str:
        return datetime.fromtimestamp(now, UTC).date().isoformat()

    def reserve(self, work: float) -> bool:
        """Reserve the warehouse from now until `work` seconds plus its auto-stop from now.
        False: today's allowance can't cover it (or the table can't be reached)."""
        now = self.clock()
        needed = now + work + WAREHOUSE_AUTO_STOP
        with self._lock:  # one reservation at a time: a page's parallel queries share it
            if needed <= self._until:
                return True  # already reserved: no table call
            day = self._day(now)
            try:
                for _ in range(self.attempts):
                    row, etag = self.table.read(self.PARTITION, day)
                    used, until = (float(row["Seconds"]), float(row["Until"])) if row else (0, 0)
                    if needed <= until:
                        self._until = until
                        return True
                    end = needed + self.step
                    used += end - max(now, until)
                    if used > self.seconds:
                        return False
                    reserved = {
                        "PartitionKey": self.PARTITION,
                        "RowKey": day,
                        "Seconds": float(round(used, 1)),
                        "Until": float(end),
                    }
                    if self.table.write(reserved, etag):
                        self._until = end
                        return True
                return False  # changed under us every time: refuse rather than over-count
            except TABLE_ERRORS as e:
                print(f"warehouse allowance table unreachable, warehouse not used: {e}", flush=True)
                return False

    def can_reserve(self, work: float) -> bool:
        """Whether `reserve(work)` would succeed now, without reserving anything."""
        now = self.clock()
        needed = now + work + WAREHOUSE_AUTO_STOP
        if needed <= self._until:
            return True
        try:
            row, _ = self.table.read(self.PARTITION, self._day(now))
        except TABLE_ERRORS as e:
            print(f"warehouse allowance table unreachable, warehouse not used: {e}", flush=True)
            return False
        used, until = (float(row["Seconds"]), float(row["Until"])) if row else (0, 0)
        return needed <= until or used + needed + self.step - max(now, until) <= self.seconds

    def left(self) -> float:
        """Seconds of today's allowance not reserved yet (0 if the table can't be reached)."""
        try:
            row, _ = self.table.read(self.PARTITION, self._day(self.clock()))
            return max(0.0, self.seconds - (float(row["Seconds"]) if row else 0.0))
        except TABLE_ERRORS as e:
            print(f"warehouse allowance table unreachable, warehouse not used: {e}", flush=True)
            return 0.0
