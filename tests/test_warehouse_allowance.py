"""The website's daily SQL warehouse allowance (live mode): pages and chat together, shared by
every visitor, kept in the same Azure Table as the chat's daily count."""

import json
import urllib.error
from datetime import UTC, datetime

import pytest

from heavden.agent import retrieval
from heavden_api import config
from heavden_api.app import create_app
from heavden_api.chat import limits
from heavden_api.chat.agent import LLMReply
from heavden_api.chat.databricks_tools import (
    ALLOWANCE_HINT,
    ALLOWANCE_USED,
    DatabricksToolbox,
    Genie,
)
from heavden_api.chat.limits import (
    WAREHOUSE_AUTO_STOP,
    AllowanceUsed,
    MemoryTable,
    WarehouseAllowance,
)
from heavden_api.config import Settings
from heavden_api.snapshot import Snapshot

pytest.importorskip("faiss")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

TABLE = "https://heavdenjlytf.table.core.windows.net/websitecounters"
NOON = datetime(2026, 10, 22, 12, tzinfo=UTC).timestamp()
WORK, STEP = 30.0, 30.0
ONE_WAKE = WORK + WAREHOUSE_AUTO_STOP + STEP  # 6 minutes for a query on a sleeping warehouse


def allowance(seconds: float, now: list[float], table=None) -> WarehouseAllowance:
    return WarehouseAllowance(seconds, table or MemoryTable(), clock=lambda: now[0], step=STEP)


def used(cap: WarehouseAllowance) -> float:
    return cap.seconds - cap.left()


def test_a_burst_of_queries_makes_one_reservation():
    now = [NOON]
    cap = allowance(7200, now)
    assert cap.reserve(WORK)
    for seconds in (1, 2, 5, 20):  # a page's parallel queries, then the next page
        now[0] = NOON + seconds
        assert cap.reserve(WORK)
    assert used(cap) == ONE_WAKE  # 6 minutes, not 5 x 6


def test_time_already_reserved_is_not_counted_twice():
    now = [NOON]
    cap = allowance(7200, now)
    cap.reserve(WORK)
    now[0] = NOON + 100  # still browsing: only the extension is counted
    cap.reserve(WORK)
    assert used(cap) == ONE_WAKE + 100
    now[0] = NOON + 1000  # the warehouse has stopped since: a new wake-up
    cap.reserve(WORK)
    assert used(cap) == 2 * ONE_WAKE + 100


def test_once_the_days_seconds_are_reserved_the_warehouse_is_refused_until_midnight_utc():
    now = [NOON]
    cap = allowance(2 * ONE_WAKE, now)
    assert cap.reserve(WORK)
    now[0] += 1000
    assert cap.reserve(WORK)
    now[0] += 1000
    assert not cap.reserve(WORK) and cap.left() == 0  # a third wake-up would exceed the day
    now[0] = datetime(2026, 10, 23, 0, 1, tzinfo=UTC).timestamp()
    assert cap.left() == 2 * ONE_WAKE and cap.reserve(WORK)


def test_a_slow_trickle_cannot_keep_the_warehouse_on_all_day():
    """One query every 4 minutes would keep the warehouse on around the clock (~$67 a day):
    with 2 hours a day, the warehouse is used for about 2 hours and then refused."""
    now = [NOON - 12 * 3600]  # midnight UTC
    cap = allowance(7200, now)
    allowed = 0
    for _ in range(24 * 15):  # every 4 minutes for a day
        allowed += cap.reserve(WORK)
        now[0] += 240
    assert used(cap) <= 7200 and allowed < 30  # ~2 hours of a 24-hour trickle


class AzureTable:
    """Table Storage as the counters use it: GET a row (404 if none), POST a new one (409 if it
    exists), PUT it only if its ETag still matches (412 if not)."""

    def __init__(self):
        self.rows: dict[tuple[str, str], tuple[dict, int]] = {}
        self.requests = []
        self.before_put = None  # another writer, between our read and our write

    def urlopen(self, request, timeout):
        self.requests.append(request)
        method, url = request.get_method(), request.full_url
        if method == "POST":
            row = json.loads(request.data)
            key = (row["PartitionKey"], row["RowKey"])
            if key in self.rows:
                raise urllib.error.HTTPError(url, 409, "EntityAlreadyExists", {}, None)
            self.rows[key] = (row, 1)
            return Response(b"")
        key = tuple(part.split("'")[1] for part in url.split("(")[1].split(","))
        if method == "GET":
            if key not in self.rows:
                raise urllib.error.HTTPError(url, 404, "ResourceNotFound", {}, None)
            row, version = self.rows[key]
            return Response(json.dumps(row).encode(), {"ETag": f"etag-{version}"})
        assert method == "PUT"
        if self.before_put:
            self.before_put(), setattr(self, "before_put", None)
        row, version = self.rows[key]
        if request.get_header("If-match") != f"etag-{version}":
            raise urllib.error.HTTPError(url, 412, "UpdateConditionNotSatisfied", {}, None)
        self.rows[key] = (json.loads(request.data), version + 1)
        return Response(b"")


class Response:
    def __init__(self, body: bytes, headers: dict | None = None):
        self.body, self.headers = body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body

    def close(self):
        pass


@pytest.fixture
def table(monkeypatch):
    fake = AzureTable()
    monkeypatch.setattr(limits.urllib.request, "urlopen", fake.urlopen)
    return fake


def test_the_allowance_survives_restarts_and_names_its_own_row(table):
    now = [NOON]
    first = allowance(7200, now, limits.AzureTable(TABLE, lambda: "token"))
    assert first.reserve(WORK)
    now[0] += 1000
    restarted = allowance(7200, now, limits.AzureTable(TABLE, lambda: "token"))
    assert restarted.reserve(WORK)  # a cold start: same table, the count goes on
    row, _ = table.rows[("warehouse-seconds", "2026-10-22")]
    assert row["Seconds"] == 2 * ONE_WAKE and row["Until"] == now[0] + ONE_WAKE
    get = table.requests[0]
    assert get.full_url == f"{TABLE}(PartitionKey='warehouse-seconds',RowKey='2026-10-22')"
    assert get.get_header("Authorization") == "Bearer token"
    assert isinstance(json.loads(table.requests[1].data)["Seconds"], float)  # Edm.Double


def test_a_reservation_made_meanwhile_elsewhere_is_counted_once(table):
    now = [NOON]
    cap = allowance(7200, now, limits.AzureTable(TABLE, lambda: "token"))
    cap.reserve(WORK)
    now[0] += 1000

    def elsewhere():  # another copy of the API reserves first
        key = ("warehouse-seconds", "2026-10-22")
        row, version = table.rows[key]
        table.rows[key] = (row | {"Seconds": row["Seconds"] + 50, "Until": row["Until"]}, 9)

    table.before_put = elsewhere
    assert cap.reserve(WORK)
    row, _ = table.rows[("warehouse-seconds", "2026-10-22")]
    assert row["Seconds"] == 2 * ONE_WAKE + 50  # retried on the new total: nothing lost


def test_an_unreachable_table_means_no_warehouse(monkeypatch):
    def down(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(limits.urllib.request, "urlopen", down)
    cap = allowance(7200, [NOON], limits.AzureTable(TABLE, lambda: "token"))
    assert not cap.reserve(WORK) and cap.left() == 0


def test_the_table_is_used_on_container_apps_and_memory_locally(monkeypatch):
    monkeypatch.setattr(config, "managed_identity_token", lambda client_id, scope: lambda: "t")
    local = Settings(chat_budget_table_url=None).warehouse_allowance()
    assert isinstance(local.table, MemoryTable) and local.seconds == 7200  # 2 hours a day
    cloud = Settings(chat_budget_table_url=TABLE, managed_identity="web-id").warehouse_allowance()
    assert isinstance(cloud.table, limits.AzureTable) and cloud.table.url == TABLE


def test_genie_asks_nothing_once_the_allowance_is_used_up():
    class NoCalls:
        def do(self, *args, **kwargs):
            raise AssertionError("Genie was asked")

    spent = allowance(0, [NOON])
    genie = Genie(NoCalls(), "sp1", allowance=spent)
    with pytest.raises(AllowanceUsed):
        genie.ask("How many stays?")
    toolbox = DatabricksToolbox("heavden_prod", None, genie, None)
    payload, step = toolbox.call("query_gold", {"question": "How many stays?"})
    assert json.loads(payload) == {"error": ALLOWANCE_USED, "next": ALLOWANCE_HINT}
    assert step["error"] == ALLOWANCE_USED  # what the visitor sees in the answer's steps


# The website in live mode, on the fake Databricks of test_live_mode (DuckDB over the snapshot).


def _live(demo_snapshot, seconds: int, llm=None):
    from test_live_mode import DatabricksOnDuckDB

    api = DatabricksOnDuckDB(Snapshot.load(demo_snapshot))
    settings = Settings(
        mode="live",
        snapshot_dir=demo_snapshot,
        llm_provider="openai",
        openai_api_key=None,
        warehouse_seconds_per_day=seconds,
    )
    app = create_app(settings, embedder=retrieval.HashingEmbedder(), llm=llm, api=api)
    return fastapi_testclient.TestClient(app), api


@pytest.fixture(scope="module")
def demo(demo_snapshot):
    settings = Settings(
        mode="demo", snapshot_dir=demo_snapshot, llm_provider="openai", openai_api_key=None
    )
    app = create_app(settings, embedder=retrieval.HashingEmbedder())
    with fastapi_testclient.TestClient(app) as client:
        yield client


PAGES = [
    ("/risk/board", {"limit": 10}),
    ("/risk/patients/E-001", {"hours": 6}),
    ("/analytics/summary", {}),
    ("/analytics/alerts", {"days": 3}),
]


def test_within_the_allowance_pages_use_the_warehouse_and_count_its_time(demo_snapshot):
    client, api = _live(demo_snapshot, 7200)
    with client:
        assert client.get("/risk/board").status_code == 200
        cap = client.app.state.heavden.warehouse_allowance
    assert api.statements() and used(cap) == ONE_WAKE


def test_past_the_allowance_pages_come_from_the_snapshot_without_a_query(demo_snapshot, demo):
    client, api = _live(demo_snapshot, 0)
    with client:
        for path, params in PAGES:
            assert client.get(path, params=params).json() == demo.get(path, params=params).json()
        assert client.get("/health").json()["as_of"] == demo.get("/health").json()["as_of"]
    assert api.statements() == []  # the warehouse was never asked


def test_past_the_allowance_the_chat_stays_live_on_the_documents_without_the_warehouse(
    demo_snapshot,
):
    """Numbers and patients need the warehouse, so their tools refuse at once and tell the model
    to answer from the documents; document search doesn't, so it still answers."""
    from test_chat import ScriptedLLM, call

    llm = ScriptedLLM(
        [
            LLMReply(None, [call("query_gold", question="How many stays?")]),
            LLMReply("Live numbers are paused until midnight UTC.", []),
            LLMReply(None, [call("search_documents", query="escalation protocol")]),
            LLMReply("The protocol says...", []),
        ]
    )
    client, api = _live(demo_snapshot, 0, llm=llm)
    with client:
        number = client.post("/chat", json={"message": "How many stays?"}).json()
        document = client.post("/chat", json={"message": "What is the protocol?"}).json()
    assert number["mode"] == "live" and number["steps"][0]["error"] == ALLOWANCE_USED
    told = json.loads(llm.calls[1][-1]["content"])  # the tool result the model received
    assert told == {"error": ALLOWANCE_USED, "next": ALLOWANCE_HINT}
    assert document["mode"] == "live" and document["steps"][0]["tool"] == "search_documents"
    assert "error" not in document["steps"][0]
    assert not [c for c in api.calls if c[1].endswith("start-conversation")]  # Genie not asked
    assert api.statements() == []  # nor the warehouse


def test_health_says_whether_the_pages_are_live(demo_snapshot, demo):
    within, _ = _live(demo_snapshot, 7200)
    past, _ = _live(demo_snapshot, 0)
    with within, past:
        assert within.get("/health").json()["live_data"] is True
        assert past.get("/health").json()["live_data"] is False
    assert demo.get("/health").json()["live_data"] is False


def test_can_reserve_answers_like_reserve_without_reserving():
    now = [NOON]
    cap = allowance(ONE_WAKE, now)
    assert cap.can_reserve(WORK) and cap.left() == ONE_WAKE  # nothing reserved by asking
    assert cap.reserve(WORK)
    now[0] += 10
    assert cap.can_reserve(WORK)  # inside the reservation: free
    now[0] += 1000
    assert not cap.can_reserve(WORK) and not cap.reserve(WORK)  # a new wake-up won't fit


def test_health_never_wakes_the_warehouse(demo_snapshot, demo):
    client, api = _live(demo_snapshot, 7200)
    with client:
        body = client.get("/health").json()
    assert body["mode"] == "live" and body["as_of"] == demo.get("/health").json()["as_of"]
    assert api.statements() == []  # the snapshot's hour until a page has asked prod


def test_made_up_patient_ids_are_refused_without_a_query(demo_snapshot):
    client, api = _live(demo_snapshot, 7200)
    with client:
        for junk in ("test", "E-12x", "P-1042", "E-" + "9" * 12):
            assert client.get(f"/risk/patients/{junk}").status_code == 404
            changes = {"encounter_id": junk, "changes": {"spo2": 90}}
            assert client.post("/risk/score", json=changes).status_code == 404
    assert api.statements() == []
