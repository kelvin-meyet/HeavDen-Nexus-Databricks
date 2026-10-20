"""The daily cap on live chat answers, kept in an Azure Table so it survives restarts."""

import json
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

from heavden_api import config
from heavden_api.chat import limits
from heavden_api.chat.limits import DailyBudget, TableBudget
from heavden_api.config import Settings

TABLE = "https://heavdenjlytf.table.core.windows.net/websitecounters"


class AzureTable:
    """Table Storage as far as TableBudget uses it: GET a row (404 if none), POST a new row
    (409 if it exists), PUT a row only if its ETag still matches (412 if not)."""

    def __init__(self):
        self.rows: dict[str, tuple[int, str]] = {}
        self.requests = []
        self.before_put = None  # simulates another answer landing between read and write

    def urlopen(self, request, timeout):
        self.requests.append(request)
        method, url = request.get_method(), request.full_url
        if method == "POST":
            row = json.loads(request.data)
            if row["RowKey"] in self.rows:
                raise urllib.error.HTTPError(url, 409, "EntityAlreadyExists", {}, None)
            self.rows[row["RowKey"]] = (row["Answers"], "etag-1")
            return Response(b"")
        day = url.split("RowKey='")[1].split("'")[0]
        if method == "GET":
            if day not in self.rows:
                raise urllib.error.HTTPError(url, 404, "ResourceNotFound", {}, None)
            answers, etag = self.rows[day]
            return Response(json.dumps({"Answers": answers}).encode(), {"ETag": etag})
        assert method == "PUT"
        if self.before_put:
            self.before_put(), setattr(self, "before_put", None)
        if request.get_header("If-match") != self.rows[day][1]:
            raise urllib.error.HTTPError(url, 412, "UpdateConditionNotSatisfied", {}, None)
        version = int(self.rows[day][1].split("-")[1]) + 1
        self.rows[day] = (json.loads(request.data)["Answers"], f"etag-{version}")
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


def budget(limit: int, now: list[datetime]) -> TableBudget:
    return TableBudget(limit, TABLE, lambda: "token", clock=lambda: now[0])


def test_the_count_survives_the_api_being_stopped_and_started(table):
    now = [datetime(2026, 10, 20, 9, tzinfo=UTC)]
    first = budget(3, now)
    assert first.allow() and first.allow()
    restarted = budget(3, now)  # a cold start: a new process, same table
    assert restarted.allow()
    assert not restarted.allow()  # the 4th answer of the day is refused
    assert table.rows["2026-10-20"][0] == 3


def test_a_new_utc_day_starts_from_zero(table):
    now = [datetime(2026, 10, 20, 23, 59, tzinfo=UTC)]
    cap = budget(1, now)
    assert cap.allow() and not cap.allow()
    now[0] += timedelta(minutes=2)
    assert cap.allow()
    assert {day: answers for day, (answers, _) in table.rows.items()} == {
        "2026-10-20": 1,
        "2026-10-21": 1,
    }


def test_two_answers_at_once_are_both_counted(table):
    now = [datetime(2026, 10, 20, 9, tzinfo=UTC)]
    cap = budget(5, now)
    assert cap.allow()  # the day's row exists with 1

    def other_answer():
        table.rows["2026-10-20"] = (2, "etag-9")

    table.before_put = other_answer  # lands between our read and our write
    assert cap.allow()
    assert table.rows["2026-10-20"][0] == 3  # retried on the new count: nothing lost


def test_an_unreachable_table_refuses_live_answers(monkeypatch):
    def down(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(limits.urllib.request, "urlopen", down)
    assert not budget(300, [datetime(2026, 10, 20, tzinfo=UTC)]).allow()


def test_requests_carry_the_identitys_token_and_name_the_days_row(table):
    budget(300, [datetime(2026, 10, 20, tzinfo=UTC)]).allow()
    get, post = table.requests
    assert get.get_method() == "GET"
    assert get.full_url == f"{TABLE}(PartitionKey='chat-answers',RowKey='2026-10-20')"
    assert get.get_header("Authorization") == "Bearer token"
    assert get.get_header("X-ms-version") == "2021-08-06"
    assert post.get_method() == "POST" and post.full_url == TABLE
    assert json.loads(post.data) == {
        "PartitionKey": "chat-answers",
        "RowKey": "2026-10-20",
        "Answers": 1,
    }


def test_the_table_is_used_on_container_apps_and_memory_locally(monkeypatch):
    monkeypatch.setattr(config, "managed_identity_token", lambda client_id, scope: lambda: "t")
    assert isinstance(Settings(chat_budget_table_url=None).chat_budget(), DailyBudget)
    cap = Settings(chat_budget_table_url=TABLE, managed_identity="web-id").chat_budget()
    assert isinstance(cap, TableBudget) and cap.limit == 100
    with pytest.raises(ValueError, match="MANAGED_IDENTITY_CLIENT_ID"):
        Settings(chat_budget_table_url=TABLE, managed_identity=None)
