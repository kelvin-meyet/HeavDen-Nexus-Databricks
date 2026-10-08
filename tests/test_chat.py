import json

import duckdb
import pytest

from heavden.agent import retrieval
from heavden.agent.sql_guard import MAX_ROWS, SqlRejected, check_sql
from heavden_api.app import create_app
from heavden_api.chat import sandbox as sandbox_module
from heavden_api.chat.agent import LLMReply, ToolCall, run_agent, system_prompt
from heavden_api.chat.limits import RateLimiter
from heavden_api.chat.recorded import Recordings, normalise
from heavden_api.chat.sandbox import SqlSandbox
from heavden_api.config import Settings
from heavden_api.snapshot import Snapshot

pytest.importorskip("faiss")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

TABLES = {"alerts_fact", "encounters"}


# --- SQL guard -------------------------------------------------------------------------------


def test_guard_allows_selects_over_gold_and_ctes_and_caps_rows():
    sql = "WITH a AS (SELECT site_id FROM gold.alerts_fact) SELECT site_id FROM a GROUP BY 1"
    safe = check_sql(sql + ";", TABLES)
    assert safe.startswith("SELECT * FROM (WITH a AS") and safe.endswith(f"LIMIT {MAX_ROWS}")


@pytest.mark.parametrize(
    "sql,reason",
    [
        ("DROP TABLE gold.alerts_fact", "single SELECT"),
        ("COPY gold.alerts_fact TO 'x.csv'", "single SELECT"),
        ("ATTACH 'other.db'", "single SELECT"),
        ("SELECT 1; SELECT 2", "one statement"),
        ("SELECT * FROM read_csv('secrets.csv')", "table functions"),
        ("SELECT * FROM 'data/file.parquet'", "not available"),
        ("SELECT * FROM main.alerts_fact", "not available"),
        ("SELECT * FROM gold.risk_scores", "not available"),  # not on this allow-list
        ("SELECT getenv('OPENAI_API_KEY')", "getenv"),
        ("SELECT * FROM gold.encounters WHERE site_id IN (SELECT * FROM read_text('x'))", "table"),
        ("   ", "empty"),
    ],
)
def test_guard_rejects_anything_but_a_gold_select(sql, reason):
    with pytest.raises(SqlRejected, match=reason):
        check_sql(sql, TABLES)


# --- Sandbox ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sandbox(demo_snapshot):
    return SqlSandbox(Snapshot.load(demo_snapshot))


def test_sandbox_answers_and_hides_internal_ids(sandbox):
    result = sandbox.run("SELECT site_id, count(*) AS n FROM gold.encounters GROUP BY 1 ORDER BY 1")
    assert result.columns == ["site_id", "n"] and [r[0] for r in result.rows] == [
        "SITE_A",
        "SITE_B",
    ]
    assert "patient_id" not in dict(sandbox.columns["encounters"])
    assert "gold.alerts_fact" in sandbox.schema_text()
    with pytest.raises(SqlRejected, match="patient_id"):
        sandbox.run("SELECT patient_id FROM gold.encounters")


def test_sandbox_reports_times_in_utc(sandbox):
    row = sandbox.run("SELECT max(prediction_ts) AS t FROM gold.risk_scores").rows[0]
    assert row[0].endswith("+00:00")


def test_sandbox_blocks_files_even_without_the_guard(sandbox):
    with pytest.raises(duckdb.PermissionException):
        sandbox.con.execute("SELECT * FROM read_csv('pyproject.toml')").fetchall()
    with pytest.raises(duckdb.InvalidInputException):
        sandbox.con.execute("SET enable_external_access = true")


def test_sandbox_stops_slow_queries(sandbox, monkeypatch):
    monkeypatch.setattr(sandbox_module, "TIMEOUT_SECONDS", 0.05)
    slow = "SELECT count(*) FROM gold.risk_scores a, gold.risk_scores b, gold.risk_scores c"
    with pytest.raises(SqlRejected, match="longer than"):
        sandbox.run(slow)
    assert sandbox.run("SELECT 1 AS ok").rows == [[1]]  # still usable afterwards


# --- Agent loop with a scripted LLM ----------------------------------------------------------


class ScriptedLLM:
    """Replays a fixed list of replies and records what it was sent."""

    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def call(name, **arguments):
    return ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(arguments))


@pytest.fixture(scope="module")
def state(demo_snapshot):
    from heavden_api.deps import build_state

    settings = Settings(snapshot_dir=demo_snapshot, openai_api_key=None)
    return build_state(settings, embedder=retrieval.HashingEmbedder())


def test_agent_runs_tools_then_answers(state):
    from heavden_api.chat.tools import Toolbox

    llm = ScriptedLLM(
        [
            LLMReply(
                None,
                [
                    call(
                        "query_gold",
                        sql="SELECT count(*) AS n FROM gold.encounters",
                        purpose="count",
                    )
                ],
            ),
            LLMReply(None, [call("search_documents", query="RRT extension Valley", as_of=None)]),
            LLMReply(None, [call("explain_patient", patient="e-001")]),
            LLMReply("There are 40 stays [1].", []),
        ]
    )
    answer = run_agent(llm, Toolbox(state, state.sandbox), "system", "how many?")
    assert answer.answer == "There are 40 stays [1]." and answer.rounds == 4
    tools = [s["tool"] for s in answer.steps]
    assert tools == ["query_gold", "search_documents", "explain_patient"]
    assert answer.steps[0]["rows"] == [[40]]
    assert answer.citations and answer.citations[0]["ref"] == 1
    assert answer.steps[2]["result"]["encounter_id"] == "E-001"
    # the tool results went back to the LLM, tied to their call ids
    last = llm.calls[-1]
    assert [m["role"] for m in last[-6:]] == ["assistant", "tool"] * 3
    assert last[-1]["tool_call_id"] == "call_explain_patient"


def test_agent_reports_tool_errors_to_the_llm(state):
    from heavden_api.chat.tools import Toolbox

    llm = ScriptedLLM(
        [
            LLMReply(None, [call("query_gold", sql="DROP TABLE gold.encounters", purpose="x")]),
            LLMReply(None, [ToolCall("bad", "query_gold", "{not json")]),
            LLMReply(None, [call("no_such_tool")]),
            LLMReply(None, [call("explain_patient", patient="P-0")]),
            LLMReply("Sorry.", []),
        ]
    )
    answer = run_agent(llm, Toolbox(state, state.sandbox), "system", "break things")
    results = [json.loads(m["content"]) for m in llm.calls[-1] if m["role"] == "tool"]
    assert "single SELECT" in results[0]["error"]
    assert "valid JSON" in results[1]["error"] and "unknown tool" in results[2]["error"]
    assert "no patient" in results[3]["error"]
    assert answer.answer == "Sorry."


def test_agent_stops_after_max_rounds(state):
    from heavden_api.chat.tools import Toolbox

    looping = ScriptedLLM([LLMReply(None, [call("explain_patient", patient="E-001")])] * 3)
    answer = run_agent(looping, Toolbox(state, state.sandbox), "s", "q", max_rounds=3)
    assert "couldn't finish" in answer.answer and len(answer.steps) == 3


def test_system_prompt_pins_now_and_bands():
    text = system_prompt("2026-11-14T23:00:00+00:00", {"high": 0.0167}, "gold.x")
    assert "TIMESTAMPTZ '2026-11-14T23:00:00+00:00'" in text and "0.0167" in text
    assert "SITE_B = HeavDen" in text and "never invent" in text


# --- /chat endpoint --------------------------------------------------------------------------


def _client(demo_snapshot, llm=None, **settings):
    settings = Settings(snapshot_dir=demo_snapshot, openai_api_key=None, **settings)
    app = create_app(settings, embedder=retrieval.HashingEmbedder(), llm=llm)
    return fastapi_testclient.TestClient(app)


def test_chat_live_returns_answer_steps_and_citations(demo_snapshot):
    llm = ScriptedLLM(
        [
            LLMReply(None, [call("search_documents", query="High band response", as_of=None)]),
            LLMReply("Review within 30 minutes [1].", []),
        ]
    )
    with _client(demo_snapshot, llm) as client:
        assert client.get("/chat/examples").json()["live"] is True
        body = client.post("/chat", json={"message": "What do I do for High?"}).json()
    assert body["mode"] == "live" and body["model"] == "scripted"
    assert body["answer"].startswith("Review") and body["citations"][0]["ref"] == 1
    assert body["steps"][0]["tool"] == "search_documents"


def test_chat_passes_history(demo_snapshot):
    llm = ScriptedLLM([LLMReply("ok", [])])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    with _client(demo_snapshot, llm) as client:
        client.post("/chat", json={"message": "and now?", "history": history})
    sent = llm.calls[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]


def test_chat_without_a_key_replays_recordings(demo_snapshot):
    recordings = Recordings.load()
    with _client(demo_snapshot) as client:
        examples = client.get("/chat/examples").json()
        assert examples["live"] is False and examples["questions"]
        if recordings.conversations:
            question = recordings.conversations[0]["question"]
            body = client.post("/chat", json={"message": question.upper() + "!!"}).json()
            assert (
                body["mode"] == "recorded"
                and body["answer"] == recordings.conversations[0]["answer"]
            )
        other = client.post("/chat", json={"message": "something nobody recorded"}).json()
    assert other["mode"] == "recorded" and other["suggestions"]


def test_chat_falls_back_when_the_llm_fails(demo_snapshot):
    class Broken:
        model = "broken"

        def complete(self, messages, tools):
            raise ConnectionError("down")

    with _client(demo_snapshot, Broken()) as client:
        body = client.post("/chat", json={"message": "anything"}).json()
    assert body["mode"] == "recorded" and "ConnectionError" in body["notice"]


def test_chat_is_rate_limited_and_validated(demo_snapshot):
    with _client(demo_snapshot, chat_requests_per_hour=2) as client:
        codes = [
            client.post("/chat", json={"message": "hello there"}).status_code for _ in range(3)
        ]
        assert codes == [200, 200, 429]
        assert client.post("/chat", json={"message": "x"}).status_code == 422
        assert client.post("/chat", json={"message": "y" * 1001}).status_code == 422


def test_rate_limiter_counts_per_client():
    limiter = RateLimiter(1)
    assert limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")


def test_recording_lookup_ignores_case_and_punctuation():
    assert normalise("What's NEWS2?") == normalise("whats news2")
