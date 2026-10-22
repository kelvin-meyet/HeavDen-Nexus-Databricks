"""The assistant's tools on Databricks (heavden_api.chat.databricks_tools), against a fake API."""

import json
from datetime import date

import numpy as np
import pytest

from heavden.agent.corpus import chunk_corpus, load_corpus
from heavden.agent.publish import corpus_dir
from heavden.agent.retrieval import HashingEmbedder, Retriever
from heavden_api.chat import databricks_tools as dt
from heavden_api.chat.agent import LLMReply, ToolCall, agent_events

CHUNKS = chunk_corpus(load_corpus(corpus_dir()))


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(dt.time, "sleep", lambda s: None)


class FakeApi:
    """Answers (method, path) from a table; a list answers one item per call."""

    def __init__(self, routes: dict):
        self.routes, self.calls = routes, []

    def do(self, method, path, query=None, body=None):
        self.calls.append((method, path, body))
        answer = self.routes[(method, path)]
        return answer.pop(0) if isinstance(answer, list) else answer


def _statement(columns, rows, state="SUCCEEDED", sid="s1"):
    return {
        "statement_id": sid,
        "status": {"state": state},
        "manifest": {"schema": {"columns": [{"name": n, "type_name": t} for n, t in columns]}},
        "result": {"data_array": rows},
    }


SPACE = "/api/2.0/genie/spaces/sp1"
MESSAGE = f"{SPACE}/conversations/c1/messages/m1"


def _genie(final_message: dict, result: dict | None = None) -> FakeApi:
    routes = {
        ("POST", f"{SPACE}/start-conversation"): {
            "conversation_id": "c1",
            "message_id": "m1",
            "message": {"status": "SUBMITTED"},
        },
        ("GET", MESSAGE): [{"status": "EXECUTING_QUERY"}, final_message],
    }
    if result is not None:
        routes[("GET", f"{MESSAGE}/attachments/a1/query-result")] = {"statement_response": result}
    return FakeApi(routes)


ANSWERED = {
    "status": "COMPLETED",
    "attachments": [
        {"attachment_id": "t1", "text": {"content": "General Hospital has 9."}},
        {"attachment_id": "a1", "query": {"query": "SELECT site_id, count(*) ..."}},
    ],
}
SITES = _statement(
    [("site_id", "STRING"), ("high_risk", "LONG")], [["SITE_A", "9"], ["SITE_B", "3"]]
)


def test_genie_waits_for_the_answer_and_returns_its_sql_and_typed_rows():
    api = _genie(ANSWERED, SITES)
    answer = dt.Genie(api, "sp1", poll=0).ask("High risk per site right now?")
    assert answer.sql.startswith("SELECT site_id") and answer.text == "General Hospital has 9."
    assert answer.columns == ["site_id", "high_risk"]
    assert answer.rows == [["SITE_A", 9], ["SITE_B", 3]]  # numbers come back as numbers
    assert api.calls[0][2] == {"content": "High risk per site right now?"}


def test_a_genie_failure_or_a_words_only_answer_is_reported_not_hidden():
    failed = {"status": "FAILED", "error": {"error": "table not found"}}
    with pytest.raises(dt.ToolError, match="table not found"):
        dt.Genie(_genie(failed), "sp1", poll=0).ask("x")
    words = {"status": "COMPLETED", "attachments": [{"text": {"content": "Which site?"}}]}
    answer = dt.Genie(_genie(words), "sp1", poll=0).ask("x")
    assert answer.sql is None and answer.rows == [] and answer.text == "Which site?"


def test_the_warehouse_waits_while_it_wakes_and_types_the_result():
    pending = {"statement_id": "s9", "status": {"state": "PENDING"}}
    done = _statement(
        [("n", "INT"), ("on_ward", "BOOLEAN"), ("risk", "DOUBLE")], [["2", "true", "0.5"]]
    )
    api = FakeApi(
        {
            ("POST", "/api/2.0/sql/statements"): pending,
            ("GET", "/api/2.0/sql/statements/s9"): [done],
        }
    )
    columns, rows = dt.Warehouse(api, "wh1").run("SELECT :p", p="P-1042")
    assert rows == [[2, True, 0.5]]
    body = api.calls[0][2]
    assert body["warehouse_id"] == "wh1"
    assert body["parameters"] == [
        {"name": "p", "value": "P-1042", "type": "STRING"}
    ]  # never pasted into the SQL
    failed = {"statement_id": "s2", "status": {"state": "FAILED", "error": {"message": "boom"}}}
    with pytest.raises(dt.ToolError, match="boom"):
        dt.Warehouse(FakeApi({("POST", "/api/2.0/sql/statements"): failed}), "wh1").run("x")


PATIENT = _statement(
    [
        ("encounter_id", "STRING"),
        ("patient_label", "STRING"),
        ("risk", "DOUBLE"),
        ("top_factors", "STRING"),
        ("recent_alerts_48h", "STRING"),
        ("on_ward_now", "BOOLEAN"),
    ],
    [["E-1", "P-1042", "0.03", '[{"factor": "oxygen saturation"}]', "[]", "true"]],
)


def _toolbox(genie_api=None, statement=PATIENT, documents=None) -> dt.DatabricksToolbox:
    sql = FakeApi({("POST", "/api/2.0/sql/statements"): statement})
    genie = dt.Genie(genie_api or _genie(ANSWERED, SITES), "sp1", poll=0)
    return dt.DatabricksToolbox("heavden_staging", dt.Warehouse(sql, "wh1"), genie, documents)


def test_query_gold_shows_genies_sql_and_rows_like_the_demo_tool():
    payload, step = _toolbox().call("query_gold", {"question": "High risk per site right now?"})
    assert json.loads(payload)["rows"] == [["SITE_A", 9], ["SITE_B", 3]]
    assert step["tool"] == "query_gold" and step["engine"] == "genie"
    assert step["purpose"] == "High risk per site right now?" and step["sql"].startswith("SELECT")
    failed = _genie({"status": "FAILED", "error": {"error": "timeout"}})
    payload, step = _toolbox(failed).call("query_gold", {"question": "q"})
    assert "timeout" in json.loads(payload)["error"] and step["error"]  # the LLM can rephrase


def test_explain_patient_calls_the_function_and_unpacks_its_json():
    toolbox = _toolbox()
    payload, step = toolbox.call("explain_patient", {"patient": "p1042"})
    row = json.loads(payload)
    assert (
        row["patient_label"] == "P-1042" and row["top_factors"][0]["factor"] == "oxygen saturation"
    )
    assert row["recent_alerts_48h"] == [] and row["on_ward_now"] is True
    sent = toolbox.warehouse.api.calls[0][2]
    assert "heavden_staging.ml.explain_patient_risk(:patient)" in sent["statement"]
    nobody = _toolbox(
        statement=_statement(PATIENT["manifest"]["schema"]["columns"] and [("x", "STRING")], [])
    )
    payload, step = nobody.call("explain_patient", {"patient": "P-9999"})
    assert json.loads(payload)["error"] == "no patient P-9999 found"


def test_document_search_cites_passages_as_the_demo_does():
    documents = Retriever(CHUNKS, HashingEmbedder())
    payload, step = _toolbox(documents=documents).call(
        "search_documents", {"query": "rapid response team NEWS2", "as_of": None}
    )
    passages = json.loads(payload)["passages"]
    assert [p["ref"] for p in passages] == [1, 2, 3, 4] and step["refs"] == [1, 2, 3, 4]


def test_vector_search_keeps_only_the_versions_in_force():
    protocol = [c for c in CHUNKS if c.doc_id == "HD-CLIN-001"]
    old, new = (next(c for c in protocol if c.version == v) for v in ("1.0", "2.0"))
    index = "/api/2.0/vector-search/indexes/heavden_staging.ml.corpus_chunks_index/query"
    api = FakeApi(
        {("POST", index): {"result": {"data_array": [[new.chunk_id, 0.9], [old.chunk_id, 0.8]]}}}
    )
    search = dt.VectorSearchDocuments(api, "heavden_staging.ml.corpus_chunks_index", CHUNKS)
    late_september = search.search("NEWS2 score", k=4, as_of=date(2026, 9, 25))
    assert [r.chunk.version for r in late_september] == ["1.0"]  # v2.0 wasn't in force yet
    assert "filters_json" not in api.calls[-1][2] and api.calls[-1][2]["query_type"] == "HYBRID"
    api.routes[("POST", index)] = {"result": {"data_array": [[new.chunk_id, 0.9]]}}
    current = search.search("NEWS2 score", k=4)
    assert [r.chunk.version for r in current] == ["2.0"]
    assert json.loads(api.calls[-1][2]["filters_json"]) == {"status": "current"}


def test_embeddings_come_in_batches_and_are_normalised():
    calls = []

    class Api:
        def do(self, method, path, query=None, body=None):
            calls.append(body["input"])
            return {
                "data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
            }

    vectors = dt.DatabricksEmbedder(Api(), batch=2).embed_documents(["a", "b", "c"])
    assert [len(c) for c in calls] == [2, 1] and np.allclose(vectors, [[0.6, 0.8]] * 3)


def test_the_loop_offers_genies_version_of_query_gold():
    offered = []

    class LLM:
        model = "fake"

        def complete(self, messages, tools):
            offered.append(tools)
            if len(offered) == 1:
                return LLMReply(
                    None, [ToolCall("c1", "query_gold", '{"question": "High risk per site?"}')]
                )
            return LLMReply("General Hospital has 9.")

    events = list(agent_events(LLM(), _toolbox(), "system", "High risk per site?"))
    params = offered[0][0]["function"]["parameters"]["properties"]
    assert list(params) == ["question"]  # Genie writes the SQL, not the LLM
    assert (
        events[-1]["answer"] == "General Hospital has 9."
        and events[-1]["steps"][0]["engine"] == "genie"
    )


def test_now_is_the_latest_scored_hour():
    api = FakeApi(
        {
            ("POST", "/api/2.0/sql/statements"): _statement(
                [("a", "TIMESTAMP"), ("b", "TIMESTAMP")],
                [["2026-09-20T01:00:00Z", "2026-10-05T23:00:00Z"]],
            )
        }
    )
    assert dt.ward_now(dt.Warehouse(api, "wh1"), "heavden_staging") == (
        "2026-10-05T23:00:00Z",
        "2026-09-20T01:00:00Z",
    )


def test_the_packaged_index_round_trips_with_the_same_embedding_model(tmp_path):
    class Api:
        def do(self, method, path, query=None, body=None):
            assert path == f"/serving-endpoints/{dt.EMBEDDING_ENDPOINT}/invocations"
            vectors = HashingEmbedder().embed_documents(body["input"])
            return {"data": [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vectors)]}

    message = dt.build_document_index(Api(), tmp_path / "docs")
    assert message.endswith(f"{len(CHUNKS)} chunks embedded with databricks-gte-large-en")
    docs = dt.packaged_documents(Api(), tmp_path / "docs")  # refuses another embedding model
    assert len(docs.chunks) == len(CHUNKS)
    found = docs.search("field safety notice FSN-2026-10")
    assert any(r.chunk.doc_id == "HD-DEV-001" for r in found)  # the device manual holds it
