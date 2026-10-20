"""Live mode (MODE=live): prod's Gold through the SQL warehouse, the chat's numbers from Genie.

The fake Databricks here runs each SQL statement on DuckDB over the demo snapshot, as the warehouse
would on prod, so live mode must give the same answers as demo mode on the same data.
"""

import pandas as pd
import pytest

from heavden.agent import retrieval
from heavden_api.app import create_app
from heavden_api.chat.agent import LLMReply
from heavden_api.chat.databricks_tools import DatabricksToolbox, Warehouse
from heavden_api.config import Settings
from heavden_api.snapshot import Snapshot
from heavden_api.sources import DemoSource, LiveSource, _frame, _parameter

pytest.importorskip("faiss")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

SPACES = "/api/2.0/genie/spaces"
STATEMENTS = "/api/2.0/sql/statements"
MESSAGE = f"{SPACES}/sp1/conversations/c1/messages/m1"


def _value(parameter: dict):
    if "value" not in parameter:
        return None
    value, kind = parameter["value"], parameter["type"]
    if kind == "INT":
        return int(value)
    if kind == "DOUBLE":
        return float(value)
    if kind == "TIMESTAMP":
        return pd.Timestamp(value, tz="UTC")
    if kind == "BOOLEAN":
        return value == "true"
    return value


def _type(column: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(column):
        return "TIMESTAMP"
    if pd.api.types.is_bool_dtype(column):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(column):
        return "LONG"
    if pd.api.types.is_float_dtype(column):
        return "DOUBLE"
    return "STRING"


def _cell(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class DatabricksOnDuckDB:
    """A fake Databricks API: statements run on DuckDB over the snapshot (the catalog is
    ignored), one Genie space is listed and answers one question."""

    def __init__(self, snapshot: Snapshot):
        self.duckdb = DemoSource(snapshot)
        self.calls = []

    def do(self, method, path, query=None, body=None):
        self.calls.append((method, path, body))
        if path == SPACES:
            return {"spaces": [{"space_id": "sp1", "title": "HeavDen Ward Data (prod)"}]}
        if path == STATEMENTS:
            params = {p["name"]: _value(p) for p in body["parameters"]}
            frame = self.duckdb.query(body["statement"], params)
            return {
                "statement_id": "s1",
                "status": {"state": "SUCCEEDED"},
                "manifest": {
                    "schema": {
                        "columns": [{"name": c, "type_name": _type(frame[c])} for c in frame]
                    }
                },
                "result": {
                    "data_array": [[_cell(v) for v in row] for row in frame.itertuples(False)]
                },
            }
        if path == f"{SPACES}/sp1/start-conversation":
            return {"conversation_id": "c1", "message_id": "m1", "message": {"status": "SUBMITTED"}}
        if path == MESSAGE:
            return {
                "status": "COMPLETED",
                "attachments": [{"attachment_id": "a1", "query": {"query": "SELECT 40"}}],
            }
        if path == f"{MESSAGE}/attachments/a1/query-result":
            schema = {"columns": [{"name": "stays", "type_name": "LONG"}]}
            result = {"manifest": {"schema": schema}, "result": {"data_array": [["40"]]}}
            return {"statement_response": result}
        raise AssertionError(f"unexpected call {method} {path}")

    def statements(self) -> list[dict]:
        return [body for _, path, body in self.calls if path == STATEMENTS]


def _client(demo_snapshot, mode: str, api=None, llm=None):
    settings = Settings(
        mode=mode, snapshot_dir=demo_snapshot, llm_provider="openai", openai_api_key=None
    )
    app = create_app(settings, embedder=retrieval.HashingEmbedder(), llm=llm, api=api)
    return fastapi_testclient.TestClient(app)


@pytest.fixture(scope="module")
def clients(demo_snapshot):
    api = DatabricksOnDuckDB(Snapshot.load(demo_snapshot))
    with _client(demo_snapshot, "demo") as demo, _client(demo_snapshot, "live", api) as live:
        yield demo, live, api


@pytest.mark.parametrize(
    "path,params",
    [
        ("/analytics/summary", {}),
        ("/analytics/summary", {"site_id": "SITE_A"}),
        ("/analytics/census", {"hours": 6}),
        ("/analytics/units", {}),
        ("/analytics/alerts", {"days": 3}),
        ("/analytics/devices", {"days": 1}),
        ("/risk/board", {"limit": 10}),
        ("/risk/board", {"band": "High"}),
        ("/risk/patients/E-001", {"hours": 6}),
    ],
)
def test_live_pages_match_demo_pages_on_the_same_data(clients, path, params):
    demo, live, _ = clients
    assert live.get(path, params=params).json() == demo.get(path, params=params).json()


def test_live_health_takes_now_from_the_warehouse(clients):
    demo, live, api = clients
    body, demo_body = live.get("/health").json(), demo.get("/health").json()
    assert body["mode"] == "live" and body["as_of"] == demo_body["as_of"]
    assert body["data_start"] == demo_body["data_start"] and body["rows"] == {}
    sent = api.statements()
    assert all(s["catalog"] == "heavden_prod" for s in sent)  # gold.x is heavden_prod.gold.x
    assert sum("max(prediction_ts)" in s["statement"] for s in sent) == 1  # "now" is kept


def test_live_chat_asks_genie_for_numbers(demo_snapshot):
    from test_chat import ScriptedLLM, call

    api = DatabricksOnDuckDB(Snapshot.load(demo_snapshot))
    llm = ScriptedLLM(
        [LLMReply(None, [call("query_gold", question="How many stays?")]), LLMReply("40.", [])]
    )
    with _client(demo_snapshot, "live", api, llm) as live:
        state = live.app.state.heavden
        assert isinstance(state.toolbox(), DatabricksToolbox)
        assert "plain-English question" in state.system_prompt()  # the Genie rules
        reply = live.post("/chat", json={"message": "How many stays?"}).json()
    assert reply["mode"] == "live" and reply["answer"] == "40."
    assert reply["steps"][0]["engine"] == "genie" and reply["steps"][0]["rows"] == [[40]]
    asked = next(b for _, p, b in api.calls if p.endswith("start-conversation"))
    assert asked == {"content": "How many stays?"}


def test_live_mode_needs_the_genie_space(demo_snapshot):
    class NoSpaces(DatabricksOnDuckDB):
        def do(self, method, path, query=None, body=None):
            return {"spaces": []} if path == SPACES else super().do(method, path, query, body)

    api = NoSpaces(Snapshot.load(demo_snapshot))
    with pytest.raises(RuntimeError, match="no Genie space"):
        with _client(demo_snapshot, "live", api):
            pass


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, {"name": "p", "type": "STRING"}),
        ("SITE_A", {"name": "p", "value": "SITE_A", "type": "STRING"}),
        (72, {"name": "p", "value": "72", "type": "INT"}),
        (True, {"name": "p", "value": "true", "type": "BOOLEAN"}),
        (0.5, {"name": "p", "value": "0.5", "type": "DOUBLE"}),
        (
            pd.Timestamp("2026-10-13T19:00:00-04:00"),
            {"name": "p", "value": "2026-10-13 23:00:00", "type": "TIMESTAMP"},
        ),
    ],
)
def test_parameters_are_typed_and_times_sent_in_utc(value, expected):
    assert _parameter("p", value) == expected


def test_results_are_typed_like_demo_mode():
    columns = [("t", "TIMESTAMP"), ("d", "DATE"), ("n", "LONG"), ("x", "DECIMAL"), ("b", "BOOLEAN")]
    response = {
        "manifest": {"schema": {"columns": [{"name": n, "type_name": t} for n, t in columns]}},
        "result": {
            "data_array": [
                ["2026-10-13T23:00:00.000Z", "2026-10-13", "3", "0.25", "true"],
                [None, None, None, None, None],
            ]
        },
    }
    frame = _frame(response)
    assert frame["t"].iloc[0] == pd.Timestamp("2026-10-13 23:00", tz="UTC")
    assert frame["d"].iloc[0] == pd.Timestamp("2026-10-13", tz="UTC")
    assert frame["n"].iloc[0] == 3 and pd.isna(frame["n"].iloc[1]) and frame["x"].iloc[0] == 0.25
    assert bool(frame["b"].iloc[0]) and pd.isna(frame["b"].iloc[1])


class CountingWarehouse:
    def __init__(self, rows):
        self.rows, self.sent = rows, []

    def execute(self, statement, parameters=None):
        self.sent.append((statement, parameters))
        schema = {"columns": [{"name": "latest", "type_name": "TIMESTAMP"}]}
        if "min(prediction_ts)" in statement:
            schema["columns"].insert(0, {"name": "first", "type_name": "TIMESTAMP"})
        return {"manifest": {"schema": schema}, "result": {"data_array": self.rows}}


def test_live_source_keeps_answers_and_now_and_sends_only_named_parameters():
    warehouse = CountingWarehouse([["2026-09-20T01:00:00Z", "2026-10-13T23:00:00Z"]])
    source = LiveSource(warehouse)
    assert source.now() == (
        pd.Timestamp("2026-10-13 23:00", tz="UTC"),
        pd.Timestamp("2026-09-20 01:00", tz="UTC"),
    )
    source.now()
    assert len(warehouse.sent) == 1  # asked once per 30 minutes

    warehouse.rows = [["2026-10-13T23:00:00Z"]]
    sql = "SELECT max(t) AS latest FROM gold.x WHERE t <= :as_of"
    params = {"as_of": pd.Timestamp("2026-10-13 23:00", tz="UTC"), "site_id": None}
    first, again = source.query(sql, params), source.query(sql, params)
    assert first.equals(again) and len(warehouse.sent) == 2  # the same query: not sent again
    assert [p["name"] for p in warehouse.sent[1][1]] == ["as_of"]  # site_id isn't in the SQL
    source.query(sql, params | {"as_of": pd.Timestamp("2026-10-14 00:00", tz="UTC")})
    assert len(warehouse.sent) == 3  # a new "now" is a new query


def test_warehouse_reads_every_result_chunk():
    class Api:
        def do(self, method, path, query=None, body=None):
            if method == "POST":
                assert body["catalog"] == "heavden_prod"
                return {
                    "statement_id": "s1",
                    "status": {"state": "SUCCEEDED"},
                    "manifest": {"schema": {"columns": [{"name": "n", "type_name": "INT"}]}},
                    "result": {"data_array": [["1"]], "next_chunk_internal_link": "/chunk/1"},
                }
            return {"data_array": [["2"]]}

    response = Warehouse(Api(), "w1", catalog="heavden_prod").execute("SELECT n FROM gold.x")
    assert response["result"]["data_array"] == [["1"], ["2"]]


def test_the_app_tool_reads_alerts_up_to_the_patients_own_last_hour():
    import duckdb

    from heavden_api.chat.tools import PATIENT_ALERTS_SQL
    from heavden_api.sources import _PARAM

    last_hour = pd.Timestamp("2026-10-11 13:00", tz="UTC")
    alerts = pd.DataFrame(
        {
            "encounter_id": ["E-1"] * 3,
            "alert_ts": [last_hour - pd.Timedelta(hours=h) for h in (50, 4, 0)],
            "risk": [0.2, 0.3, 0.4],
            "escalated_within_6h": [0.0, 0.0, None],
        }
    )
    con = duckdb.connect()
    con.execute("CREATE SCHEMA gold")
    con.register("alerts", alerts)
    con.execute("CREATE TABLE gold.alerts_fact AS SELECT * FROM alerts")
    sql = _PARAM.sub(r"$\1", PATIENT_ALERTS_SQL)
    rows = con.execute(sql, {"encounter_id": "E-1", "last_hour": last_hour}).df()
    assert rows["risk"].tolist() == [0.3, 0.4]  # within 48 h of that hour, not of the ward's now
