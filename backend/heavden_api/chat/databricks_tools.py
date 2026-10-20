"""The assistant's three tools on Databricks (Plan.md §11.2): the demo tools of `chat/tools.py`,
served by the platform instead of the snapshot.

* `query_gold`: the Genie space over Gold; a plain-English question in, the SQL it ran and the
  rows out.
* `search_documents`: the Vector Search index over `ml.corpus_chunks` while it exists, else the
  packaged index (the same chunks, embedded by the same model).
* `explain_patient`: the SQL table function `ml.explain_patient_risk` on the SQL warehouse.

Everything is a plain REST call through a Databricks `ApiClient` (`.do(method, path, ...)`), so
the same code runs in a Model Serving endpoint (automatic credentials), in a job, in the quality
test or on a laptop (CLI profile), with any SDK version.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from datetime import date

import numpy as np

from heavden.agent.corpus import Chunk
from heavden.agent.retrieval import SearchResult, in_force
from heavden_api.chat.tools import ROWS_TO_LLM, TOOL_SPECS, Toolbox

EMBEDDING_ENDPOINT = "databricks-gte-large-en"
GENIE_DONE = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}

# query_gold takes a question instead of SQL: Genie writes the SQL
GENIE_TOOL_SPECS = copy.deepcopy(TOOL_SPECS)
GENIE_TOOL_SPECS[0]["function"].update(
    description="Ask one plain-English question about the ward data: counts, rates, trends, "
    "comparisons between sites, units, days, patients. The data tool writes and runs the SQL "
    "and returns it with the rows.",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "One question, with its time window and sites or units in words.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    },
)


class ToolError(RuntimeError):
    """A tool couldn't answer; the message goes to the LLM, which can rephrase or say so."""


def _typed(value, type_name: str | None):
    """A value of a JSON_ARRAY result (all strings) as the column's type."""
    if value is None:
        return None
    kind = (type_name or "STRING").upper()
    if kind in ("INT", "LONG", "SHORT", "BYTE"):
        return int(value)
    if kind in ("DOUBLE", "FLOAT", "DECIMAL"):
        return float(value)
    if kind == "BOOLEAN":
        return str(value).lower() == "true"
    return value


def _table(statement_response: dict) -> tuple[list[str], list[list]]:
    """Columns and typed rows of a SQL statement response (inline JSON_ARRAY)."""
    schema = (statement_response.get("manifest") or {}).get("schema") or {}
    columns = schema.get("columns") or []
    names = [c["name"] for c in columns]
    types = [c.get("type_name") for c in columns]
    data = (statement_response.get("result") or {}).get("data_array") or []
    return names, [[_typed(v, t) for v, t in zip(row, types, strict=True)] for row in data]


class Warehouse:
    """SQL on a SQL warehouse through the Statement Execution API. With `catalog`, unqualified
    names resolve in it (`gold.risk_scores` is `<catalog>.gold.risk_scores`)."""

    def __init__(self, api, warehouse_id: str, timeout: float = 120.0, catalog: str | None = None):
        self.api, self.warehouse_id, self.timeout = api, warehouse_id, timeout
        self.catalog = catalog

    def execute(self, statement: str, parameters: list[dict] | None = None) -> dict:
        """The finished statement, all result chunks in `result.data_array`. `parameters` are
        the API's own ({"name", "value", "type"}; no value is NULL)."""
        body = {
            "warehouse_id": self.warehouse_id,
            "statement": statement,
            "parameters": parameters or [],
            "wait_timeout": "30s",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
        }
        if self.catalog:
            body["catalog"] = self.catalog
        response = self.api.do("POST", "/api/2.0/sql/statements", body=body)
        deadline = time.monotonic() + self.timeout
        while response["status"]["state"] in ("PENDING", "RUNNING"):  # a warehouse waking up
            if time.monotonic() > deadline:
                raise ToolError("the SQL warehouse didn't answer in time")
            time.sleep(1.0)
            response = self.api.do("GET", f"/api/2.0/sql/statements/{response['statement_id']}")
        if response["status"]["state"] != "SUCCEEDED":
            error = (response["status"].get("error") or {}).get("message", "unknown error")
            raise ToolError(f"the query failed: {error}")
        result = response.get("result") or {}
        link = result.get("next_chunk_internal_link")
        while link:  # a large result comes in chunks
            chunk = self.api.do("GET", link)
            result.setdefault("data_array", []).extend(chunk.get("data_array") or [])
            link = chunk.get("next_chunk_internal_link")
        return response

    def run(self, statement: str, **params: str) -> tuple[list[str], list[list]]:
        parameters = [{"name": k, "value": v, "type": "STRING"} for k, v in params.items()]
        return _table(self.execute(statement, parameters))


@dataclass(frozen=True)
class GenieAnswer:
    question: str
    sql: str | None
    columns: list[str]
    rows: list[list]
    text: str | None  # Genie's own short summary, if it wrote one


class Genie:
    """One question to a Genie space (Conversation API): start a conversation, wait for the
    message, read its query attachment's SQL and result."""

    def __init__(self, api, space_id: str, timeout: float = 120.0, poll: float = 1.5):
        self.api, self.space_id, self.timeout, self.poll = api, space_id, timeout, poll

    def ask(self, question: str) -> GenieAnswer:
        base = f"/api/2.0/genie/spaces/{self.space_id}"
        started = self.api.do("POST", f"{base}/start-conversation", body={"content": question})
        path = f"{base}/conversations/{started['conversation_id']}/messages/{started['message_id']}"
        message = started.get("message") or {}
        deadline = time.monotonic() + self.timeout
        while message.get("status") not in GENIE_DONE:
            if time.monotonic() > deadline:
                raise ToolError("the data tool didn't answer in time")
            time.sleep(self.poll)
            message = self.api.do("GET", path)
        if message["status"] != "COMPLETED":
            error = (message.get("error") or {}).get("error") or message["status"].lower()
            raise ToolError(f"the data tool couldn't answer: {error}")
        attachments = message.get("attachments") or []
        text = next((a["text"].get("content") for a in attachments if a.get("text")), None)
        query = next((a for a in attachments if a.get("query")), None)
        if query is None:  # Genie answered in words only (e.g. asked to clarify)
            return GenieAnswer(question, None, [], [], text)
        result = self.api.do("GET", f"{path}/attachments/{query['attachment_id']}/query-result")
        columns, rows = _table(result.get("statement_response") or {})
        return GenieAnswer(question, query["query"].get("query"), columns, rows, text)


class DatabricksEmbedder:
    """Embeddings from a Databricks embedding endpoint (OpenAI-style `input` / `data`).
    Small batches: on 16 Oct 2026 this account's pay-per-token limit refused a 64-text request
    for 5 minutes ("Exceeded workspace QPS rate limit") while 8-text requests went straight
    through; `pause` spaces the requests out."""

    def __init__(self, api, endpoint: str = EMBEDDING_ENDPOINT, batch: int = 8, pause: float = 0.2):
        self.api, self.endpoint, self.batch, self.name = api, endpoint, batch, endpoint
        self.pause = pause

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            if i:
                time.sleep(self.pause)
            response = self.api.do(
                "POST",
                f"/serving-endpoints/{self.endpoint}/invocations",
                body={"input": texts[i : i + self.batch]},
            )
            data = sorted(response["data"], key=lambda d: d.get("index", 0))
            out += [d["embedding"] for d in data]
        x = np.asarray(out, dtype=np.float32)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class VectorSearchDocuments:
    """The Vector Search index over `ml.corpus_chunks` (hybrid search), mapped back to the
    packaged chunks; which version of a document is in force is decided here, from all the
    chunks, exactly as the packaged index decides it."""

    def __init__(self, api, index_name: str, chunks: list[Chunk]):
        self.api, self.index_name = api, index_name
        self.chunks = {c.chunk_id: c for c in chunks}
        self._all = list(chunks)

    def search(
        self, query: str, k: int = 4, mode: str = "hybrid", as_of: date | None = None
    ) -> list[SearchResult]:
        allowed = {
            c.chunk_id for c, ok in zip(self._all, in_force(self._all, as_of), strict=True) if ok
        }
        filters = {"status": "current"} if as_of is None else {}
        body = {
            "columns": ["chunk_id"],
            "query_text": query,
            "num_results": k if as_of is None else 4 * k,
            "query_type": "HYBRID" if mode == "hybrid" else "ANN",
        }
        if filters:
            body["filters_json"] = json.dumps(filters)
        response = self.api.do(
            "POST", f"/api/2.0/vector-search/indexes/{self.index_name}/query", body=body
        )
        rows = (response.get("result") or {}).get("data_array") or []
        found = [(r[0], float(r[-1])) for r in rows if r[0] in allowed]
        return [
            SearchResult(self.chunks[cid], rank + 1, score, float("nan"))
            for rank, (cid, score) in enumerate(found[:k])
        ]


class DatabricksToolbox(Toolbox):
    """The three tools on Databricks; citations and step shapes as in the demo `Toolbox`, so the
    chat page shows them the same way."""

    specs = GENIE_TOOL_SPECS

    def __init__(self, catalog: str, warehouse: Warehouse, genie: Genie, documents):
        self.catalog, self.warehouse, self.genie, self._documents = (
            catalog,
            warehouse,
            genie,
            documents,
        )
        self.citations: list[dict] = []

    def documents(self):
        return self._documents

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        try:
            return super().call(name, arguments)
        except ToolError as exc:
            step = {"tool": name, **arguments, "error": str(exc)}
            return json.dumps({"error": str(exc)}), step

    def query_gold(self, question: str) -> tuple[str, dict]:
        answer = self.genie.ask(question)
        payload = {
            "sql": answer.sql,
            "columns": answer.columns,
            "rows": answer.rows[:ROWS_TO_LLM],
            "row_count": len(answer.rows),
            "truncated": len(answer.rows) > ROWS_TO_LLM,
        }
        if answer.sql is None:
            payload["note"] = answer.text or "no query was run"
        step = {
            "tool": "query_gold",
            "purpose": question,
            "sql": answer.sql,
            "columns": answer.columns,
            "rows": answer.rows[:20],
            "row_count": len(answer.rows),
            "engine": "genie",
        }
        return json.dumps(payload, default=str), step

    def explain_patient(self, patient: str) -> tuple[str, dict]:
        columns, rows = self.warehouse.run(
            f"SELECT * FROM {self.catalog}.ml.explain_patient_risk(:patient)", patient=patient
        )
        if not rows:
            message = f"no patient {patient.strip()} found"
            return json.dumps({"error": message}), {"tool": "explain_patient", "error": message}
        row = dict(zip(columns, rows[0], strict=True))
        row["top_factors"] = json.loads(row.get("top_factors") or "[]")
        row["recent_alerts_48h"] = json.loads(row.get("recent_alerts_48h") or "[]")
        step = {"tool": "explain_patient", "patient": row["patient_label"], "result": row}
        return json.dumps(row, default=str), step


def ward_now(warehouse: Warehouse, catalog: str) -> tuple[str, str]:
    """("now", first hour) of the hospital's data: the latest and first scored hours (UTC)."""
    _, rows = warehouse.run(
        f"SELECT min(prediction_ts), max(prediction_ts) FROM {catalog}.gold.risk_scores"
    )
    first, latest = rows[0]
    if latest is None:
        raise ToolError(f"{catalog}.gold.risk_scores has no scored hours")
    return str(latest), str(first)


def build_document_index(api, folder) -> str:
    """The packaged document index: every chunk of the corpus shipped in the wheel, embedded by
    the endpoint Vector Search uses, so the two searches differ only in the search engine."""
    from heavden.agent.corpus import chunk_corpus, load_corpus
    from heavden.agent.publish import corpus_dir
    from heavden.agent.retrieval import Retriever

    chunks = chunk_corpus(load_corpus(corpus_dir()))
    Retriever(chunks, DatabricksEmbedder(api)).save(folder)
    return f"{folder}: {len(chunks)} chunks embedded with {EMBEDDING_ENDPOINT}"


def packaged_documents(api, folder):
    """The packaged index, searched with questions embedded by the same endpoint."""
    from heavden.agent.retrieval import Retriever

    return Retriever.load(folder, DatabricksEmbedder(api))
