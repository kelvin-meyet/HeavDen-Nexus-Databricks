"""The assistant's three tools (Plan.md §11), demo-mode implementations.

| Tool | Demo mode (here) | Live mode (Phase 4-5) |
|---|---|---|
| `query_gold` | LLM-written SQL in the locked DuckDB sandbox | Genie space over gold |
| `search_documents` | FAISS + BM25 over the snapshot's chunks | Vector Search index |
| `explain_patient` | latest `risk_scores` row + top factors | UC function `explain_patient_risk` |

Each call returns a compact JSON string for the LLM and a `step` record for the app, which
shows the tool used, the SQL and the sources.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import TYPE_CHECKING

from heavden.agent.sql_guard import SqlRejected
from heavden_api.chat.sandbox import SqlSandbox
from heavden_api.snapshot import to_records

if TYPE_CHECKING:
    from heavden_api.deps import AppState

ROWS_TO_LLM = 50

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "query_gold",
            "description": "Run one read-only DuckDB SELECT over the gold tables to answer "
            "questions about numbers: counts, rates, trends, comparisons between sites, units, "
            "days, patients. Returns columns and rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement."},
                    "purpose": {
                        "type": "string",
                        "description": "One short sentence: what this query answers.",
                    },
                },
                "required": ["sql", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the hospital's documents (escalation protocol, NEWS2 "
            "reference, device manual and safety notices, drift runbook, model card, data card, "
            "data dictionary, site handbook). Returns numbered passages to cite as [n].",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "as_of": {
                        "type": ["string", "null"],
                        "description": "YYYY-MM-DD to use the document versions in force that "
                        "day (e.g. a past protocol), or null for the current versions.",
                    },
                },
                "required": ["query", "as_of"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_patient",
            "description": "Latest deterioration risk for one patient: risk, band, change over "
            "6 hours, NEWS2, the top reasons (SHAP factors) and recent alerts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient": {
                        "type": "string",
                        "description": "A patient label like P-1042 or a stay id like E-100941.",
                    }
                },
                "required": ["patient"],
                "additionalProperties": False,
            },
        },
    },
]

PATIENT_SQL = """
SELECT e.encounter_id, e.patient_label, e.age, e.sex, e.conditions, e.site_id, e.unit_id,
       e.bed_id, e.admit_ts, e.discharge_ts,
       r.prediction_ts, r.risk, r.risk_band, r.news2_total, r.top_factors,
       earlier.risk AS risk_6h_before
FROM gold.encounters e
JOIN gold.risk_scores r ON r.encounter_id = e.encounter_id
LEFT JOIN gold.risk_scores earlier
       ON earlier.encounter_id = r.encounter_id
      AND earlier.prediction_ts = r.prediction_ts - INTERVAL 6 HOURS
WHERE (e.patient_label = :patient OR e.encounter_id = :patient) AND r.prediction_ts <= :as_of
ORDER BY r.prediction_ts DESC
LIMIT 1
"""

PATIENT_ALERTS_SQL = """
SELECT alert_ts, round(risk, 4) AS risk, escalated_within_6h
FROM gold.alerts_fact
WHERE encounter_id = :encounter_id AND alert_ts > :as_of - INTERVAL 48 HOURS
  AND alert_ts <= :as_of
ORDER BY alert_ts
"""


class Toolbox:
    """Runs tool calls for one chat turn and numbers document passages across the turn."""

    def __init__(self, state: AppState, sandbox: SqlSandbox):
        self.state = state
        self.sandbox = sandbox
        self.citations: list[dict] = []

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        handler = {
            "query_gold": self.query_gold,
            "search_documents": self.search_documents,
            "explain_patient": self.explain_patient,
        }.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"}), {"tool": name, "error": True}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments: {exc}"}), {"tool": name, "error": True}

    def query_gold(self, sql: str, purpose: str = "") -> tuple[str, dict]:
        try:
            result = self.sandbox.run(sql)
        except SqlRejected as exc:
            step = {"tool": "query_gold", "purpose": purpose, "sql": sql, "error": str(exc)}
            return json.dumps({"error": str(exc)}), step
        payload = {
            "columns": result.columns,
            "rows": result.rows[:ROWS_TO_LLM],
            "row_count": len(result.rows),
            "truncated": result.truncated or len(result.rows) > ROWS_TO_LLM,
        }
        step = {
            "tool": "query_gold",
            "purpose": purpose,
            "sql": sql,
            "columns": result.columns,
            "rows": result.rows[:20],
            "row_count": len(result.rows),
        }
        return json.dumps(payload, default=str), step

    def search_documents(self, query: str, as_of: str | None = None) -> tuple[str, dict]:
        retriever = self.state.retriever()
        if retriever is None:
            return json.dumps({"error": "no document index"}), {"tool": "search_documents"}
        day = date.fromisoformat(as_of) if as_of else None
        results = retriever.search(query, k=4, as_of=day)
        passages = []
        for r in results:
            ref = next(
                (c["ref"] for c in self.citations if c["chunk_id"] == r.chunk.chunk_id), None
            )
            if ref is None:
                ref = len(self.citations) + 1
                self.citations.append(
                    {
                        "ref": ref,
                        "chunk_id": r.chunk.chunk_id,
                        "citation": r.chunk.citation,
                        "doc_id": r.chunk.doc_id,
                        "version": r.chunk.version,
                        "text": r.chunk.text,
                    }
                )
            passages.append({"ref": ref, "source": r.chunk.citation, "text": r.chunk.text})
        step = {
            "tool": "search_documents",
            "query": query,
            "as_of": as_of,
            "refs": [p["ref"] for p in passages],
        }
        return json.dumps({"passages": passages}), step

    def explain_patient(self, patient: str) -> tuple[str, dict]:
        patient = patient.strip().upper()
        if re.fullmatch(r"P-?\d+", patient):
            patient = "P-" + patient.removeprefix("P").removeprefix("-")
        rows = to_records(
            self.state.source.query(PATIENT_SQL, {"patient": patient, "as_of": self.state.as_of})
        )
        if not rows:
            message = f"no patient {patient} found"
            return json.dumps({"error": message}), {"tool": "explain_patient", "error": message}
        row = rows[0]
        row["top_factors"] = json.loads(row["top_factors"] or "[]")
        row["on_ward_now"] = row["prediction_ts"] == self.state.as_of.isoformat()
        row["recent_alerts_48h"] = to_records(
            self.state.source.query(
                PATIENT_ALERTS_SQL,
                {"encounter_id": row["encounter_id"], "as_of": self.state.as_of},
            )
        )
        row["risk_bands"] = self.state.snapshot.manifest.bands
        step = {"tool": "explain_patient", "patient": patient, "result": row}
        return json.dumps(row, default=str), step
