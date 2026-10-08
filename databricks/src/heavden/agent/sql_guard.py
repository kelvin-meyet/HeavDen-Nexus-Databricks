"""Checks on LLM-written SQL before it runs (the assistant's text-to-SQL tool).

The assistant writes SQL from a user's question, so the SQL is untrusted. `check_sql` parses it
with DuckDB and only allows a **single SELECT** that reads **allow-listed `gold` tables** (or
CTEs defined in the same query), with no table functions (`read_csv`, `read_parquet`, ...) and
no file-touching or environment functions. The sandbox that runs it is a second, independent
layer (see `heavden_api.chat.sandbox`): no external access, configuration locked.

On Databricks the same role is played by Genie plus Unity Catalog grants: the service principal
can only SELECT from the gold schema.
"""

from __future__ import annotations

import json
import re

import duckdb

MAX_ROWS = 200
DENIED_FUNCTIONS = re.compile(
    r"^(read_|glob$|getenv$|current_setting$|duckdb_|pragma_|sniff_|parquet_|query$|query_table$)"
)


class SqlRejected(ValueError):
    """The SQL was not allowed to run; the message says why (shown to the LLM to fix it)."""


def _walk(node, visit) -> None:
    if isinstance(node, dict):
        visit(node)
        for value in node.values():
            _walk(value, visit)
    elif isinstance(node, list):
        for value in node:
            _walk(value, visit)


def check_sql(sql: str, allowed_tables: set[str], schema: str = "gold") -> str:
    """Validate `sql` and return it with a row limit applied. Raises `SqlRejected`."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SqlRejected("empty query")
    con = duckdb.connect()
    try:
        tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    finally:
        con.close()
    if tree.get("error"):
        raise SqlRejected(f"only a single SELECT query is allowed ({tree.get('error_message')})")
    if len(tree["statements"]) != 1:
        raise SqlRejected("only one statement is allowed")

    ctes: set[str] = set()
    tables: list[tuple[str, str]] = []
    problems: list[str] = []

    def visit(node: dict) -> None:
        if isinstance(node.get("cte_map"), dict):
            ctes.update(entry["key"].lower() for entry in node["cte_map"].get("map", []))
        kind = node.get("type")
        if kind == "BASE_TABLE":
            tables.append(((node.get("schema_name") or "").lower(), node["table_name"].lower()))
        elif kind == "TABLE_FUNCTION":
            problems.append("table functions are not allowed")
        elif node.get("class") == "FUNCTION" and DENIED_FUNCTIONS.match(
            str(node.get("function_name", "")).lower()
        ):
            problems.append(f"function {node['function_name']} is not allowed")

    _walk(tree["statements"], visit)
    available = ", ".join(sorted(f"{schema}.{t}" for t in allowed_tables))
    for table_schema, name in tables:
        if table_schema == "" and name in ctes:
            continue
        if table_schema != schema or name not in allowed_tables:
            label = f"{table_schema}.{name}" if table_schema else name
            problems.append(f"table {label} is not available; use {available}")
    if problems:
        raise SqlRejected("; ".join(dict.fromkeys(problems)))
    return f"SELECT * FROM ({sql}) AS q LIMIT {MAX_ROWS}"
