"""The Genie space over Gold (Plan.md §11.2 step 3), built from `genie_space.yaml`.

* `serialized_space(spec, catalog)`: Genie's `serialized_space` (version 2) from the readable
  spec. Ids are derived from what they name, so an unchanged spec gives the same payload; arrays
  are sorted as the API requires; `__CATALOG__` becomes the environment's catalog.
* `publish_space(api, ...)`: creates the space, or updates it in place (found by title), then
  gives the managers group CAN_MANAGE and the listed service principals (the website's identity
  in prod) CAN_RUN. `api` is a Databricks `ApiClient` (`.do(...)`); plain
  REST calls, so any SDK version on serverless will do.

Run as a module to write the payload to a file instead, e.g. to try a space by hand:
`python -m heavden.agent.genie --catalog heavden_staging --out space.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

import yaml

SPEC = Path(__file__).with_name("genie_space.yaml")
CATALOG = "__CATALOG__"
NAMESPACE = uuid.UUID("5f0c7c2e-6a43-4c1e-9d55-3b8f6f1d2a10")  # fixed: ids stay stable


def load_spec(path: Path = SPEC) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _id(kind: str, key: str) -> str:
    return uuid.uuid5(NAMESPACE, f"{kind}:{key}").hex


def _lines(text: str) -> list[str]:
    """Genie stores long text as a list of lines (each keeping its newline)."""
    return text.strip().splitlines(keepends=True)


def _column(name: str, conf: dict) -> dict:
    out: dict = {"column_name": name}
    if conf.get("exclude"):
        out["exclude"] = True
        return out
    if conf.get("description"):
        out["description"] = [conf["description"].strip()]
    if conf.get("synonyms"):
        out["synonyms"] = list(conf["synonyms"])
    if conf.get("matching"):  # a low-cardinality value people name directly
        out["enable_format_assistance"] = True
        out["enable_entity_matching"] = True
    return out


def _backticks(condition: str) -> str:
    return re.sub(r"\b(\w+)\.(\w+)\b", r"`\1`.`\2`", condition)


def serialized_space(spec: dict, catalog: str) -> dict:
    gold = f"{CATALOG}.gold"
    tables = [
        {
            "identifier": f"{gold}.{name}",
            "description": [t["description"].strip()],
            "column_configs": sorted(
                (_column(c, conf) for c, conf in t["columns"].items()),
                key=lambda c: c["column_name"],
            ),
        }
        for name, t in spec["tables"].items()
    ]
    joins = [
        {
            "id": _id("join", condition),
            "left": {"identifier": f"{gold}.{left}", "alias": left},
            "right": {"identifier": f"{gold}.{right}", "alias": right},
            "sql": [_backticks(condition), f"--rt=FROM_RELATIONSHIP_TYPE_{relationship}--"],
        }
        for left, right, condition, relationship in spec["joins"]
    ]
    measures = [
        {
            "id": _id("measure", m["alias"]),
            "alias": m["alias"],
            "display_name": m["display_name"],
            "sql": [m["sql"]],
            "synonyms": list(m.get("synonyms", [])),
            **({"instruction": [m["instruction"]]} if m.get("instruction") else {}),
        }
        for m in spec["measures"]
    ]
    filters = [
        {
            "id": _id("filter", f["display_name"]),
            "display_name": f["display_name"],
            "sql": [f["sql"]],
            "synonyms": list(f.get("synonyms", [])),
        }
        for f in spec["filters"]
    ]
    examples = [
        {"id": _id("example", e["question"]), "question": [e["question"]], "sql": _lines(e["sql"])}
        for e in spec["examples"]
    ]
    payload = {
        "version": 2,
        "config": {
            "sample_questions": sorted(
                ({"id": _id("sample", q), "question": [q]} for q in spec["sample_questions"]),
                key=lambda q: q["id"],
            )
        },
        "data_sources": {"tables": sorted(tables, key=lambda t: t["identifier"])},
        "instructions": {
            "text_instructions": [
                {"id": _id("text", "instructions"), "content": _lines(spec["instructions"])}
            ],
            "example_question_sqls": sorted(examples, key=lambda e: e["id"]),
            "sql_functions": sorted(
                ({"id": _id("function", f), "identifier": f} for f in spec["functions"]),
                key=lambda f: (f["id"], f["identifier"]),
            ),
            "join_specs": sorted(joins, key=lambda j: j["id"]),
            "sql_snippets": {
                "filters": sorted(filters, key=lambda f: f["id"]),
                "measures": sorted(measures, key=lambda m: m["id"]),
            },
        },
    }
    return json.loads(json.dumps(payload).replace(CATALOG, catalog))


def summary(payload: dict) -> str:
    inst = payload["instructions"]
    return (
        f"{len(payload['data_sources']['tables'])} tables, "
        f"{len(inst['example_question_sqls'])} example questions, "
        f"{len(inst['sql_snippets']['measures'])} measures, "
        f"{len(inst['sql_snippets']['filters'])} filters, {len(inst['join_specs'])} joins, "
        f"{len(inst['sql_functions'])} function"
    )


def find_space(api, title: str) -> str | None:
    """The id of the space with this title (raises if there are several), else None."""
    found, token = [], None
    while True:
        page = api.do("GET", "/api/2.0/genie/spaces", query={"page_token": token} if token else {})
        found += [s["space_id"] for s in page.get("spaces", []) if s.get("title") == title]
        token = page.get("next_page_token")
        if not token:
            break
    if len(found) > 1:
        raise RuntimeError(f"{len(found)} Genie spaces are titled {title!r}; keep one")
    return found[0] if found else None


def publish_space(
    api,
    payload: dict,
    title: str,
    description: str,
    warehouse_id: str,
    folder: str,
    managers: str,
    users: list[str] | tuple[str, ...] = (),
) -> str:
    """Create the space or update it in place, then give `managers` CAN_MANAGE and `users`
    (service principals' application ids) CAN_RUN. PATCH adds to the space's permissions without
    removing others. Returns what it did."""
    body = {
        "title": title,
        "description": description,
        "warehouse_id": warehouse_id,
        "serialized_space": json.dumps(payload),
    }
    space_id = find_space(api, title)
    if space_id is None:
        api.do("POST", "/api/2.0/workspace/mkdirs", body={"path": folder})
        space_id = api.do("POST", "/api/2.0/genie/spaces", body={**body, "parent_path": folder})[
            "space_id"
        ]
        action = "created"
    else:
        api.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)
        action = "updated"
    acl = [{"group_name": managers, "permission_level": "CAN_MANAGE"}] + [
        {"service_principal_name": user, "permission_level": "CAN_RUN"} for user in users
    ]
    api.do("PATCH", f"/api/2.0/permissions/genie/{space_id}", body={"access_control_list": acl})
    runners = f"; CAN_RUN for {len(users)} service principal(s)" if users else ""
    return (
        f"Genie space {title!r} ({space_id}): {action}; {summary(payload)}; "
        f"CAN_MANAGE for {managers}{runners}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the Genie space's serialized_space.")
    parser.add_argument("--catalog", required=True, help="e.g. heavden_staging")
    parser.add_argument("--out", required=True, help="file to write the payload to")
    args = parser.parse_args(argv)
    payload = serialized_space(load_spec(), args.catalog)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{args.out}: {summary(payload)}")


if __name__ == "__main__":
    main()
