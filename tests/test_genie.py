"""The Genie space over Gold (heavden.agent.genie): the payload Genie's API accepts, built from
`genie_space.yaml`, and publishing it in place."""

import json
import re

import pandas as pd
import pytest
from test_batch_scoring import scored  # noqa: F401  (fixture)
from test_scoring_gold import fitted  # noqa: F401  (fixture)

from heavden.agent import genie
from heavden.analytics import gold

SPEC = genie.load_spec()
HEX = re.compile(r"[0-9a-f]{32}")


@pytest.fixture(scope="module")
def payload() -> dict:
    return genie.serialized_space(SPEC, "heavden_staging")


def _ids(items: list[dict]) -> list[str]:
    return [i["id"] for i in items]


def test_the_payload_follows_genies_rules(payload):
    inst, snippets = payload["instructions"], payload["instructions"]["sql_snippets"]
    keyed = [
        payload["config"]["sample_questions"],
        inst["example_question_sqls"],
        inst["join_specs"],
        inst["text_instructions"],
        snippets["filters"],
        snippets["measures"],
    ]
    for items in keyed:  # 32-hex ids, sorted by id
        assert all(HEX.fullmatch(i) for i in _ids(items)) and _ids(items) == sorted(_ids(items))
    instruction_ids = [i for items in keyed[1:] for i in _ids(items)]
    instruction_ids += _ids(inst["sql_functions"])
    assert len(set(instruction_ids)) == len(instruction_ids)  # unique across instructions
    tables = payload["data_sources"]["tables"]
    assert [t["identifier"] for t in tables] == sorted(t["identifier"] for t in tables)
    for t in tables:
        names = [c["column_name"] for c in t["column_configs"]]
        assert names == sorted(names) and len(set(names)) == len(names)
    for j in inst["join_specs"]:
        assert len(j["sql"]) == 2 and re.fullmatch(
            r"--rt=FROM_RELATIONSHIP_TYPE_\w+--", j["sql"][1]
        )
        assert "`" in j["sql"][0]  # columns in backticks
    assert len(inst["text_instructions"]) == 1
    assert len("".join(inst["text_instructions"][0]["content"])) < 2000  # Genie's guidance
    text = json.dumps(payload)
    assert genie.CATALOG not in text and "heavden_staging.gold.risk_scores" in text
    assert all(len(s) <= 25_000 for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text))


def test_text_and_sql_survive_as_lines(payload):
    example = payload["instructions"]["example_question_sqls"][0]
    assert all(line.endswith("\n") for line in example["sql"][:-1])
    original = next(e for e in SPEC["examples"] if e["question"] == example["question"][0])
    expected = original["sql"].strip().replace(genie.CATALOG, "heavden_staging")
    assert "".join(example["sql"]) == expected


def test_ids_are_stable_and_follow_what_they_name():
    again = genie.serialized_space(SPEC, "heavden_prod")
    first = genie.serialized_space(SPEC, "heavden_staging")
    assert _ids(again["instructions"]["example_question_sqls"]) == _ids(
        first["instructions"]["example_question_sqls"]
    )
    edited = {**SPEC, "sample_questions": ["A new question?", *SPEC["sample_questions"][1:]]}
    ids = set(_ids(genie.serialized_space(edited, "x")["config"]["sample_questions"]))
    assert len(ids & set(_ids(first["config"]["sample_questions"]))) == 4  # only one changed


def test_every_example_reads_only_the_spaces_tables():
    allowed = {f"{genie.CATALOG}.gold.{t}" for t in SPEC["tables"]}
    for e in SPEC["examples"] + SPEC["filters"]:
        used = set(re.findall(r"__CATALOG__\.\w+\.\w+", e["sql"]))
        assert used <= allowed and (used or e in SPEC["filters"]), e


def _columns(table: str) -> set[str]:
    return set(SPEC["tables"][table]["columns"])


def test_the_score_tables_have_every_column_the_space_describes(scored):  # noqa: F811
    tables = scored[-1]
    for name in ("risk_scores", "alerts_fact", "escalations_fact", "site_kpis_hourly"):
        assert _columns(name) <= set(tables[name].columns), name


def test_the_pipeline_tables_have_every_column_the_space_describes(fitted):  # noqa: F811
    table = fitted[0]  # a patient-hour feature table, as Gold builds it
    assert _columns("patient_hour_features") <= set(table.columns)
    t0 = pd.Timestamp("2026-10-01", tz="UTC")
    readings = pd.DataFrame(
        {
            "device_id": "DEV-A-0001",
            "ts": pd.date_range(t0, periods=12, freq="5min"),
            **{v: 80.0 for v in gold.STUCK_VITALS},
            "spo2": 96.0,
            "battery_pct": 90.0,
            "firmware": "3.1.4",
        }
    )
    assignments = pd.DataFrame(
        {"device_id": ["DEV-A-0001"], "start_ts": [t0], "end_ts": [t0 + pd.Timedelta(hours=1)]}
    )
    daily = gold.device_health_daily(readings, assignments, t0, t0 + pd.Timedelta(days=1))
    assert _columns("device_health_daily") <= set(daily.columns)


class FakeApi:
    """Genie's REST API as far as publishing uses it; `spaces` pages of existing spaces."""

    def __init__(self, pages: list[list[dict]]):
        self.pages, self.calls = pages, []

    def do(self, method, path, query=None, body=None):
        self.calls.append((method, path, query, body))
        if method == "GET":
            i = int((query or {}).get("page_token", 0))
            more = {"next_page_token": str(i + 1)} if i + 1 < len(self.pages) else {}
            return {"spaces": self.pages[i], **more}
        if method == "POST" and path == "/api/2.0/genie/spaces":
            return {"space_id": "new-space"}
        return {}


def _publish(api, payload):
    return genie.publish_space(
        api, payload, "HeavDen Ward Data (staging)", "about", "wh1", "/Workspace/x/genie", "admins"
    )


def test_a_new_space_is_created_in_its_folder_and_handed_to_the_managers(payload):
    api = FakeApi([[{"space_id": "other", "title": "Something else"}]])
    message = _publish(api, payload)
    methods = [(m, p) for m, p, _, _ in api.calls]
    assert methods == [
        ("GET", "/api/2.0/genie/spaces"),
        ("POST", "/api/2.0/workspace/mkdirs"),
        ("POST", "/api/2.0/genie/spaces"),
        ("PATCH", "/api/2.0/permissions/genie/new-space"),
    ]
    create = api.calls[2][3]
    assert create["parent_path"] == "/Workspace/x/genie" and create["warehouse_id"] == "wh1"
    assert json.loads(create["serialized_space"]) == payload
    assert api.calls[3][3]["access_control_list"][0] == {
        "group_name": "admins",
        "permission_level": "CAN_MANAGE",
    }
    assert message.startswith(
        "Genie space 'HeavDen Ward Data (staging)' (new-space): created; 7 tables"
    )


def test_an_existing_space_is_found_on_any_page_and_updated_in_place(payload):
    api = FakeApi(
        [
            [{"space_id": "a", "title": "x"}],
            [{"space_id": "s1", "title": "HeavDen Ward Data (staging)"}],
        ]
    )
    assert "(s1): updated" in _publish(api, payload)
    patch = next(c for c in api.calls if c[0] == "PATCH" and "genie/spaces" in c[1])
    assert patch[1] == "/api/2.0/genie/spaces/s1" and "parent_path" not in patch[3]


def test_two_spaces_with_the_same_title_stop_the_job(payload):
    twins = [{"space_id": i, "title": "HeavDen Ward Data (staging)"} for i in ("a", "b")]
    with pytest.raises(RuntimeError, match="2 Genie spaces"):
        _publish(FakeApi([twins]), payload)
