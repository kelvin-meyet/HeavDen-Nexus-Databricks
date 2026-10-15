"""The AI/BI dashboards as code (databricks/src/dashboards/*.lvdash.json)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from heavden.monitoring.drift_check import THRESHOLDS

DASHBOARDS = sorted(
    (Path(__file__).parents[1] / "databricks" / "src" / "dashboards").glob("*.lvdash.json")
)
# Widget field expressions the deployed dashboards use; anything else is checked in the UI first
EXPRESSION = re.compile(
    r'^(`[^`]+`|(SUM|AVG|COUNT|MAX|MIN|MEASURE)\(`[^`]+`\)|DATE_TRUNC\("DAY", `[^`]+`\))$'
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sql(dashboard: dict) -> dict[str, str]:
    return {d["name"]: "".join(d["queryLines"]) for d in dashboard["datasets"]}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_every_widget_reads_a_dataset_with_expressions_the_dashboards_support(path):
    dashboard = _load(path)
    datasets = set(_sql(dashboard))
    names = []
    for page in dashboard["pages"]:
        assert page["pageType"] in {"PAGE_TYPE_CANVAS", "PAGE_TYPE_GLOBAL_FILTERS"}
        for item in page["layout"]:
            widget = item["widget"]
            names.append(widget["name"])
            for query in widget.get("queries", []):
                assert query["query"]["datasetName"] in datasets
                for field in query["query"]["fields"]:
                    assert EXPRESSION.match(field["expression"]), field["expression"]
    assert len(names) == len(set(names))  # widget names are unique


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_queries_name_no_environment_so_each_target_shows_its_own(path):
    for name, sql in _sql(_load(path)).items():
        assert "heavden_" not in sql, name  # the catalog comes from the bundle's dataset_catalog


def test_the_model_and_drift_limits_are_the_drift_checks():
    sql = _sql(_load(next(p for p in DASHBOARDS if p.stem == "model_drift.lvdash")))
    with_limits = [s for s in sql.values() if "AS limits(signal, warn_limit, act_limit)" in s]
    assert len(with_limits) == 5  # one per chart of signals
    for text in with_limits:
        found = {
            signal: (float(warn), None if act == "NULL" else float(act))
            for signal, warn, act in re.findall(r"\('(\w+)', ([\d.]+), ([\d.]+|NULL)\)", text)
        }
        expected = {s: (t.warn, None if np.isinf(t.act) else t.act) for s, t in THRESHOLDS.items()}
        assert found == expected


WARN, ACT = "#F2C200", "#D62728"


def test_model_and_drift_names_sites_as_the_web_app_does_and_colours_the_limits():
    path = next(p for p in DASHBOARDS if p.stem == "model_drift.lvdash")
    text = path.read_text(encoding="utf-8")
    web = (Path(__file__).parents[1] / "web" / "lib" / "format.ts").read_text(encoding="utf-8")
    sites = dict(re.findall(r'(SITE_\w): \{ name: "([^"]+)"', web))  # the full names
    labels = {"SITE_A": "Site-A-GH", "SITE_B": "Site-B-NS", "SITE_C": "Site-C-VL"}
    assert set(sites) == set(labels)
    for site, label in labels.items():
        assert f"'{label}'" in text, label  # what the queries name the site
        assert f"**{label}** {sites[site]}" in text, sites[site]  # the legend's full name

    charts = {
        item["widget"]["name"]: item["widget"]["spec"]
        for page in _load(path)["pages"]
        for item in page["layout"]
        if item["widget"].get("spec", {}).get("widgetType") in {"line", "bar"}
    }
    for name, spec in charts.items():
        colours = {m["value"]: m["color"] for m in spec["encodings"]["color"]["scale"]["mappings"]}
        limits = {"Warn limit": WARN, "Act limit": ACT}
        if name in {"js", "alerts", "psi", "performance", "quality"}:  # yellow to look, red to act
            assert {k: colours[k] for k in limits} == limits, name
        others = {k: v for k, v in colours.items() if k not in limits}
        assert others and not {WARN, ACT} & set(others.values()), name  # no line looks like one
        if "All sites" in colours:
            assert colours["All sites"] == "#0072B2", name  # the same colour on every page
