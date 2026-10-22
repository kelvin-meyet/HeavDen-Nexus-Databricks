"""The assistant's quality test (heavden_api.chat.quality): its questions, its checks and the
scorecard read back from MLflow."""

import re
import string
from collections import Counter

import pytest
import yaml

from heavden_api.chat import quality as q
from heavden_api.config import Settings
from heavden_api.deps import build_state

SPEC = q.load_questions()


def test_23_questions_one_behaviour_each():
    questions = SPEC["questions"]
    assert len(questions) == 23 and len({x["id"] for x in questions}) == 23
    assert Counter(x["kind"] for x in questions) == {
        "numbers": 9,
        "documents": 6,
        "patient": 4,
        "two-tool": 2,
        "safety": 2,
    }
    for x in questions:
        assert x["behaviour"] and isinstance(x["tools"], list)
        assert (x["kind"] == "safety") == (x["tools"] == [])
        placeholders = {f for _, f, _, _ in string.Formatter().parse(x["question"]) if f}
        assert placeholders <= {"patient", "stay", "gone"}
        for sql in [x.get("reference") or ""]:
            assert set(re.findall(r"\{(\w+)\}", sql)) <= {"gold", "now"}


def test_every_reference_runs_on_the_apps_snapshot(demo_snapshot, tmp_path):
    no_llm = {"llm_provider": "openai", "openai_api_key": None}  # whatever .env picks
    state = build_state(Settings(snapshot_dir=demo_snapshot, **no_llm))
    target = q.app_target(state)
    # Nobody leaves the ward in this small snapshot, so p4 (a patient who has left) can't be
    # asked: the test stops instead of asking about "None". The rest runs without it.
    with pytest.raises(ValueError, match="p4_patient_left: the setup found no gone"):
        q.dataset(target)
    without_p4 = [x for x in SPEC["questions"] if x["id"] != "p4_patient_left"]
    path = tmp_path / "assistant.yaml"
    path.write_text(yaml.safe_dump({**SPEC, "questions": without_p4}), encoding="utf-8")
    records = q.dataset(target, path)
    assert len(records) == 22
    filled = {r["tags"]["id"]: r for r in records}
    assert "{" not in filled["p1_why_patient"]["inputs"]["question"]
    assert filled["p1_why_patient"]["expectations"]["first_factor"] in q.FACTOR_WORDS
    assert filled["n8_ward_most_high_now"]["expectations"]["units"]  # the top ward(s), named
    assert filled["n6_alert_load_valley"]["expectations"]["min_decimals"] == 1
    assert filled["p3_unknown_patient"]["expectations"]["grounded"] is False


@pytest.mark.parametrize(
    "text, expected, found",
    [
        ("General Hospital has 9 and Valley 1.", 9, True),
        ("P-1001 is flagged", 1, False),  # a count is a whole number, not part of a label
        ("about 9.5 alerts", 9, False),
        ("uptime 97.4% (97.36% exactly)", 97.36, True),
        ("uptime 97%", 97.36, True),
        ("uptime 97.3%", 97.36, False),
        ("1,204 readings", 1204, True),
        ("risk 3.8 %", 3.846, True),
        ("90.32 % flagged", 90.3, True),  # more precise than a reference rounded in SQL
        ("uptime 97.2047%", 97.2, True),
        ("uptime 97.31%", 97.2, False),
        ("2.4 alerts per nurse", 2.25, False),
        ("each of these wards has three High-risk patients", 3, True),  # a count in words
        ("Three wards are tied", 3, True),
        ("thirty patients", 3, False),
        ("someone on the ward", 1, False),
    ],
)
def test_numbers_are_matched_as_shown(text, expected, found):
    assert q.mentions_number(text, expected) is found


@pytest.mark.parametrize(
    "text, found",
    [
        ("about 2 alerts per nurse per shift (the budget)", False),  # a round budget, not a measure
        ("2.3 alerts per nurse per shift", True),
        ("2.25 alerts per nurse per shift", True),
        ("2.4 alerts per nurse per shift", False),
    ],
)
def test_a_measure_needs_its_decimals(text, found):
    assert q.mentions_number(text, 2.25, min_decimals=1) is found


def test_the_checks_judge_what_they_should():
    out = {"tools": ["query_gold"], "answer": "", "citations": [], "redacted": False, "steps": []}
    assert q.right_tool(out, {"tools": ["query_gold"]})
    assert not q.right_tool(out, {"tools": []})  # a safety question must use no tool
    assert not q.right_tool(out, {"tools": ["query_gold", "search_documents"]})
    cites = {**out, "citations": [{"doc_id": "HD-CLIN-001", "version": "1.0"}]}
    assert not q.cites_document(cites, {"cites": {"doc_id": "HD-CLIN-001", "version": "2.0"}})
    assert q.cites_document(cites, {"cites": {"doc_id": "HD-CLIN-001"}})
    words = {**out, "answer": "The step‑down unit at Northshore: SpO2 is the top reason."}
    assert q.says_what_it_should(words, {"units": q.unit_words("SITE_B-STEP_DOWN")})
    assert q.says_what_it_should(words, {"first_factor": "oxygen saturation"})
    pattern = {x["id"][:2]: x.get("pattern") for x in SPEC["questions"]}
    for retrain in [
        "No, do not retrain: it is a sensor fault.",
        "**No.** When SpO₂ drifts at a single site the runbook calls it device bias.",
    ]:
        assert q.says_what_it_should({**out, "answer": retrain}, {"pattern": pattern["d6"]})
    assert not q.says_what_it_should(
        {**out, "answer": "Yes, retrain on the new readings."}, {"pattern": pattern["d6"]}
    )
    for unknown in ["I can’t find P-9999.", "I didn’t find a record for P-9999."]:
        assert q.says_what_it_should({**out, "answer": unknown}, {"pattern": pattern["p3"]})
    for left in [
        "P-1307 is no longer on the ward: their last score was at 02:00 on 24 Sep.",
        "P-1307 isn’t on the ward right now; they left on 24 Sep.",
        "P-1307 isn't currently in the ward.",
        "P-1307 was discharged on 24 Sep, so they aren't flagged now.",
    ]:
        assert q.says_what_it_should({**out, "answer": left}, {"pattern": pattern["p4"]})
    assert not q.says_what_it_should(
        {**out, "answer": "P-1307 is flagged because SpO2 fell."}, {"pattern": pattern["p4"]}
    )
    assert q.says_what_it_should(out, {}) is None
    assert not q.citation_format({**out, "answer": "Call at 5 or more【2†L1】."})
    passage = [{"ref": 1, "doc_id": "HD-CLIN-001", "version": "2.0", "citation": "c", "text": "t"}]
    assert q.citation_format({**out, "answer": "Review within 30 min [1].", "citations": passage})
    # a [1] with no passage behind it: steps written from memory, made to look sourced (run 3)
    assert not q.citation_format({**out, "answer": "Notify the senior nurse [1]."})


def test_the_scorecard_is_read_back_from_mlflow(tmp_path, monkeypatch):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("quality-test")
    monkeypatch.setenv("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")
    rows = {
        "setup": [
            {
                "patient": "P-1001",
                "stay": "E-1",
                "risk_pct": 3.8,
                "top_factors": '[{"factor": "heart rate"}]',
                "gone": "P-2002",
            }
        ]
    }
    target = q.Target(
        "fake",
        "2026-10-05 23:00:00",
        lambda sql: (
            rows["setup"]
            if "patient_label" in sql
            else [
                {
                    "high": 4,
                    "alerts": 7,
                    "unit_id": "SITE_A-GENERAL",
                    "pct": 50.0,
                    "uptime": 97.0,
                    "alert_load": 2.25,
                    "patients": 70,
                }
            ]
        ),
        lambda: None,
        lambda canary: "system",
    )

    def canned(question, llm, target, detector):  # right tools, right figures, except one miss
        spec = next(
            x
            for x in SPEC["questions"]
            if x["question"].format(patient="P-1001", stay="E-1", gone="P-2002") == question
        )
        answer = (
            "4 7 50 97 2.25 70 3.8 22 17 general P-1001 heart rate HeavDen review, "
            "do not retrain, cannot find, no longer on the ward, can't share instructions"
        )
        if spec["id"] == "n7_informal":
            answer = "nobody"  # the one miss
        cited = [
            {
                "ref": 1,
                "doc_id": spec.get("cites", {}).get("doc_id", "X"),
                "version": spec.get("cites", {}).get("version", "1.0"),
                "citation": "c",
                "text": "t",
            }
        ]
        return {
            "answer": answer,
            "tools": spec["tools"],
            "steps": [{"tool": t} for t in spec["tools"]],
            "citations": cited,
            "redacted": False,
            "guardrail": None,
            "seconds": 1.0,
        }

    monkeypatch.setattr(q, "answer_question", canned)
    card = q.evaluate(target, "fake-model", llm=None, judge=lambda question, answer, context: True)
    assert card.counts("right_tool") == (23, 23)
    assert card.counts("numbers_right") == (11, 12)  # the one miss, allowed
    # not judged: the 2 safety questions, and p3 / p4 (their right answer denies the premise)
    assert card.counts("groundedness") == (19, 19) and card.passed
    assert card.row().endswith("| 1.0 s | PASS |") and "| fake-model |" in card.row()
    run = mlflow.search_runs(experiment_names=["quality-test"]).iloc[0]
    assert run["tags.quality"] == "PASS" and run["metrics.numbers_right_misses"] == 1
