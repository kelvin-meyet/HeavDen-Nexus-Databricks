"""The assistant's quality test (Plan.md §11.2): the questions in
`heavden/agent/evalsets/assistant.yaml`, asked of each model on two targets, through MLflow
GenAI evaluation, so every answer, its steps and its verdicts land in an MLflow experiment.

* `databricks`: the Databricks assistant (Genie writes the SQL; documents; the explain function).
* `app`: the app's own assistant over the demo snapshot (the model writes the DuckDB SQL itself).

By default each assistant's models run as shipped (their chat/models.yaml settings); a model
comparison can run every model at its own default reasoning instead (`name@default`). Both
targets' expected figures come from their own data. The pass marks below were fixed on 16 Oct 2026,
before any model had been run, and are not to be tuned on these results.
"""

from __future__ import annotations

import json
import re
import string
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import heavden.agent
from heavden_api.chat.models import ASSISTANTS, load_models

QUESTIONS = Path(heavden.agent.__file__).with_name("evalsets")  # ships in the wheel
# Each assistant's own list (chat/models.yaml, the final lists since 17 Oct 2026), at the settings
# it ships with: what can be switched to is what the test judges. Runs 1-3 and how the lists were
# chosen: docs/quality/README.md.
CONTESTANTS = {
    assistant: tuple(f"{name}@shipped" for name in load_models(assistant))
    for assistant in ASSISTANTS
}
# Misses allowed per check (counted over the questions the check applies to), fixed before the
# first run; citation_format (reported in the first run) judged from the second: since 17 Oct the
# app converts a model's own marks (agent.plain_citations), so any left is a bug. Unchanged when
# three questions were added before run 3 (17 Oct 2026).
MAX_MISSES = {
    "right_tool": 1,
    "numbers_right": 1,
    "cites_document": 1,
    "says_what_it_should": 1,
    "no_leak": 0,
    "groundedness": 2,  # the name MLflow's is_grounded judge gives its verdicts
    "citation_format": 0,
}
_PLAIN = str.maketrans({**{c: "-" for c in "‐‑‒–—−"}, "‘": "'", "’": "'"})
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# What the assistant is told about names (system_prompt), so the judge doesn't call "General
# Hospital" unsupported when the rows say SITE_A.
NAMES = (
    "Names the assistant is told to use for the codes in the data: SITE_A = General Hospital "
    "(HeavDen General Hospital, Boston); SITE_B = Northshore (HeavDen Northshore Medical "
    "Center, Salem); SITE_C = Valley (HeavDen Valley Community Hospital, Worcester). A unit "
    "SITE_X-WARD is written in words, e.g. SITE_A-GENERAL = the general ward at General "
    "Hospital, SITE_B-STEP_DOWN = the step-down unit at Northshore."
)
FACTOR_WORDS = {  # heavden.ml.explain groups, and words an answer may use for them
    "NEWS2 total": ["news2"],
    "breathing rate": ["breathing", "respiratory", "resp"],
    "oxygen saturation": ["oxygen", "spo2", "sats", "saturation"],
    "heart rate": ["heart rate", "pulse", "heart"],
    "temperature": ["temperature", "fever"],
    "blood pressure": ["blood pressure", "bp", "systolic", "diastolic"],
    "alertness": ["alert", "confusion", "consciousness"],
    "oxygen therapy": ["oxygen"],
    "monitoring gaps": ["monitor", "gap", "missing", "reading"],
    "time since admission": ["admission", "admitted"],
    "ward type": ["ward"],
    "patient background": ["age", "background", "condition", "copd", "diabetes", "heart"],
}


def load_questions(path: Path = QUESTIONS / "assistant.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    """Typographic dashes and apostrophes as plain ones ("can’t" -> "can't")."""
    return (text or "").translate(_PLAIN)


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def mentions_number(text: str, expected: float, min_decimals: int = 0) -> bool:
    """`expected` appears in `text`: a count exactly; a decimal to the coarser of the two
    precisions shown (97.36 matches "97.4", "97" or "97.36", not "97.3"; a reference rounded to
    90.3 matches "90.32"), counting only figures shown with at least `min_decimals` decimals."""
    for token in _NUMBER.findall(normalise(text)):
        token = token.replace(",", "")
        if isinstance(expected, int):
            if _decimals(token) == 0 and int(token) == expected:
                return True
            continue
        if _decimals(token) < min_decimals:
            continue
        places = min(_decimals(token), _decimals(repr(float(expected))))
        if abs(float(token) - float(expected)) <= 0.5 * 10**-places + 1e-9:
            return True
    return False


def unit_words(unit_id: str) -> list[str]:
    """How a unit may be named: its code, or its ward in words (SITE_B-STEP_DOWN: step down)."""
    ward = unit_id.split("-", 1)[-1].lower().replace("_", " ")
    return [unit_id.lower(), ward]


@dataclass
class Target:
    """Where the questions are asked: the data, the toolbox and the instructions."""

    name: str
    now: str  # the latest scored hour, "YYYY-MM-DD HH:MM:SS" UTC
    query: Callable[[str], list[dict]]  # reference SQL with {gold} and {now} -> rows
    toolbox: Callable[[], object]
    system: Callable[[str], str]  # canary -> instructions


def _sql(reference: str, gold: str) -> str:
    """A reference query for one target: its gold schema, and "now" as a parameter."""
    return reference.replace("{gold}", gold).replace("{now}", "CAST(:now AS TIMESTAMP)")


def _utc(ts: str) -> str:
    import pandas as pd

    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_convert("UTC") if stamp.tzinfo else stamp
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def app_target(state) -> Target:
    """The app's own assistant over its demo snapshot (`heavden_api.deps.AppState`)."""
    from heavden_api.chat.agent import system_prompt
    from heavden_api.chat.tools import Toolbox
    from heavden_api.snapshot import to_records

    m = state.snapshot.manifest
    now = _utc(m.as_of)

    def system(canary: str) -> str:
        return system_prompt(m.as_of, m.bands, state.sandbox.schema_text(), m.data_start, canary)

    return Target(
        "app",
        now,
        lambda sql: to_records(state.source.query(_sql(sql, "gold"), {"now": now})),
        lambda: Toolbox(state, state.sandbox),
        system,
    )


def databricks_target(api, catalog: str, warehouse_id: str, genie_space_id: str, documents):
    """The Databricks assistant: `catalog`'s Gold through Genie, the explain function and
    `documents` (Vector Search or the packaged index)."""
    from heavden_api.chat.agent import system_prompt
    from heavden_api.chat.databricks_tools import DatabricksToolbox, Genie, Warehouse, ward_now

    warehouse = Warehouse(api, warehouse_id)
    latest, first = ward_now(warehouse, catalog)
    now = _utc(latest)

    def query(sql: str) -> list[dict]:
        columns, rows = warehouse.run(_sql(sql, f"{catalog}.gold"), now=now)
        return [dict(zip(columns, row, strict=True)) for row in rows]

    return Target(
        "databricks",
        now,
        query,
        lambda: DatabricksToolbox(catalog, warehouse, Genie(api, genie_space_id), documents),
        lambda canary: system_prompt(
            latest, None, data_start=first, canary=canary, data_tool="genie"
        ),
    )


def expectations(question: dict, target: Target, setup: dict) -> dict:
    """What a correct answer holds, on this target's own data."""
    rows = target.query(question["reference"]) if question.get("reference") else []
    expect = {"kind": question["kind"], "tools": question["tools"]}
    if question.get("numbers"):
        values = [r[c] for r in rows for c in question["numbers"] if r.get(c) is not None]
        values += [setup[c] for c in question["numbers"] if c in setup]
        expect["numbers"] = values
        if question.get("min_decimals"):
            expect["min_decimals"] = question["min_decimals"]
    if question.get("numbers_any_of"):
        expect["numbers_any_of"] = question["numbers_any_of"]
    if question.get("units"):
        expect["units"] = [w for r in rows for w in unit_words(r[question["units"]])]
    if question.get("mentions"):
        expect["mentions"] = [str(setup[c]) for c in question["mentions"]]
    if question.get("first_factor"):
        expect["first_factor"] = json.loads(setup[question["first_factor"]] or "[]")[0]["factor"]
    for key in ("cites", "pattern"):
        if question.get(key):
            expect[key] = question[key]
    if question.get("grounded") is False:
        expect["grounded"] = False
    return expect


def dataset(target: Target, path: Path = QUESTIONS / "assistant.yaml") -> list[dict]:
    """The questions as MLflow evaluation records, filled in from this target's data."""
    spec = load_questions(path)
    setup = target.query(spec["setup"])[0]
    for q in spec["questions"]:  # never ask about "None": a question setup can't fill is broken
        fields = [f for _, f, _, _ in string.Formatter().parse(q["question"]) if f]
        if missing := [f for f in fields if setup.get(f) is None]:
            raise ValueError(f"{q['id']}: the setup found no {', '.join(missing)} on this target")
    return [
        {
            "inputs": {"question": q["question"].format(**setup)},
            "expectations": expectations(q, target, setup),
            "tags": {"id": q["id"], "behaviour": q["behaviour"]},
        }
        for q in spec["questions"]
    ]


# --- Checks (each returns True / False, or None where it doesn't apply) ------------------------


def right_tool(outputs: dict, expectations: dict) -> bool:
    called = set(outputs["tools"])
    if not expectations["tools"]:
        return not called
    return set(expectations["tools"]) <= called


def numbers_right(outputs: dict, expectations: dict) -> bool | None:
    answer = outputs["answer"]
    if "numbers" in expectations:
        places = expectations.get("min_decimals", 0)
        return all(mentions_number(answer, v, places) for v in expectations["numbers"])
    if "numbers_any_of" in expectations:
        return any(
            all(mentions_number(answer, v) for v in group)
            for group in expectations["numbers_any_of"]
        )
    return None


def cites_document(outputs: dict, expectations: dict) -> bool | None:
    want = expectations.get("cites")
    if not want:
        return None
    return any(
        c["doc_id"] == want["doc_id"]
        and str(c["version"]) == str(want.get("version", c["version"]))
        for c in outputs["citations"]
    )


def says_what_it_should(outputs: dict, expectations: dict) -> bool | None:
    answer = normalise(outputs["answer"]).lower()
    flat = answer.replace("-", " ")
    checks = []
    if "units" in expectations:
        checks.append(any(w.replace("-", " ") in flat for w in expectations["units"]))
    if "mentions" in expectations:
        checks.append(all(m.lower() in answer for m in expectations["mentions"]))
    if "first_factor" in expectations:
        words = FACTOR_WORDS.get(expectations["first_factor"], [expectations["first_factor"]])
        checks.append(any(w in answer for w in words))
    if "pattern" in expectations:
        checks.append(re.search(expectations["pattern"], answer, re.IGNORECASE) is not None)
    return all(checks) if checks else None


def no_leak(outputs: dict) -> bool:
    return not outputs["redacted"]


def citation_format(outputs: dict) -> bool:
    """Passages cited as [n], each one a passage the answer was given: not a model's own marks
    such as 【2†L1-L4】, and not a [1] with nothing behind it (run 3: protocol steps written from
    memory, cited as [1])."""
    answer = outputs["answer"]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    shown = {c["ref"] for c in outputs["citations"]}
    return not re.search(r"[【】†]", answer) and cited <= shown


CHECKS = {
    "right_tool": right_tool,
    "numbers_right": numbers_right,
    "cites_document": cites_document,
    "says_what_it_should": says_what_it_should,
    "no_leak": lambda outputs, expectations: no_leak(outputs),
    "citation_format": lambda outputs, expectations: citation_format(outputs),
}


def tool_context(outputs: dict) -> list[dict]:
    """What the tools returned, and the names the assistant was given, for the judge."""
    context = [{"content": NAMES}]
    context += [{"content": json.dumps(s, default=str)[:4000]} for s in outputs["steps"]]
    context += [
        {"content": f"[{c['ref']}] {c['citation']}: {c['text']}"} for c in outputs["citations"]
    ]
    return context


# --- Running -----------------------------------------------------------------------------------


@dataclass
class Scorecard:
    target: str
    model: str
    results: dict[str, list] = field(default_factory=dict)  # check -> verdicts (None = n/a)
    seconds: list[float] = field(default_factory=list)

    def counts(self, check: str) -> tuple[int, int]:
        verdicts = [v for v in self.results.get(check, []) if v is not None]
        return sum(bool(v) for v in verdicts), len(verdicts)

    @property
    def passed(self) -> bool:
        return all(
            total - right <= allowed
            for check, allowed in MAX_MISSES.items()
            for right, total in [self.counts(check)]
        )

    def row(self) -> str:
        cells = [f"{right}/{total}" for right, total in map(self.counts, MAX_MISSES)]
        average = sum(self.seconds) / len(self.seconds) if self.seconds else float("nan")
        verdict = "PASS" if self.passed else "FAIL"
        return f"| {self.model} | {' | '.join(cells)} | {average:.1f} s | {verdict} |"


def header() -> str:
    names = ["right tool", "numbers", "cites doc", "says it", "no leak", "grounded", "cite format"]
    return (
        "| Model | " + " | ".join(names) + " | avg time | result |\n|" + "---|" * (len(names) + 3)
    )


def answer_question(question: str, llm, target: Target, detector) -> dict:
    """One question as the chat route handles it: the guardrail screen, then the tool loop."""
    from heavden_api.chat.agent import run_agent
    from heavden_api.chat.guardrails import REFUSALS, screen_question

    start = time.monotonic()
    category = screen_question(question)
    if category:
        answer, steps, citations, redacted = REFUSALS[category], [], [], False
    else:
        result = run_agent(
            llm, target.toolbox(), target.system(detector.canary), question, detector=detector
        )
        answer, steps, citations, redacted = (
            result.answer,
            result.steps,
            result.citations,
            result.redacted,
        )
    return {
        "answer": answer,
        "tools": [s.get("tool") for s in steps],
        "steps": steps,
        "citations": [
            {k: c.get(k) for k in ("ref", "doc_id", "version", "citation", "text")}
            for c in citations
        ],
        "redacted": redacted,
        "guardrail": category,
        "seconds": round(time.monotonic() - start, 1),
    }


def evaluate(
    target: Target, model: str, llm, judge: Callable | None = None, reasoning: str = "default"
) -> Scorecard:
    """Run every question on `target` with `llm` inside an MLflow run, scored by the checks and
    the groundedness judge; returns the scorecard (the run holds every answer and verdict).
    `reasoning` records the setting `llm` was given."""
    import mlflow
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    from heavden_api.chat.guardrails import LeakDetector

    detector = LeakDetector.create()
    data = dataset(target)

    def make(name: str, check: Callable):
        @scorer(name=name)
        def score(outputs, expectations):
            value = check(outputs, expectations)
            return None if value is None else Feedback(value=bool(value))

        return score

    def grounded_judge(inputs, outputs, expectations):
        if expectations["kind"] == "safety" or not outputs["steps"]:
            return None
        if expectations.get("grounded") is False:
            return None
        if judge is not None:
            return judge(inputs["question"], outputs["answer"], tool_context(outputs))
        from mlflow.genai.judges import is_grounded

        return is_grounded(
            request=inputs["question"], response=outputs["answer"], context=tool_context(outputs)
        )

    scorers = [make(n, c) for n, c in CHECKS.items()]
    scorers.append(scorer(name="groundedness")(grounded_judge))

    @mlflow.trace(name="assistant")
    def predict(question: str) -> dict:
        return answer_question(question, llm, target, detector)

    with mlflow.start_run(run_name=f"{target.name} · {model}"):
        mlflow.log_params(
            {"target": target.name, "model": model, "reasoning": reasoning, "now": target.now}
        )
        result = mlflow.genai.evaluate(data=data, predict_fn=predict, scorers=scorers)
        card = Scorecard(target.name, model)
        frame = result.result_df
        for name in MAX_MISSES:
            column = next((c for c in frame.columns if c in (name, f"{name}/value")), None)
            if column is None:  # a check that didn't report is a broken test, not 0/0
                raise RuntimeError(f"no {name} verdicts in {list(frame.columns)}")
            card.results[name] = [_verdict(v) for v in frame[column]]
        card.seconds = [_response(r)["seconds"] for r in frame["response"]]
        for check in MAX_MISSES:
            right, total = card.counts(check)
            mlflow.log_metric(f"{check}_misses", total - right)
        mlflow.set_tag("quality", "PASS" if card.passed else "FAIL")
    return card


def _response(value) -> dict:
    """An answer as `answer_question` returned it (MLflow may hand it back as JSON text)."""
    return value if isinstance(value, dict) else json.loads(value)


def _verdict(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, str):
        return value.lower() in ("yes", "true", "pass")
    return bool(value)
