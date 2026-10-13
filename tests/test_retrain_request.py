import importlib.util
import json
from pathlib import Path

import pytest

from heavden.ml import model_ops

SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "retrain_request.py"
spec = importlib.util.spec_from_file_location("retrain_request", SCRIPT)
request = importlib.util.module_from_spec(spec)
spec.loader.exec_module(request)

CHECK = "cc6d54c8-3c99-4135-bee0-722140851436"
SENSOR = ["alert-load@SITE_B", "inputs@SITE_B", "predictions@SITE_B"]
PROTOCOL = ["performance"]


def _run(run_id: int, kinds: list[str] | None, status: str = "waiting") -> dict:
    check = {"episode": "some-episode", "check_id": CHECK, "act_kinds": kinds}
    name = "started by hand" if kinds is None else request.title(check)
    return {
        "databaseId": run_id,
        "status": status,
        "url": f"https://github.com/o/r/actions/runs/{run_id}",
        "displayTitle": f"Retrain request: {name}",
    }


def test_the_title_carries_what_the_audit_trail_and_the_next_check_need():
    check = {"episode": "protocol-change", "check_id": CHECK, "act_kinds": PROTOCOL}
    name = f"Retrain request: {request.title(check)}"
    assert name == f"Retrain request: episode protocol-change, check {CHECK}, act: performance"
    assert request.kinds_in(name) == {"performance"}
    review = {
        "workflow": "retrain",
        "run_id": 1,
        "title": name,
        "run_url": "u",
        "decided_at": "2026-10-13T01:55:50Z",
        "environment": "retrain-approval",
        "actor": "kelvin-meyet",
        "comment": "wait a week",
    }
    row = model_ops.rejection_rows("c.ml.m", [review], set())[0]  # the audit trail reads it back
    assert (row["check_id"], row["episode"]) == (CHECK, "protocol-change")
    no_episode = request.title({"episode": None, "check_id": CHECK, "act_kinds": SENSOR})
    assert no_episode == f"check {CHECK}, act: {' '.join(SENSOR)}"
    assert request.kinds_in("Retrain request: started by hand") is None


def test_no_second_request_while_one_for_the_same_drift_is_held():
    held = _run(1, PROTOCOL)
    assert request.decide([held], PROTOCOL) == ("skip", [held])
    wider = _run(2, [*PROTOCOL, "alert-load@SITE_B"])
    assert request.decide([wider], PROTOCOL) == ("skip", [wider])  # already covered
    assert request.decide([], PROTOCOL) == ("open", [])


def test_a_different_kind_of_drift_replaces_the_open_request():
    sensor = _run(1, SENSOR)
    assert request.decide([sensor], PROTOCOL) == ("open", [sensor])
    # the same problem, now with something new reaching act: newer evidence, new request
    assert request.decide([_run(2, PROTOCOL)], [*PROTOCOL, "alert-load@SITE_B"])[0] == "open"


def test_requests_started_by_hand_and_running_retrains_are_never_cancelled():
    by_hand = _run(1, None)
    running = _run(2, SENSOR, status="in_progress")
    assert request.decide([by_hand, running], PROTOCOL) == ("open", [])
    assert request.decide([running], SENSOR) == ("skip", [running])  # being retrained already


def test_a_check_without_kinds_always_opens_a_request():
    """A missed alert is worse than a duplicate request."""
    assert request.decide([_run(1, SENSOR)], [])[0] == "open"


@pytest.mark.parametrize(
    ("open_runs", "kinds", "calls"),
    [
        ([], PROTOCOL, ["list", "workflow"]),
        ([_run(1, PROTOCOL)], PROTOCOL, ["list"]),
        ([_run(1, SENSOR)], PROTOCOL, ["list", "workflow", "cancel"]),
        ("the listing fails", PROTOCOL, ["list", "workflow"]),
    ],
)
def test_main_opens_before_it_cancels(monkeypatch, tmp_path, open_runs, kinds, calls):
    made = []

    def fake_gh(*args):
        made.append(args[1] if args[0] == "run" else args[0])
        if args[:2] == ("run", "list"):
            if isinstance(open_runs, str):
                raise request.subprocess.CalledProcessError(1, "gh")
            return json.dumps(open_runs)
        if args[0] == "workflow":
            assert f"title={request.title(check)}" in args and f"check={raw}" in args
        return ""

    check = {"episode": "protocol-change", "check_id": CHECK, "act_kinds": kinds}
    raw = json.dumps(check)
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(request, "gh", fake_gh)
    monkeypatch.setenv("CHECK", raw)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    request.main()
    assert made == calls
    text = summary.read_text(encoding="utf-8")
    assert ("already open" in text) == (calls == ["list"])
    assert ("replaces" in text) == ("cancel" in calls)
