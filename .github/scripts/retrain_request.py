"""Open a retrain request for a drift check that reached "act": the last job of a data run
(prod-run, or staging-run for a rehearsal).

One open request per kind of drift:

* an open request already shows this kind of drift (the check's `act_kinds`, e.g. performance)
  -> open nothing. A request may be held for days until enough outcomes exist, and every check
  run meanwhile would otherwise queue a duplicate;
* the open request shows a different kind (say a sensor fault nobody decided on, and now the
  outcomes changed) -> open a new request with the newer evidence, then cancel the old one, so
  the reviewer is emailed and a retrain is recorded against the right check.

Requests started by hand and retrains already running are never cancelled. If the open requests
can't be listed, a request is opened anyway: a duplicate is better than a missed alert.

Staging and prod requests are the same workflow with a different `target`; a staging request
has "(staging)" in its name, and each environment only looks at its own requests.

Standard library only (it runs on the Actions runner's own Python). Needs `gh`, GH_TOKEN with
`actions: write`, CHECK (the JSON after DRIFT_CHECK_RESULT), TARGET (prod or staging),
GITHUB_REPOSITORY, GITHUB_REF_NAME.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

WORKFLOW = "retrain.yml"
KINDS_IN_TITLE = re.compile(r", act: (.+)$")
WAITING = ("waiting", "pending", "queued", "requested")  # not yet approved: safe to replace
STAGING_MARK = " (staging):"  # in the name of a request for the staging environment


def title(check: dict) -> str:
    """The request's name on GitHub: the episode, the check and the kinds of drift it shows.
    `heavden.ml.model_ops` reads the episode and the check back from it for the audit trail."""
    episode = f"episode {check['episode']}, " if check.get("episode") else ""
    return f"{episode}check {check['check_id']}, act: {' '.join(check.get('act_kinds') or [])}"


def kinds_in(run_title: str) -> set[str] | None:
    """The kinds of drift a request was opened for (None: it was started by hand)."""
    found = KINDS_IN_TITLE.search(run_title)
    return set(found.group(1).split()) if found else None


def for_target(runs: list[dict], target: str) -> list[dict]:
    """The requests that belong to `target` (prod or staging)."""
    return [r for r in runs if (STAGING_MARK in r["displayTitle"]) == (target == "staging")]


def decide(open_runs: list[dict], kinds: list[str]) -> tuple[str, list[dict]]:
    """("skip", [the request that already shows these kinds]) or ("open", [requests the new one
    replaces]). `open_runs` are the retrain workflow's runs that have not completed."""
    for run in open_runs:
        shown = kinds_in(run["displayTitle"])
        if kinds and shown is not None and set(kinds) <= shown:
            return "skip", [run]
    replaced = [
        run
        for run in open_runs
        if run["status"] in WAITING and kinds_in(run["displayTitle"]) is not None
    ]
    return "open", replaced


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def say(line: str) -> None:
    print(line)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as summary:
            summary.write(line + "\n\n")


def link(run: dict) -> str:
    return f"[{run['displayTitle']}]({run['url']})"


def main() -> None:
    raw = os.environ["CHECK"]
    check = json.loads(raw)
    repo, ref = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_REF_NAME"]
    target = os.environ.get("TARGET", "prod")
    try:
        fields = "databaseId,status,url,displayTitle"
        listed = gh(
            "run", "list", "--repo", repo, "--workflow", WORKFLOW, "--limit", "20", "--json", fields
        )
        open_runs = [run for run in json.loads(listed) if run["status"] != "completed"]
        open_runs = for_target(open_runs, target)
    except (subprocess.CalledProcessError, ValueError) as error:
        print(f"::warning::could not list the open retrain requests ({error}); opening one anyway")
        open_runs = []

    action, runs = decide(open_runs, check.get("act_kinds") or [])
    if action == "skip":
        say(
            f"A retrain request for this kind of drift is already open, so no second one was "
            f"opened: {link(runs[0])}. This check is recorded in `monitoring.drift_checks`."
        )
        return
    name = title(check)
    inputs = ["-f", f"check={raw}", "-f", f"title={name}", "-f", f"target={target}"]
    gh("workflow", "run", WORKFLOW, "--repo", repo, "--ref", ref, *inputs)
    say(
        f"Retrain request opened ({name}): Actions -> Drift alert - retrain request. "
        "It waits for approval; GitHub emails the reviewer."
    )
    for run in runs:  # only once the new request exists, so there is never none
        try:
            gh("run", "cancel", str(run["databaseId"]), "--repo", repo)
            say(f"It replaces {link(run)}, which showed a different kind of drift.")
        except subprocess.CalledProcessError as error:
            say(f"The older request {link(run)} was not cancelled ({error}): cancel it by hand.")


if __name__ == "__main__":
    main()
