"""Smoke-test a running API: what the release checks on the image before pushing it, and on the
website after deploying it. Standard library only, so it runs on any CI machine.

    python3 backend/scripts/smoke_api.py http://127.0.0.1:8000 [--snapshot <name>] [--wait 180]

Checks: /health (the snapshot named, the document index present), each page answers, the board
has patients, an unchanged what-if gives the risk the board shows for patients its model scored
(the website's two numbers for one patient), and a document search finds passages.

With ORIGIN_SECRET in the environment, every request carries it as the website's proxy does, and
a request without it must be refused (the API on Container Apps answers only the website).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PAGES = ("/analytics/summary", "/analytics/census?hours=72", "/analytics/units")
SAME = 1e-9
ORIGIN_HEADER = "x-origin-secret"  # heavden_api.config.ORIGIN_HEADER
SECRET = os.environ.get("ORIGIN_SECRET") or None


def call(
    base: str,
    path: str,
    body: dict | None = None,
    timeout: float = 120,
    secret: str | None = SECRET,
) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} | ({ORIGIN_HEADER: secret} if secret else {})
    request = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def refused_without_the_secret(base: str) -> bool:
    try:
        call(base, "/health", timeout=30, secret=None)
    except urllib.error.HTTPError as error:
        return error.code == 403
    return False


def wait_for_health(base: str, seconds: float, snapshot: str | None) -> dict:
    """/health, retried while the API starts (a cold start pulls the image and the snapshot) and,
    after a deploy, until the new revision answers: the one serving `snapshot`."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            health = call(base, "/health", timeout=30)
            if snapshot is None or health.get("snapshot") == snapshot:
                return health
            last = f"it serves snapshot {health.get('snapshot')!r}, not {snapshot!r}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last = str(error)
        if time.monotonic() > deadline:
            raise SystemExit(f"FAIL: after {seconds:.0f} s, {last}")
        time.sleep(5)


def smoke(base: str, snapshot: str | None, wait: float) -> list[str]:
    health = wait_for_health(base, wait, snapshot)
    version = health["model"]["version"]
    problems = []
    if health["status"] != "ok":
        problems.append(f"/health status {health['status']!r}")
    if not health["documents"]:
        problems.append("no document index")
    if SECRET and not refused_without_the_secret(base):
        problems.append("it answers requests without the website's secret")
    for page in PAGES:
        call(base, page)
    patients = call(base, "/risk/board?limit=50")["patients"]
    if not patients:
        problems.append("the board has no patients")
    own = [p for p in patients if str(p.get("model_version")) == str(version)][:5]
    if patients and not own:
        problems.append(f"no patient on the board was scored by v{version}, the snapshot's model")
    for p in own:
        what_if = call(base, "/risk/score", {"encounter_id": p["encounter_id"], "changes": {}})
        if abs(what_if["risk_before"] - p["risk"]) > SAME:
            problems.append(
                f"{p['encounter_id']}: board {p['risk']:.6f}, what-if {what_if['risk_before']:.6f}"
            )
    found = call(base, "/documents/search?q=escalation%20protocol%20high%20band")["results"]
    if not found:
        problems.append("a document search found nothing")
    print(
        f"{base}: {health['mode']} mode, snapshot {health.get('snapshot')}, "
        f"as of {health['as_of']}, model v{version}; {len(PAGES) + 1} pages, "
        f"{len(patients)} patients on the board, "
        f"{len(own)} what-ifs equal to the board, {len(found)} passages found"
    )
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base")
    parser.add_argument("--snapshot", help="the snapshot it must serve (waits for it)")
    parser.add_argument("--wait", type=float, default=60, help="seconds to wait for /health")
    args = parser.parse_args()
    try:
        problems = smoke(args.base, args.snapshot, args.wait)
    except urllib.error.HTTPError as error:
        problems = [f"{error.url}: HTTP {error.code} {error.read()[:300]!r}"]
    for problem in problems:
        print(f"FAIL: {problem}")
    sys.exit(1 if problems else 0)
