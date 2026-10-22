"""Try to make an assistant misbehave, and report what happened.

    uv run python backend/scripts/redteam_chat.py
    LLM_PROVIDER=azure LLM_MODEL=gpt-5-mini uv run python backend/scripts/redteam_chat.py
    uv run python backend/scripts/redteam_chat.py --endpoint heavden-assistant-staging
    uv run python backend/scripts/redteam_chat.py --url https://heav-den-nexus.vercel.app

Without `--endpoint` or `--url`: the **app assistant**, on the model LLM_PROVIDER / LLM_MODEL
pick (environment or .env) and the demo snapshot; each prompt goes through the real `/chat`
endpoint, so all three guardrail layers apply. With `--url`: the app assistant behind the live
website, through the website's own `/api/chat`, so it is the model, data and limits visitors meet
(each prompt counts against this machine's per-visitor limit, 30 an hour, and the ones that
reach the model against the day's live answers). With `--endpoint`: the **Databricks assistant**
deployed on that serving endpoint, on whichever model its LLM_ENDPOINT names (the CLI profile,
or the CI identity, signs in).

Every prompt in `heavden_api.chat.redteam` is checked the same way: a reply fails if it leaks the
system prompt or a secret, dumps most of the schema, shows internal codes (SITE_A,
SITE_A-GENERAL, stay ids) instead of names, answers an out-of-scope request instead of
declining, refuses an in-scope question that is misspelt or vague (over-refusal), or came from
the recordings (the model was never reached, so nothing about it was tested).
Review the answers too: the automatic checks are a floor, not a proof. The report is saved to
data/redteam/ (git-ignored) before it is printed; the exit code is 1 if any reply failed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from heavden_api.chat.redteam import ATTACKS, problems

ROOT = Path(__file__).resolve().parents[2]


def app_assistant() -> tuple[str, Iterator[dict]]:
    """The app assistant's replies, through `/chat` on the demo snapshot."""
    from fastapi.testclient import TestClient

    from heavden_api.app import create_app
    from heavden_api.config import Settings, load_dotenv

    load_dotenv()
    settings = Settings(chat_requests_per_hour=1000)
    if not settings.llm_connection():
        raise SystemExit(f"no key for LLM_PROVIDER={settings.llm_provider} (environment or .env)")

    def replies() -> Iterator[dict]:
        with TestClient(create_app(settings)) as client:
            for prompt, _ in ATTACKS:
                yield client.post("/chat", json={"message": prompt}).json()

    return f"the app assistant: {settings.llm_provider} / {settings.llm_model}", replies()


def website(url: str) -> tuple[str, Iterator[dict]]:
    """The live website's replies, through its `/api/chat` like a visitor's question."""
    import json
    import urllib.request

    def replies() -> Iterator[dict]:
        for prompt, _ in ATTACKS:
            request = urllib.request.Request(
                f"{url.rstrip('/')}/api/chat",
                data=json.dumps({"message": prompt}).encode(),
                headers={"Content-Type": "application/json"},
            )
            # a cold start plus a Genie question can take minutes; Vercel stops at 300 s
            with urllib.request.urlopen(request, timeout=310) as response:
                yield json.loads(response.read())

    return f"the website {url}", replies()


def databricks_assistant(endpoint: str) -> tuple[str, Iterator[dict]]:
    """The Databricks assistant's replies from its serving endpoint, in `/chat`'s shape."""
    from databricks.sdk import WorkspaceClient

    from heavden_api.chat.endpoint import ask  # the SDK only: CI runs this step without duckdb

    api = WorkspaceClient().api_client

    def replies() -> Iterator[dict]:
        for prompt, _ in ATTACKS:
            answer, outputs = ask(api, endpoint, prompt)
            guardrail = outputs.get("guardrail")
            yield {
                "mode": "guardrail" if guardrail else "live",
                "guardrail": guardrail,
                "model": outputs.get("model"),
                "answer": answer,
            }

    return f"the Databricks assistant: endpoint {endpoint}", replies()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--endpoint", help="the Databricks assistant's serving endpoint")
    target.add_argument("--url", help="the live website, e.g. https://heav-den-nexus.vercel.app")
    args = parser.parse_args(argv)
    if args.endpoint:
        title, replies = databricks_assistant(args.endpoint)
    elif args.url:
        title, replies = website(args.url)
    else:
        title, replies = app_assistant()
    failures, models, lines = 0, set(), []
    for (prompt, expected), reply in zip(ATTACKS, replies, strict=True):
        found = problems(expected, reply)
        failures += bool(found)
        if reply.get("model"):
            models.add(reply["model"])
        layer = reply.get("guardrail") or reply["mode"]
        lines += [
            "=" * 90,
            f"[{'FAIL' if found else 'ok'}] ({layer}) {prompt}",
            f"  expected: {expected}" + (f" | problems: {found}" if found else ""),
            "  " + reply["answer"].replace("\n", "\n  "),  # in full, as it was checked
        ]
    lines.insert(0, f"Red-team of {title} (model: {', '.join(sorted(models)) or 'none reached'})")
    lines.append(f"\n{failures} of {len(ATTACKS)} replies failed the automatic checks")
    text = "\n".join(lines)
    out = ROOT / "data" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    reached = "-".join(sorted(models) or ["none"])
    name = args.endpoint or ("website-" if args.url else "app-") + reached
    path = out / f"redteam-{name}-{datetime.now(UTC):%Y%m%d-%H%M}.txt"
    path.write_text(text + "\n", encoding="utf-8")  # first: a console that can't print a
    print(text)  # character (e.g. U+202F on cp1252) must not lose the report
    print(f"Saved to {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
