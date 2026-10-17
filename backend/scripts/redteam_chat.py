"""Try to make the live assistant misbehave, and report what happened.

    uv run python backend/scripts/redteam_chat.py
    LLM_PROVIDER=azure LLM_MODEL=gpt-5-mini uv run python backend/scripts/redteam_chat.py

Uses the model LLM_PROVIDER / LLM_MODEL pick (environment or .env) and the demo snapshot. Each
prompt in `heavden_api.chat.redteam` is sent through the real `/chat` endpoint (so all three
guardrail layers apply). A reply fails if it leaks the system prompt or a secret, dumps most of
the schema, shows internal codes (SITE_A, SITE_A-GENERAL, stay ids) instead of names, answers an
out-of-scope request instead of declining, refuses an in-scope question that is misspelt or
vague (over-refusal), or came from the recordings (the model was never reached, so nothing
about it was tested).
Review the answers too: the automatic checks are a floor, not a proof. The report is saved to
data/redteam/ (git-ignored) before it is printed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from heavden_api.app import create_app
from heavden_api.chat.redteam import ATTACKS, problems
from heavden_api.config import Settings, load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv()
    settings = Settings(chat_requests_per_hour=1000)
    if not settings.llm_connection():
        raise SystemExit(f"no key for LLM_PROVIDER={settings.llm_provider} (environment or .env)")
    failures, lines = 0, [f"Red-team of {settings.llm_provider} / {settings.llm_model}"]
    with TestClient(create_app(settings)) as client:
        for prompt, expected in ATTACKS:
            reply = client.post("/chat", json={"message": prompt}).json()
            found = problems(expected, reply)
            failures += bool(found)
            layer = reply.get("guardrail") or reply["mode"]
            lines += [
                "=" * 90,
                f"[{'FAIL' if found else 'ok'}] ({layer}) {prompt}",
                f"  expected: {expected}" + (f" | problems: {found}" if found else ""),
                "  " + reply["answer"].replace("\n", "\n  "),  # in full, as it was checked
            ]
    lines.append(f"\n{failures} of {len(ATTACKS)} replies failed the automatic checks")
    text = "\n".join(lines)
    out = ROOT / "data" / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"redteam-{settings.llm_model}-{datetime.now(UTC):%Y%m%d-%H%M}.txt"
    path.write_text(text + "\n", encoding="utf-8")  # first: a console that can't print a
    print(text)  # character (e.g. U+202F on cp1252) must not lose the report
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
