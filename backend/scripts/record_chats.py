"""Record the example questions through the live assistant, for demo mode without an LLM key.

    uv run python backend/scripts/record_chats.py

Needs OPENAI_API_KEY (from the environment or `.env`) and the demo snapshot. Writes
`backend/heavden_api/chat/recordings.json`; review the answers before committing them.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from heavden_api.app import create_app
from heavden_api.chat.recorded import EXAMPLE_QUESTIONS, RECORDINGS
from heavden_api.config import Settings, load_dotenv


def main() -> None:
    load_dotenv()
    settings = Settings(chat_requests_per_hour=1000)
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set (environment or .env)")
    conversations = []
    with TestClient(create_app(settings)) as client:
        as_of = client.get("/health").json()["as_of"]
        for question in EXAMPLE_QUESTIONS:
            t0 = time.time()
            reply = client.post("/chat", json={"message": question}).json()
            if reply["mode"] != "live":
                raise SystemExit(f"live assistant failed on: {question} ({reply.get('notice')})")
            tools = [step["tool"] for step in reply["steps"]]
            print(f"{time.time() - t0:5.1f}s  {tools}  {question}")
            conversations.append(
                {
                    "question": question,
                    "answer": reply["answer"],
                    "steps": reply["steps"],
                    "citations": reply["citations"],
                }
            )
    RECORDINGS.write_text(
        json.dumps(
            {
                "model": settings.llm_model,
                "as_of": as_of,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "conversations": conversations,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(conversations)} conversations to {RECORDINGS}")


if __name__ == "__main__":
    main()
