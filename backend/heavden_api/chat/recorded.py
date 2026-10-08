"""Recorded conversations: what `/chat` serves when no LLM key is configured.

`backend/scripts/record_chats.py` runs the example questions through the live assistant and
saves question, answer, tool steps and citations here. Without a key (or if the LLM call fails)
the API replays a recording when the question matches one, and otherwise offers the recorded
questions instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RECORDINGS = Path(__file__).with_name("recordings.json")

# Questions shown in the app; also the set that gets recorded.
EXAMPLE_QUESTIONS = (
    "How many patients are in the High risk band at each site right now?",
    "Which unit raised the most alerts in the last 24 hours?",
    "What share of escalations in the last 7 days were flagged in advance, by site?",
    "How did daily alerts change over the last week?",
    "What should a nurse do when a patient's risk band turns High?",
    "What NEWS2 score triggers a rapid response call, and how did that change in November?",
    "If SpO2 readings drift down at one site only, should we retrain the model?",
    "How much better than NEWS2 is the model, and what are its limitations?",
    "Why is the highest-risk patient flagged right now, and what does the protocol say to do?",
    "Is device uptime at Site B different from the other sites this week?",
)


def normalise(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", question.lower()).strip()


@dataclass
class Recordings:
    model: str | None = None
    as_of: str | None = None
    recorded_at: str | None = None
    conversations: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = RECORDINGS) -> Recordings:
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def find(self, question: str) -> dict | None:
        key = normalise(question)
        return next((c for c in self.conversations if normalise(c["question"]) == key), None)

    def questions(self) -> list[str]:
        return [c["question"] for c in self.conversations] or list(EXAMPLE_QUESTIONS)
