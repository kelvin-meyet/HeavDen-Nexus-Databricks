"""Conversation memory for follow-up questions, kept on the server.

The browser only sends a conversation id; the server keeps what was said. That matters twice:

* **Better follow-ups.** Each remembered turn keeps a compact note of what the tools did (the
  SQL and the rows it returned, the documents found, the patient looked up), so "show that by
  day" or "why is she flagged?" can be resolved, not just the answer's wording.
* **No forged history.** If the client sent the history, anyone could invent "assistant" turns
  that carry instructions. Here only turns the server itself produced are remembered.

In memory, bounded: at most `max_conversations` (least recently used dropped first), idle ones
expire after `ttl_seconds`, and only the last `max_turns` turns are kept. A restart forgets
everything; the next question simply starts a new conversation.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

ID_LENGTH = 32  # hex characters
NOTE_MAX_CHARS = 1500
ROWS_IN_NOTE = 8


@dataclass
class Remembered:
    question: str
    answer: str
    note: str  # what the tools did, for resolving follow-ups


@dataclass
class Conversation:
    turns: list[Remembered] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)


def tool_note(steps: list[dict]) -> str:
    """A compact record of a turn's tool calls (kept out of the visible answer)."""
    lines = []
    for step in steps:
        tool = step.get("tool")
        if tool == "query_gold" and step.get("sql") and not step.get("error"):
            rows = step.get("rows", [])[:ROWS_IN_NOTE]
            lines.append(
                f"query_gold ({step.get('purpose', '')}): {' '.join(step['sql'].split())} "
                f"-> columns {step.get('columns')} rows {rows}"
            )
        elif tool == "search_documents":
            lines.append(f"search_documents: {step.get('query')!r}")
        elif tool == "explain_patient" and isinstance(step.get("result"), dict):
            r = step["result"]
            lines.append(
                f"explain_patient: {r.get('patient_label')} (stay {r.get('encounter_id')}, "
                f"{r.get('unit_id')}), risk {r.get('risk')}, band {r.get('risk_band')}"
            )
    note = "\n".join(lines)
    return note[:NOTE_MAX_CHARS] + (" ..." if len(note) > NOTE_MAX_CHARS else "")


class ConversationStore:
    def __init__(
        self,
        max_conversations: int = 500,
        ttl_seconds: float = 1800,
        max_turns: int = 6,
        clock=time.monotonic,
    ):
        self.max_conversations = max_conversations
        self.ttl = ttl_seconds
        self.max_turns = max_turns
        self.clock = clock
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def valid_id(conversation_id: str | None) -> bool:
        return (
            isinstance(conversation_id, str)
            and len(conversation_id) == ID_LENGTH
            and all(c in "0123456789abcdef" for c in conversation_id)
        )

    def resolve(self, conversation_id: str | None) -> tuple[str, list[Remembered]]:
        """The id to use and the turns remembered so far (a fresh id if unknown or expired)."""
        with self._lock:
            self._expire()
            if self.valid_id(conversation_id) and conversation_id in self._items:
                conversation = self._items[conversation_id]
                conversation.last_used = self.clock()
                self._items.move_to_end(conversation_id)
                return conversation_id, list(conversation.turns)
            return secrets.token_hex(ID_LENGTH // 2), []

    def remember(self, conversation_id: str, question: str, answer: str, steps: list[dict]) -> None:
        with self._lock:
            conversation = self._items.get(conversation_id) or Conversation()
            conversation.turns.append(Remembered(question, answer, tool_note(steps)))
            conversation.turns = conversation.turns[-self.max_turns :]
            conversation.last_used = self.clock()
            self._items[conversation_id] = conversation
            self._items.move_to_end(conversation_id)
            while len(self._items) > self.max_conversations:
                self._items.popitem(last=False)

    def _expire(self) -> None:
        now = self.clock()
        for key in [k for k, c in self._items.items() if now - c.last_used > self.ttl]:
            del self._items[key]

    def __len__(self) -> int:
        return len(self._items)


def as_messages(turns: list[Remembered]) -> list[dict]:
    """Remembered turns as chat messages. The tool note rides along with each answer, marked as
    context, so the model can resolve references without the user seeing it."""
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.question})
        content = turn.answer
        if turn.note:
            content += f"\n\n(Context from this turn, for follow-up questions:\n{turn.note})"
        messages.append({"role": "assistant", "content": content})
    return messages
