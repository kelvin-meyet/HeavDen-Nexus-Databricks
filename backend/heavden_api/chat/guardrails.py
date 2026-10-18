"""Guardrails for the public assistant, in three independent layers.

1. `screen_question`, before the LLM: cheap, deterministic patterns for obvious attempts to
   extract the system prompt, configuration, credentials or source code, or to override the
   instructions. These get a fixed, helpful refusal and never reach the LLM (and cost nothing).
2. The system prompt (`agent.system_prompt`) sets the scope, forbids revealing or paraphrasing
   its instructions, and marks text inside tool results as data, not instructions.
3. `LeakDetector`, after the LLM: checks every answer (also while it streams) for a canary
   marker hidden in the system prompt, distinctive instruction phrases and secret-key patterns.
   A leaking answer is replaced by a refusal.

`plain_names`, also after the LLM, is a tidy-up rather than a guard: site and unit codes a model
writes anyway (some copy them from the rows) become the names the instructions ask for. The
Databricks assistant applies it to its answer, which it sends once whole.

None of this makes leaks impossible: a paraphrase can slip through. The real protection is
that the prompt holds nothing secret: no keys, no internal URLs. Keys live only in the server's
environment, never in the prompt.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

REPO_URL = "https://github.com/kelvin-meyet/HeavDen-Nexus-Databricks"

REFUSALS = {
    "instructions": (
        "I can't share my instructions or how I'm configured. I can answer questions about the "
        "wards' data, the hospital's documents, or why a patient is flagged."
    ),
    "secrets": (
        "I can't help with keys, passwords or system settings. I can answer questions about the "
        "wards' data, the hospital's documents, or why a patient is flagged."
    ),
    "code": (
        "I can't share code from here, but the whole project is open source: see the "
        f"[HeavDen-Nexus repository on GitHub]({REPO_URL}), and the *Behind the scenes* page "
        "explains how it's built. I can answer questions about the wards' data, the hospital's "
        "documents, or why a patient is flagged."
    ),
    "leak": (
        "I can't share that. I can answer questions about the wards' data, the hospital's "
        "documents, or why a patient is flagged."
    ),
}

# Each pattern is specific enough not to catch ordinary ward questions (tested in test_chat.py).
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}"
            r"\b(instructions?|rules|prompt|guidelines|guardrails|restrictions)\b",
            re.I,
        ),
    ),
    ("instructions", re.compile(r"\bsystem[\s_-]*(prompt|message|instructions?)\b", re.I)),
    (
        "instructions",
        re.compile(
            r"\b(reveal|show|print|repeat|output|dump|display|list|tell me|what are|give me)\b"
            r"[^.?!]{0,30}\b(your|the hidden|the system)\s+(instructions|prompt|rules|"
            r"configuration|config|guidelines|tool definitions)\b",
            re.I,
        ),
    ),
    ("instructions", re.compile(r"\b(jailbreak|developer mode|debug mode|DAN mode)\b", re.I)),
    (
        "secrets",
        re.compile(
            r"\b(api[\s_-]?keys?|secret keys?|access tokens?|passwords?|credentials?|"
            r"env(ironment)? variables?|\.env\b|openai[_\s-]?key)\b",
            re.I,
        ),
    ),
    (
        "code",
        re.compile(
            r"\b(source[\s_-]*code|backend[\s_-]*codes?|your code|the code (behind|for) (you|this)|"
            r"implementation details|python code (behind|for) (you|this))\b",
            re.I,
        ),
    ),
]


def screen_question(text: str) -> str | None:
    """The refusal category for an obvious extraction or override attempt, else None."""
    for category, pattern in _PATTERNS:
        if pattern.search(text):
            return category
    return None


# Phrases that only appear in the system prompt; an answer quoting them is leaking it.
PROMPT_MARKERS = (
    "choose tools by question type",
    "sql rules:",
    "answer rules:",
    "use this timestamp instead of",
    "never do arithmetic in your head",
    "confidentiality:",
    "untrusted content:",
)
_SECRET = re.compile(
    r"\b(sk-(proj-)?[A-Za-z0-9_-]{20,}|(AZURE_)?OPENAI_(API_)?KEY|Bearer\s+[A-Za-z0-9._-]{20,})"
)


@dataclass
class LeakDetector:
    """Flags answers that quote the system prompt or contain secrets."""

    canary: str

    @classmethod
    def create(cls) -> LeakDetector:
        return cls(canary=f"hdx-{secrets.token_hex(6)}")

    def leaks(self, text: str) -> bool:
        lowered = re.sub(r"\s+", " ", text.lower())  # the prompt wraps lines mid-phrase
        if self.canary.lower() in lowered or _SECRET.search(text):
            return True
        return sum(marker in lowered for marker in PROMPT_MARKERS) >= 1


SITE_NAMES = {"A": "General Hospital", "B": "Northshore", "C": "Valley"}
WARD_NAMES = {
    "GENERAL": "general ward",
    "STEP_DOWN": "step-down unit",
    "RESPIRATORY": "respiratory ward",
}
# SITE_A or SITE_A-GENERAL, with any hyphen a model may write (U+2010-2013), maybe in backticks
_CODE = (
    r"`?\bSITE_(?P<site>[ABC])"
    r"(?:[-\u2010-\u2013](?P<ward>GENERAL|STEP_DOWN|RESPIRATORY))?\b`?"
)
# ... in brackets, maybe labelled: "(SITE_A-GENERAL)", "(unit `SITE_A-GENERAL`)", "(id: SITE_A)"
_CODE_IN_BRACKETS = re.compile(rf"\s*\(\s*(?:(?i:unit|site|ward|code|id)\b[\s:]*)*{_CODE}\s*\)")
_CODE_ALONE = re.compile(_CODE)


def plain_names(answer: str) -> str:
    """The answer with site and unit codes in words. A code in brackets after its name is
    dropped ("the general ward at General Hospital (SITE_A-GENERAL)"); any other code becomes
    the name (SITE_B-STEP_DOWN -> "the step-down unit at Northshore")."""
    return _CODE_ALONE.sub(_name, _CODE_IN_BRACKETS.sub("", answer))


def _name(code: re.Match) -> str:
    site = SITE_NAMES[code["site"]]
    name = f"the {WARD_NAMES[code['ward']]} at {site}" if code["ward"] else site
    before = code.string[: code.start()].rstrip(" *_#>|-")  # list marks and bold don't count
    return name[0].upper() + name[1:] if not before or before[-1] in ".!?\n" else name
