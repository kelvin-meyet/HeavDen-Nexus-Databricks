"""The tool-calling loop: the LLM picks tools, we run them, until it writes an answer.

The LLM sits behind a small interface (`LLM.complete`), so tests use a scripted fake and live
mode can later swap in the Databricks agent endpoint. Only text turns of the conversation are
kept between requests; tool calls are re-done each turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from heavden_api.chat.tools import TOOL_SPECS, Toolbox

MAX_ROUNDS = 6


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as the LLM wrote it


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    model: str

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...


class OpenAILLM:
    """OpenAI Chat Completions with function calling."""

    # "none": Chat Completions only allows function tools with reasoning off for gpt-5.x models;
    # it is also the fastest and cheapest setting
    def __init__(self, api_key: str, model: str, reasoning_effort: str | None = "none"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=60, max_retries=2)
        self.model = model
        self.reasoning_effort = reasoning_effort

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, **extra
        )
        message = response.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, c.function.arguments) for c in message.tool_calls or []
        ]
        return LLMReply(message.content, calls)


@dataclass
class ChatAnswer:
    answer: str
    steps: list[dict]
    citations: list[dict]
    rounds: int


def system_prompt(as_of: str, bands: dict, schema: str, data_start: str | None = None) -> str:
    return f"""You are the HeavDen-Nexus assistant for staff of a FICTIONAL hospital network.
All patients and data are synthetic. Data covers {data_start or "the last 14 days"} to now.
"Now" is {as_of} (UTC); use this timestamp instead of
now() or current_date in SQL, written as TIMESTAMPTZ '{as_of}'. Timestamps are UTC.

Choose tools by question type:
- numbers, counts, rates, trends, comparisons -> query_gold (DuckDB SQL over gold tables)
- protocols, NEWS2, devices, safety notices, drift procedures, the model and how well it
  performs (the model card has the evaluation; don't compute your own metrics)
  -> search_documents
- why one patient is (or isn't) high risk -> explain_patient (find the patient's label with
  query_gold first if the question only describes them, e.g. "the highest-risk patient")
Some questions need two tools (e.g. a patient's risk and what the protocol says to do).

Sites (use the codes in SQL): SITE_A = HeavDen General Hospital (Boston), SITE_B = HeavDen
Northshore Medical Center (Salem), SITE_C = HeavDen Valley Community Hospital (Worcester).

SQL rules: one SELECT; only the gold.* tables below; aggregate rather than dump rows; round
numbers; risk is a probability (0-1), so show it as a percentage. An alert is a patient
entering the High band (risk >= {bands["high"]:.4f}); Medium is risk >= {bands["medium"]:.4f}.
For daily figures, count only complete days (00:00-24:00 UTC) and say which days; the first
and last day of a window are often partial. Compute differences and percentages in SQL; never
do arithmetic in your head. If a query fails, read the error, fix the SQL and try again.

Tables:
{schema}

Answer rules:
- Be brief and concrete. Answer only what was asked; don't add extra comparisons.
- Check the question's premise against the data. If the data doesn't show what the question
  assumes (e.g. a problem that isn't there right now), say so plainly.
- Call a tool only when you need its result; never run placeholder queries.
- Use only numbers returned by tools (or quoted in documents); never invent or recompute them.
- Cite document facts with the passage number, e.g. [2].
- If the tools don't answer the question, say so plainly.
- Do not give clinical advice beyond what the protocols say; you support, not replace,
  clinical judgement.
"""


def run_agent(
    llm: LLM,
    toolbox: Toolbox,
    system: str,
    question: str,
    history: list[dict] | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> ChatAnswer:
    messages = [{"role": "system", "content": system}]
    messages += [m for m in history or [] if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": question})
    steps: list[dict] = []
    for round_ in range(1, max_rounds + 1):
        reply = llm.complete(messages, TOOL_SPECS)
        if not reply.tool_calls:
            answer = (reply.content or "").strip() or "I couldn't produce an answer."
            return ChatAnswer(answer, steps, toolbox.citations, round_)
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in reply.tool_calls
                ],
            }
        )
        for call in reply.tool_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                result, step = (
                    json.dumps({"error": "arguments were not valid JSON"}),
                    {
                        "tool": call.name,
                        "error": True,
                    },
                )
            else:
                result, step = toolbox.call(call.name, arguments)
            steps.append(step)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return ChatAnswer(
        "I couldn't finish within the allowed number of steps. Try a narrower question.",
        steps,
        toolbox.citations,
        max_rounds,
    )
