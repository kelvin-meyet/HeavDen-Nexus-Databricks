"""The AI Assistant tile: one agent, three tools (SQL over gold, documents, patient risk).

`POST /chat` returns the whole answer; `POST /chat/stream` sends the same answer as
server-sent events while it is produced (see `chat/agent.py` for the event types). Both go
through the same checks, in this order:

1. per-visitor rate limit (429)
2. guardrail screen: obvious extraction or override attempts get a fixed refusal, no LLM call
3. no LLM key, or today's live budget used up: replay a recorded answer
4. live answer, with every answer (and stream) checked for leaks

Follow-ups: live answers carry a `conversation_id`. Sending it back continues the conversation,
which the server remembers (chat/memory.py); the client never sends history itself, so history
can't be forged. Refused and redacted turns are not remembered.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from heavden_api.chat.agent import agent_events, run_agent, system_prompt
from heavden_api.chat.guardrails import REFUSALS, screen_question
from heavden_api.chat.limits import client_address
from heavden_api.chat.memory import as_messages
from heavden_api.chat.tools import Toolbox
from heavden_api.deps import AppState, state

router = APIRouter(prefix="/chat", tags=["chat"])

RECORDED_CHUNK_WORDS = 4
RECORDED_CHUNK_DELAY = 0.012  # seconds: replays read like a live answer without dragging


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=1000)
    # from the previous live answer; omit to start a new conversation
    conversation_id: str | None = Field(None, max_length=64)


def client_id(request: Request, trusted_hops: int) -> str:
    """The visitor, from the X-Forwarded-For entry our own proxy added (see chat/limits.py)."""
    peer = request.client.host if request.client else None
    return client_address(request.headers.get("x-forwarded-for"), peer, trusted_hops)


@router.get("/examples")
def examples(s: AppState = Depends(state)) -> dict:
    """Suggested questions, and whether live answers are available."""
    return {"live": s.llm is not None, "questions": s.recordings.questions()}


def _plan(body: ChatRequest, request: Request, s: AppState) -> tuple[str, dict]:
    """Decide how to answer: ("guardrail" | "recorded" | "live", details)."""
    if not s.rate_limiter.allow(client_id(request, s.settings.trusted_proxy_hops)):
        raise HTTPException(429, "Too many questions; please try again later.")
    category = screen_question(body.message)
    if category:
        return "guardrail", {"category": category}
    if s.llm is None:
        return "recorded", {}
    if not s.live_budget.allow():
        return "recorded", {
            "notice": "Today's live-answer budget is used up; showing recorded answers."
        }
    return "live", {}


def _system(s: AppState) -> str:
    m = s.snapshot.manifest
    return system_prompt(
        m.as_of, m.bands, s.sandbox.schema_text(), m.data_start, s.leak_detector.canary
    )


def _guardrail_reply(category: str) -> dict:
    return {
        "mode": "guardrail",
        "model": None,
        "answer": REFUSALS[category],
        "steps": [],
        "citations": [],
        "guardrail": category,
    }


@router.post("")
def chat(body: ChatRequest, request: Request, s: AppState = Depends(state)) -> dict:
    """Answer a question with the tools. Without an LLM key, replay a recorded answer."""
    route, details = _plan(body, request, s)
    if route == "guardrail":
        return _guardrail_reply(details["category"])
    if route == "recorded":
        return _recorded(s, body.message) | details
    conversation_id, remembered = s.conversations.resolve(body.conversation_id)
    try:
        result = run_agent(
            s.llm,
            Toolbox(s, s.sandbox),
            _system(s),
            body.message,
            as_messages(remembered),
            detector=s.leak_detector,
        )
    except Exception as exc:  # LLM outage, quota, network: fall back to recordings
        return _recorded(s, body.message) | {
            "notice": f"Live assistant unavailable ({type(exc).__name__})."
        }
    reply = {
        "mode": "live",
        "model": s.llm.model,
        "conversation_id": conversation_id,
        "answer": result.answer,
        "steps": result.steps,
        "citations": result.citations,
    }
    if result.redacted:
        reply["guardrail"] = "leak"
    else:
        s.conversations.remember(conversation_id, body.message, result.answer, result.steps)
    return reply


@router.post("/stream")
def chat_stream(body: ChatRequest, request: Request, s: AppState = Depends(state)):
    """The same answer as `/chat`, sent as server-sent events while it is produced."""
    route, details = _plan(body, request, s)  # raises 429 before the stream starts
    if route == "guardrail":
        events = _reply_events(_guardrail_reply(details["category"]), chunked=False)
    elif route == "recorded":
        events = _reply_events(_recorded(s, body.message) | details, chunked=True)
    else:
        events = _live_events(body, s)
    return StreamingResponse(
        (f"data: {json.dumps(e, default=str)}\n\n" for e in events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _live_events(body: ChatRequest, s: AppState) -> Iterator[dict]:
    conversation_id, remembered = s.conversations.resolve(body.conversation_id)
    yield {
        "type": "meta",
        "mode": "live",
        "model": s.llm.model,
        "conversation_id": conversation_id,
    }
    started = False
    try:
        for event in agent_events(
            s.llm,
            Toolbox(s, s.sandbox),
            _system(s),
            body.message,
            as_messages(remembered),
            detector=s.leak_detector,
        ):
            started = True
            if event["type"] == "done":
                if event["redacted"]:
                    event["guardrail"] = "leak"
                else:
                    s.conversations.remember(
                        conversation_id, body.message, event["answer"], event["steps"]
                    )
                event["conversation_id"] = conversation_id
            yield event
    except Exception as exc:
        if started:  # part of an answer is already on screen: say it stopped
            yield {"type": "error", "message": "The live assistant stopped unexpectedly."}
            return
        yield from _reply_events(
            _recorded(s, body.message)
            | {"notice": f"Live assistant unavailable ({type(exc).__name__})."},
            chunked=True,
        )


def _reply_events(reply: dict, chunked: bool) -> Iterator[dict]:
    """A finished reply (recorded or refusal) as the same events a live answer produces."""
    meta = {"type": "meta", "mode": reply["mode"], "model": reply.get("model")}
    for key in ("notice", "guardrail"):
        if reply.get(key):
            meta[key] = reply[key]
    yield meta
    for index, step in enumerate(reply.get("steps", [])):
        yield {"type": "step", "index": index, "step": step}
    words = reply["answer"].split(" ")
    size = RECORDED_CHUNK_WORDS if chunked else len(words)
    for start in range(0, len(words), size):
        text = " ".join(words[start : start + size])
        yield {"type": "delta", "text": text if start == 0 else " " + text}
        if chunked:
            time.sleep(RECORDED_CHUNK_DELAY)
    done = {
        "type": "done",
        "answer": reply["answer"],
        "steps": reply.get("steps", []),
        "citations": reply.get("citations", []),
    }
    if reply.get("suggestions"):
        done["suggestions"] = reply["suggestions"]
    yield done


def _recorded(s: AppState, message: str) -> dict:
    match = s.recordings.find(message)
    if match:
        return {"mode": "recorded", "model": s.recordings.model, **match}
    return {
        "mode": "recorded",
        "model": s.recordings.model,
        "answer": "Live answers aren't switched on for this demo. Try one of the recorded "
        "questions below.",
        "steps": [],
        "citations": [],
        "suggestions": s.recordings.questions(),
    }
