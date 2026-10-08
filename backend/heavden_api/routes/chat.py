"""The AI Assistant tile: one agent, three tools (SQL over gold, documents, patient risk)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from heavden_api.chat.agent import run_agent, system_prompt
from heavden_api.chat.tools import Toolbox
from heavden_api.deps import AppState, state

router = APIRouter(prefix="/chat", tags=["chat"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=10)


def client_id(request: Request) -> str:
    # behind Vercel/Render the caller is the first X-Forwarded-For address
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


@router.get("/examples")
def examples(s: AppState = Depends(state)) -> dict:
    """Suggested questions, and whether live answers are available."""
    return {"live": s.llm is not None, "questions": s.recordings.questions()}


@router.post("")
def chat(body: ChatRequest, request: Request, s: AppState = Depends(state)) -> dict:
    """Answer a question with the tools. Without an LLM key, replay a recorded answer."""
    if not s.rate_limiter.allow(client_id(request)):
        raise HTTPException(429, "Too many questions; please try again later.")

    if s.llm is not None:
        toolbox = Toolbox(s, s.sandbox)
        m = s.snapshot.manifest
        system = system_prompt(m.as_of, m.bands, s.sandbox.schema_text(), m.data_start)
        try:
            result = run_agent(
                s.llm, toolbox, system, body.message, [t.model_dump() for t in body.history]
            )
        except Exception as exc:  # LLM outage, quota, network: fall back to recordings
            fallback = _recorded(s, body.message)
            fallback["notice"] = f"Live assistant unavailable ({type(exc).__name__})."
            return fallback
        return {
            "mode": "live",
            "model": s.llm.model,
            "answer": result.answer,
            "steps": result.steps,
            "citations": result.citations,
        }
    return _recorded(s, body.message)


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
