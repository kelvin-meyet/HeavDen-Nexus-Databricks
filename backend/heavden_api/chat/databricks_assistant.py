"""The Databricks assistant: the app assistant's tool loop, instructions and three guardrail
layers, with the tools served by Databricks (Genie, the documents, `ml.explain_patient_risk`), as
an MLflow `ResponsesAgent` on Model Serving (endpoint `heavden-assistant-<env>`, deployed by
`heavden_api.chat.assistant_deploy`).

Settings are environment variables on the serving endpoint (`SETTINGS`):

    LLM_ENDPOINT          the model: an endpoint in the `databricks:` list of chat/models.yaml
    HEAVDEN_CATALOG       heavden_<env>
    HEAVDEN_WAREHOUSE_ID  the SQL warehouse the patient function and "now" run on
    HEAVDEN_GENIE_SPACE   the Genie space's id
    HEAVDEN_VECTOR_INDEX  optional: the Vector Search index; else the packaged document index

A reply is the answer as text, each tool call and its result as function-call items, and
`custom_outputs` {"model", "steps", "citations", "redacted", "guardrail"} for the live app.
The endpoint keeps no conversation: earlier turns come with each request.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from heavden_api.chat.agent import LLM, OpenAILLM, agent_events, system_prompt
from heavden_api.chat.databricks_tools import (
    DatabricksToolbox,
    Genie,
    VectorSearchDocuments,
    Warehouse,
    packaged_documents,
    ward_now,
)
from heavden_api.chat.guardrails import REFUSALS, LeakDetector, plain_names, screen_question
from heavden_api.chat.models import profile

SETTINGS = {
    "llm_endpoint": "LLM_ENDPOINT",
    "catalog": "HEAVDEN_CATALOG",
    "warehouse_id": "HEAVDEN_WAREHOUSE_ID",
    "genie_space": "HEAVDEN_GENIE_SPACE",
    "vector_index": "HEAVDEN_VECTOR_INDEX",
}
NOW_SECONDS = 600  # "now" (the latest scored hour) moves only with a data run: ask every 10 min


@dataclass(frozen=True)
class Settings:
    llm_endpoint: str
    catalog: str
    warehouse_id: str
    genie_space: str
    vector_index: str | None = None

    @classmethod
    def from_env(cls, env=os.environ) -> Settings:
        missing = [v for k, v in SETTINGS.items() if k != "vector_index" and not env.get(v)]
        if missing:
            raise RuntimeError(f"the Databricks assistant needs {', '.join(missing)}")
        return cls(**{k: env.get(v) or None for k, v in SETTINGS.items()})


@dataclass
class Services:
    """What one serving process builds once: the model, the tools' clients, the leak canary."""

    llm: LLM
    warehouse: Warehouse
    genie: Genie
    documents: object
    detector: LeakDetector


def connect(settings: Settings, documents_dir: str | None) -> Services:
    """The model and the tools, signed in as the endpoint's own identity (Model Serving passes
    credentials for the resources the model was logged with)."""
    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient()
    api = workspace.api_client
    model = profile(settings.llm_endpoint, "databricks", assistant="databricks")

    def token() -> str:  # fetched before each LLM call: Databricks tokens expire
        return workspace.config.authenticate()["Authorization"].removeprefix("Bearer ")

    llm = OpenAILLM(
        token,
        model.name,
        model.reasoning_effort,
        f"{workspace.config.host.rstrip('/')}/serving-endpoints",
        model.token_limit,
        **({"max_tokens": model.max_tokens} if model.max_tokens else {}),
    )
    packaged = packaged_documents(api, documents_dir)
    documents = (
        VectorSearchDocuments(api, settings.vector_index, packaged.chunks)
        if settings.vector_index
        else packaged
    )
    return Services(
        llm,
        Warehouse(api, settings.warehouse_id),
        Genie(api, settings.genie_space),
        documents,
        LeakDetector.create(),
    )


def question_and_history(messages: list[dict]) -> tuple[str, list[dict]]:
    """The last user message, and the text turns before it (tool items are not kept)."""
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("the request must end with a user message")
    return turns[-1]["content"], turns[:-1]


class TracedToolbox(DatabricksToolbox):
    """Each tool call as a span of the request's trace, between the LLM calls: Genie's question,
    SQL and rows, the passages found, the patient's explanation."""

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        with mlflow.start_span(name=name, span_type=SpanType.TOOL) as span:
            span.set_inputs(arguments)
            result, step = super().call(name, arguments)
            span.set_outputs(step)
            if step.get("error"):  # the LLM is told and carries on; the trace shows it in red
                span.set_status("ERROR")
        return result, step


class DatabricksAssistant(ResponsesAgent):
    def __init__(self, settings: Settings | None = None, services: Services | None = None):
        self._settings, self._services = settings, services
        self._documents_dir: str | None = None
        self._lock = threading.Lock()
        self._now: tuple[float, tuple[str, str]] | None = None

    def load_context(self, context) -> None:
        self._documents_dir = (context.artifacts or {}).get("documents")

    def _ready(self) -> tuple[Settings, Services]:
        """Built on the first request, when the endpoint's credentials exist."""
        with self._lock:
            if self._settings is None:
                self._settings = Settings.from_env()
            if self._services is None:
                self._services = connect(self._settings, self._documents_dir)
        return self._settings, self._services

    def _ward_now(self, settings: Settings, services: Services) -> tuple[str, str]:
        if self._now is None or time.monotonic() - self._now[0] > NOW_SECONDS:
            self._now = (time.monotonic(), ward_now(services.warehouse, settings.catalog))
        return self._now[1]

    def _events(self, request: ResponsesAgentRequest) -> Iterator[dict]:
        """The app's agent events for one turn; a guardrail refusal is one "done" event."""
        question, history = question_and_history(self.prep_msgs_for_cc_llm(request.input))
        category = screen_question(question)
        if category:  # layer 1: obvious attacks are refused before any model is called
            yield {"type": "done", "answer": REFUSALS[category], "steps": [], "citations": []}
            yield {"type": "guardrail", "category": category}
            return
        settings, services = self._ready()
        mlflow.update_current_trace(tags={"model": settings.llm_endpoint})  # filter traces by it
        latest, first = self._ward_now(settings, services)
        system = system_prompt(
            latest, None, data_start=first, canary=services.detector.canary, data_tool="genie"
        )
        toolbox = TracedToolbox(
            settings.catalog, services.warehouse, services.genie, services.documents
        )
        yield from agent_events(services.llm, toolbox, system, question, history, services.detector)

    def predict_stream(self, request: ResponsesAgentRequest) -> Iterator[ResponsesAgentStreamEvent]:
        guardrail, done = None, None
        for event in self._events(request):
            if event["type"] == "guardrail":
                guardrail = event["category"]
            elif event["type"] == "step":
                call_id = uuid.uuid4().hex
                step = event["step"]
                inputs = {k: v for k, v in step.items() if k in ("question", "query", "patient")}
                for item in (
                    self.create_function_call_item(
                        uuid.uuid4().hex, call_id, step.get("tool", ""), json.dumps(inputs)
                    ),
                    self.create_function_call_output_item(call_id, json.dumps(step, default=str)),
                ):
                    yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event["type"] == "done":
                done = event
        if done is None:
            raise RuntimeError("the assistant finished without an answer")
        # the answer is sent once whole: a model may write text, then decide to call tools after
        # all, and a sent token can't be taken back (and codes it wrote can still become names)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(plain_names(done["answer"]), uuid.uuid4().hex),
            custom_outputs={
                "model": self._settings.llm_endpoint if guardrail is None else None,
                "steps": done["steps"],
                "citations": done["citations"],
                "redacted": done.get("redacted", False),
                "guardrail": guardrail,
            },
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        events = list(self.predict_stream(request))
        return ResponsesAgentResponse(
            output=[e.item for e in events], custom_outputs=events[-1].custom_outputs
        )
