"""Ask a deployed Databricks assistant one question, through its serving endpoint or one of its
versions, and read the reply back as the answer text and `custom_outputs`.

Nothing here but the Databricks SDK client passed in: red-teaming an endpoint
(`backend/scripts/redteam_chat.py --endpoint`) runs in CI with the default packages only, without
the app's SQL engine (18 Oct: importing the deploy code there failed on `duckdb`).
"""

from __future__ import annotations


def answer_of(response: dict) -> tuple[str, dict]:
    """The answer text and `custom_outputs` of a reply from the endpoint."""
    texts = [
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    return "".join(texts), response.get("custom_outputs") or {}


def ask(api, endpoint: str, question: str, served_model: str | None = None) -> tuple[str, dict]:
    """Ask the endpoint (whichever version has its traffic) or one of its versions by name."""
    path = f"/serving-endpoints/{endpoint}"
    if served_model:
        path += f"/served-models/{served_model}"
    response = api.do(
        "POST", f"{path}/invocations", body={"input": [{"role": "user", "content": question}]}
    )
    return answer_of(response)
