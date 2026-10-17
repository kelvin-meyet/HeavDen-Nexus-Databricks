"""Which models the assistant may run on, and how each provider's API differs (`models.yaml`).

All of them speak OpenAI's Chat Completions with tools; three things differ:
* the name of the answer-length limit: Databricks serving endpoints take `max_tokens` and reject
  `max_completion_tokens`, which OpenAI and Azure require for gpt-5 models;
* `reasoning_effort`: sent only where the model's entry gives one;
* `max_tokens`: the answer's limit, where a model needs more than the usual 1,200 (for gpt-5
  models the limit also covers the hidden reasoning, so one that thinks long needs more).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MODELS = Path(__file__).with_name("models.yaml")
PROVIDERS = ("openai", "azure", "databricks")
ASSISTANTS = ("app", "databricks")  # each has its own list
TOKEN_LIMIT = {
    "openai": "max_completion_tokens",
    "azure": "max_completion_tokens",
    "databricks": "max_tokens",
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    reasoning_effort: str | None = None
    max_tokens: int | None = None  # None: agent.MAX_ANSWER_TOKENS

    @property
    def token_limit(self) -> str:
        return TOKEN_LIMIT[self.provider]


def load_models(assistant: str = "app", path: Path = MODELS) -> dict[str, ModelProfile]:
    """The models `assistant` ("app" or "databricks") may run on."""
    if assistant not in ASSISTANTS:
        raise ValueError(f"assistant must be one of {ASSISTANTS}, not {assistant!r}")
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))[assistant]
    return {name: ModelProfile(name, **entry) for name, entry in entries.items()}


def profile(name: str, provider: str, assistant: str = "app", path: Path = MODELS) -> ModelProfile:
    """The listed model `name`; refuses one that isn't in `assistant`'s list or is served by
    another provider."""
    known = load_models(assistant, path)
    if name not in known:
        raise ValueError(
            f"LLM model {name!r} is not in the {assistant} list of {path.name}: add it there "
            f"and run the quality test on it first (listed: {', '.join(known)})"
        )
    if known[name].provider != provider:
        raise ValueError(
            f"{name!r} is served by {known[name].provider}, but LLM_PROVIDER is {provider!r}"
        )
    return known[name]
