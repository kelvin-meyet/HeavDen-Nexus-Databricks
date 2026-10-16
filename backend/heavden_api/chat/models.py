"""Which models the assistant may run on, and how each provider's API differs (`models.yaml`).

All of them speak OpenAI's Chat Completions with tools; two things differ:
* the name of the answer-length limit: Databricks serving endpoints take `max_tokens` and reject
  `max_completion_tokens`, which OpenAI and Azure require for gpt-5 models;
* `reasoning_effort`: sent only where the model's entry gives one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MODELS = Path(__file__).with_name("models.yaml")
PROVIDERS = ("openai", "azure", "databricks")
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

    @property
    def token_limit(self) -> str:
        return TOKEN_LIMIT[self.provider]


def load_models(path: Path = MODELS) -> dict[str, ModelProfile]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]
    return {name: ModelProfile(name, **entry) for name, entry in entries.items()}


def profile(name: str, provider: str, path: Path = MODELS) -> ModelProfile:
    """The listed model `name`; refuses one that isn't listed or is served by another provider."""
    known = load_models(path)
    if name not in known:
        raise ValueError(
            f"LLM model {name!r} is not in {path.name}: add it there and run the quality test "
            f"on it first (listed: {', '.join(known)})"
        )
    if known[name].provider != provider:
        raise ValueError(
            f"{name!r} is served by {known[name].provider}, but LLM_PROVIDER is {provider!r}"
        )
    return known[name]
