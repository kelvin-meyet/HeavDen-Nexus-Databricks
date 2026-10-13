import pytest

from heavden_api.chat.agent import OpenAILLM
from heavden_api.config import Settings, azure_base_url
from heavden_api.deps import build_state

ENDPOINT = "https://my-resource.openai.azure.com"


def _settings(**overrides) -> Settings:
    # every provider setting is given, so the developer's own .env can't leak into a test
    base = {
        "llm_provider": "openai",
        "openai_api_key": None,
        "azure_openai_endpoint": None,
        "azure_openai_key": None,
        "llm_model": "some-model",
        "llm_reasoning_effort": "none",
    }
    return Settings(**(base | overrides))


@pytest.mark.parametrize(
    "pasted",
    [
        ENDPOINT,
        ENDPOINT + "/",
        ENDPOINT + "/openai/v1/",
        ENDPOINT + "/openai/deployments/gpt/chat/completions?api-version=2025-01-01-preview",
        "my-resource.openai.azure.com",
    ],
)
def test_any_form_of_the_azure_endpoint_gives_the_v1_address(pasted):
    assert azure_base_url(pasted) == ENDPOINT + "/openai/v1/"


def test_the_provider_decides_which_key_and_address_are_used():
    both = {"openai_api_key": "openai-key", "azure_openai_key": "azure-key"}
    openai = _settings(**both, azure_openai_endpoint=ENDPOINT).llm_connection()
    assert (openai.api_key, openai.base_url) == ("openai-key", None)
    azure = _settings(llm_provider="azure", **both, azure_openai_endpoint=ENDPOINT).llm_connection()
    assert (azure.api_key, azure.base_url) == ("azure-key", ENDPOINT + "/openai/v1/")


def test_a_provider_without_its_key_means_recorded_answers():
    assert _settings(azure_openai_key="azure-key").llm_connection() is None  # openai chosen
    assert _settings(llm_provider="azure", openai_api_key="openai-key").llm_connection() is None


def test_misconfiguration_is_reported_not_guessed():
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        _settings(llm_provider="azure", azure_openai_key="azure-key").llm_connection()
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        _settings(llm_provider="foundry").llm_connection()


def test_keys_never_appear_when_settings_are_printed():
    settings = _settings(openai_api_key="openai-secret", azure_openai_key="azure-secret")
    text = repr(settings) + repr(settings.llm_connection())
    assert "openai-secret" not in text and "azure-secret" not in text


def test_the_assistant_is_built_for_the_chosen_provider(demo_snapshot):
    settings = _settings(
        snapshot_dir=demo_snapshot,
        llm_provider="azure",
        azure_openai_endpoint=ENDPOINT,
        azure_openai_key="azure-key",
        llm_model="my-deployment",
        llm_reasoning_effort=None,
    )
    llm = build_state(settings).llm  # building the client makes no network call
    assert isinstance(llm, OpenAILLM) and llm.model == "my-deployment"
    assert str(llm.client.base_url) == ENDPOINT + "/openai/v1/"
    assert "reasoning_effort" not in llm._options()
    assert build_state(_settings(snapshot_dir=demo_snapshot)).llm is None
