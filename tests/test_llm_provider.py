import pytest

from heavden_api import config, deps
from heavden_api.chat import models
from heavden_api.chat.agent import MAX_ANSWER_TOKENS, OpenAILLM
from heavden_api.chat.models import ASSISTANTS, PROVIDERS, load_models
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
        "databricks_host": None,
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
        llm_model="gpt-5-mini",
        llm_reasoning_effort=None,
    )
    llm = build_state(settings).llm  # building the client makes no network call
    assert isinstance(llm, OpenAILLM) and llm.model == "gpt-5-mini"
    assert str(llm.client.base_url) == ENDPOINT + "/openai/v1/"
    # the fallback thinks at its default (no reasoning_effort sent), with room to answer: at
    # 1,200 tokens it used them all thinking, twice (quality run 2)
    assert llm._options() == {"max_completion_tokens": 4000}
    assert OpenAILLM("k", "m")._options()["max_completion_tokens"] == MAX_ANSWER_TOKENS
    assert build_state(_settings(snapshot_dir=demo_snapshot)).llm is None


def test_a_databricks_endpoint_is_called_with_a_fresh_token_and_its_own_settings(
    demo_snapshot, monkeypatch, tmp_path
):
    # the app's own list has no Databricks endpoint today: give it one
    listed = tmp_path / "models.yaml"
    listed.write_text("app:\n  an-endpoint: {provider: databricks, reasoning_effort: low}\n")
    monkeypatch.setattr(deps, "profile", lambda *a: models.profile(*a, path=listed))
    tokens = iter(["token-1", "token-2"])
    monkeypatch.setattr(config, "databricks_token", lambda: next(tokens))
    settings = _settings(
        snapshot_dir=demo_snapshot,
        llm_provider="databricks",
        databricks_host="adb-1.azuredatabricks.net/",
        llm_model="an-endpoint",
        llm_reasoning_effort=None,
    )
    llm = build_state(settings).llm  # no token is fetched until the first request
    assert str(llm.client.base_url) == "https://adb-1.azuredatabricks.net/serving-endpoints/"
    assert llm._options() == {"max_tokens": MAX_ANSWER_TOKENS, "reasoning_effort": "low"}
    llm._refresh()
    assert llm.client.api_key == "token-1"
    llm._refresh()  # every request asks again: Databricks tokens expire
    assert llm.client.api_key == "token-2"
    with pytest.raises(ValueError, match="DATABRICKS_HOST"):
        _settings(llm_provider="databricks").llm_connection()


def test_only_listed_models_and_their_own_provider_are_accepted(demo_snapshot):
    keyed = {"snapshot_dir": demo_snapshot, "openai_api_key": "openai-key"}
    with pytest.raises(ValueError, match="not in the app list of models.yaml"):
        build_state(_settings(**keyed, llm_model="gpt-4-something"))
    with pytest.raises(ValueError, match="served by databricks"):
        build_state(_settings(**keyed, llm_model="databricks-gpt-oss-120b"))
    with pytest.raises(ValueError, match="served by azure"):
        build_state(_settings(**keyed, llm_model="gpt-5-mini"))
    assert all(m.provider in PROVIDERS for a in ASSISTANTS for m in load_models(a).values())


def test_each_assistant_switches_between_its_own_final_list():
    assert set(load_models("app")) == {"gpt-5.5", "gpt-5-mini", "databricks-gpt-oss-120b"}
    assert set(load_models("databricks")) == {"databricks-gpt-oss-120b", "gpt-5-mini", "gpt-5.5"}
    # GPT-OSS at low wrote protocol steps from memory (quality run 3): its default, never low
    for assistant in ASSISTANTS:
        assert load_models(assistant)["databricks-gpt-oss-120b"].reasoning_effort is None
    assert load_models("app")["gpt-5.5"].reasoning_effort == "none"  # the main model


def test_the_reasoning_effort_setting_overrides_the_models_own(demo_snapshot):
    keyed = {"snapshot_dir": demo_snapshot, "openai_api_key": "k", "llm_model": "gpt-5.5"}
    assert build_state(_settings(**keyed, llm_reasoning_effort=None)).llm.reasoning_effort == "none"
    assert build_state(_settings(**keyed, llm_reasoning_effort="low")).llm.reasoning_effort == "low"
    assert build_state(_settings(**keyed, llm_reasoning_effort="")).llm.reasoning_effort is None
