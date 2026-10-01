import pytest
from run.runtime_client_settings import search_client_settings
from run.runtime_client_settings import configuration_client_settings, intent_client_settings


def test_role_specific_defaults_are_not_merged():
    config = configuration_client_settings({}, system_prompt="config")
    intent = intent_client_settings({}, system_prompt="intent")
    assert config["thinking"] == "disabled"
    assert config["model"] == "deepseek-v4-pro"
    assert "reasoning_max_tokens" not in config
    assert intent["max_tokens"] == 120
    assert intent["model"] == "deepseek-flash"
    assert intent["system_prompt"] == "intent"
    assert configuration_client_settings({"reasoning_max_tokens": "ignored"}, system_prompt="x")


def test_search_defaults_and_explicit_overrides():
    defaults = search_client_settings({})
    assert defaults["model"] == "deepseek-v4-pro"
    assert defaults["routine_max_tokens"] == 1600
    assert defaults["reasoning_max_tokens"] == 8192
    settings = {"model": "custom", "max_tokens": "2400", "thinking": "disabled"}
    output = search_client_settings(settings)
    assert output["model"] == "custom"
    assert output["max_tokens"] == 2400
    assert output["thinking"] == "disabled"
    assert settings["max_tokens"] == "2400"
    assert "api_key" not in output


def test_invalid_budget_is_not_silently_replaced():
    with pytest.raises(ValueError):
        search_client_settings({"max_tokens": "invalid"})
