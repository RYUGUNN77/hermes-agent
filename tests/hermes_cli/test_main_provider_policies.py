"""Main-provider-scoped config policy tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_BASE_CONFIG: dict[str, Any] = {
    "model": {"provider": "openai-codex", "default": "gpt-5.6-sol-900k"},
    "auxiliary": {
        "compression": {"provider": "anthropic", "model": "claude-opus-5"},
    },
    "fallback_providers": [
        {"provider": "anthropic", "model": "claude-opus-5"},
    ],
    "main_provider_policies": {
        "openrouter": {
            "model_overrides": {
                "z-ai/glm-5.3-flash": {"context_length": 1_048_576},
            },
            "provider_routing": {
                "require_parameters": True,
                "data_collection": "deny",
            },
            "auxiliary": {
                "compression": {
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-flash-0731",
                },
            },
            "fallback_providers": [
                {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"},
                {"provider": "anthropic", "model": "claude-opus-5"},
            ],
        },
    },
}


def test_matching_main_provider_policy_overlays_without_mutating_base():
    from hermes_cli.config import resolve_main_provider_policy

    resolved = resolve_main_provider_policy(
        _BASE_CONFIG, "openrouter", "z-ai/glm-5.3-flash"
    )

    assert resolved["model"]["default"] == "gpt-5.6-sol-900k"
    assert resolved["model"]["context_length"] == 1_048_576
    assert resolved["provider_routing"] == {
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert resolved["auxiliary"]["compression"] == {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-flash-0731",
    }
    assert resolved["fallback_providers"][0]["provider"] == "openai-codex"

    assert _BASE_CONFIG["auxiliary"]["compression"]["provider"] == "anthropic"
    assert "context_length" not in _BASE_CONFIG["model"]


def test_nonmatching_main_provider_leaves_base_policy_dormant():
    from hermes_cli.config import resolve_main_provider_policy

    resolved = resolve_main_provider_policy(
        _BASE_CONFIG, "openai-codex", "gpt-5.6-sol-900k"
    )

    assert resolved is _BASE_CONFIG
    assert resolved["auxiliary"]["compression"]["provider"] == "anthropic"
    assert resolved["fallback_providers"] == [
        {"provider": "anthropic", "model": "claude-opus-5"}
    ]
    assert "provider_routing" not in resolved


def test_model_override_does_not_leak_to_another_openrouter_model():
    from hermes_cli.config import resolve_main_provider_policy

    resolved = resolve_main_provider_policy(
        _BASE_CONFIG, "openrouter", "qwen/qwen3.8-flash"
    )

    assert "context_length" not in resolved["model"]
    assert resolved["provider_routing"]["require_parameters"] is True


def test_disabled_policy_is_dormant():
    from hermes_cli.config import resolve_main_provider_policy

    config = yaml.safe_load(yaml.safe_dump(_BASE_CONFIG))
    config["main_provider_policies"]["openrouter"]["enabled"] = False

    assert (
        resolve_main_provider_policy(config, "openrouter", "z-ai/glm-5.3-flash")
        is config
    )


def test_openrouter_agent_uses_matching_policy_at_runtime(tmp_path: Path, monkeypatch):
    config = {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol-900k"},
        "fallback_providers": [
            {"provider": "anthropic", "model": "claude-opus-5"},
        ],
        "main_provider_policies": _BASE_CONFIG["main_provider_policies"],
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli import config as config_module
    from run_agent import AIAgent

    config_module._LOAD_CONFIG_CACHE.clear()
    agent = AIAgent(
        provider="openrouter",
        requested_provider="openrouter",
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        quiet_mode=True,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        fallback_model=config["fallback_providers"],
    )

    assert getattr(agent, "provider_require_parameters") is True
    assert getattr(agent, "provider_data_collection") == "deny"
    assert [entry["provider"] for entry in getattr(agent, "_fallback_chain")] == [
        "openai-codex",
        "anthropic",
    ]
    assert getattr(agent, "context_compressor").context_length == 1_048_576


def test_auxiliary_policy_follows_live_main_provider(tmp_path: Path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.auxiliary_client import (
        _get_auxiliary_task_config,
        scoped_runtime_main,
    )
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    with scoped_runtime_main(
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"}
    ):
        openrouter_compression = _get_auxiliary_task_config("compression")
    with scoped_runtime_main(
        {"provider": "openai-codex", "model": "gpt-5.6-sol-900k"}
    ):
        codex_compression = _get_auxiliary_task_config("compression")

    assert openrouter_compression["provider"] == "openrouter"
    assert openrouter_compression["model"] == "deepseek/deepseek-v4-flash-0731"
    assert codex_compression["provider"] == "anthropic"
    assert codex_compression["model"] == "claude-opus-5"


def test_auto_auxiliary_uses_matching_policy_fallback_chain(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(_BASE_CONFIG, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent import auxiliary_client
    from hermes_cli import config as config_module

    config_module._LOAD_CONFIG_CACHE.clear()
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_fallback_entry",
        lambda entry: (object(), entry["model"]),
    )
    with auxiliary_client.scoped_runtime_main(
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"}
    ):
        _, model, provider = auxiliary_client._try_main_fallback_chain(
            "title_generation", failed_provider="openrouter"
        )

    assert provider == "openai-codex"
    assert model == "gpt-5.6-sol-900k"


def test_main_provider_policy_paths_are_known_config_keys():
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key(
        "main_provider_policies.openrouter.auxiliary.compression.model"
    ) == (True, None)
