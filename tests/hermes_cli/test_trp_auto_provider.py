from __future__ import annotations

import pytest
import yaml


@pytest.mark.parametrize("provider,unknown", [("trp-auto", False), ("not-a-real-provider", True)])
def test_doctor_recognizes_trp_auto_without_accepting_unknown_providers(tmp_path, provider, unknown):
    from hermes_cli.doctor_config import _validate_model_config

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"provider": provider, "default": "TRP_AUTO"}}))
    issues = []
    _validate_model_config(path, issues)
    assert any("is unknown" in issue for issue in issues) is unknown


def test_trp_auto_is_a_builtin_selectable_virtual_provider_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **_kwargs: [])

    from hermes_cli.inventory import ConfigContext, build_models_payload
    from hermes_cli.models import CANONICAL_PROVIDERS

    entries = [entry for entry in CANONICAL_PROVIDERS if entry.slug == "trp-auto"]
    assert [(entry.label, entry.tui_desc) for entry in entries] == [
        ("TRP_AUTO", "TRP_AUTO (virtual subscription-seat router)")
    ]

    payload = build_models_payload(
        ConfigContext("", "", "", {}, []), explicit_only=True, picker_hints=True
    )
    rows = [row for row in payload["providers"] if row["slug"] == "trp-auto"]
    assert rows == [{
        "slug": "trp-auto",
        "name": "TRP_AUTO",
        "is_current": False,
        "is_user_defined": False,
        "models": ["TRP_AUTO"],
        "total_models": 1,
        "source": "virtual",
        "authenticated": True,
        "auth_type": "virtual",
        "warning": None,
    }]


def test_selecting_trp_auto_persists_virtual_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n",
        encoding="utf-8",
    )

    from hermes_cli.model_switch import persist_model_selection, switch_model

    result = switch_model(
        "TRP_AUTO",
        current_provider="openai-codex",
        current_model="gpt-5.6-sol",
        explicit_provider="trp-auto",
    )

    assert result.success is True
    assert (result.target_provider, result.new_model) == ("trp-auto", "TRP_AUTO")
    persist_model_selection(result, config_path)
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    assert (saved["provider"], saved["default"]) == ("trp-auto", "TRP_AUTO")


def test_runtime_provider_honors_disabled_trp_auto_before_virtual_resolution(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider._config_mod,
        "load_config",
        lambda: {"providers": {"trp-auto": {"enabled": False}}},
    )
    calls = []
    monkeypatch.setattr(
        "hermes_cli.trp_auto_bridge.resolve_bootstrap_runtime",
        lambda target: calls.append(target) or {},
    )

    with pytest.raises(ValueError, match="providers.trp-auto.enabled: false"):
        runtime_provider.resolve_runtime_provider(
            requested="trp-auto", target_model="TRP_AUTO"
        )
    assert calls == []


def test_runtime_provider_bootstraps_trp_auto_to_a_physical_runtime(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class SeatDecision:\n"
        "    provider: str\n"
        "    model: str\n"
        "    reason: str\n"
        "    switch: bool\n"
        "    frontier: bool\n"
        "    control_digest: str\n"
        "def bootstrap_seat():\n"
        "    return SeatDecision('openai-codex', 'gpt-6-sol', "
        "'canonical bootstrap seat', True, False, 'abc123')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge
    from hermes_cli.runtime_provider import resolve_runtime_provider

    calls = []

    def physical(provider, model):
        calls.append((provider, model))
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "subscription-token",
            "source": "device_code",
            "requested_provider": "openai-codex",
        }

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", physical)
    runtime = resolve_runtime_provider(requested="trp-auto", target_model="TRP_AUTO")

    assert calls == [("openai-codex", "gpt-6-sol")]
    assert runtime["provider"] == "openai-codex"
    assert runtime["requested_provider"] == "trp-auto"
    assert runtime["trp_auto"] == {
        "virtual_provider": "trp-auto",
        "virtual_model": "TRP_AUTO",
        "bootstrap_provider": "openai-codex",
        "bootstrap_model": "gpt-6-sol",
        "reason": "canonical bootstrap seat",
        "frontier": False,
        "control_digest": "abc123",
    }


def test_physical_runtime_must_match_the_bootstrap_seat(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='openai-codex', model='gpt-6-sol', "
        "reason='canonical', switch=True, frontier=False, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "_resolve_physical_runtime",
        lambda *_args: {"provider": "anthropic", "api_mode": "anthropic_messages"},
    )
    with pytest.raises(bridge.TrpAutoBootstrapError, match="did not match bootstrap provider"):
        bridge.resolve_bootstrap_runtime("TRP_AUTO")


def test_grok_seat_maps_to_xai_oauth_device_code_runtime(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='xai', model='grok-4.7', "
        "reason='canonical', switch=True, frontier=True, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    calls = []

    def physical(provider, model):
        calls.append((provider, model))
        return {
            "provider": "xai-oauth",
            "api_mode": "codex_responses",
            "api_key": "subscription-token",
            "source": "device_code",
        }

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", physical)

    runtime = bridge.resolve_bootstrap_runtime("TRP_AUTO")

    assert calls == [("xai-oauth", "grok-4.7")]
    assert runtime["provider"] == "xai-oauth"
    assert runtime["requested_provider"] == "trp-auto"


def test_static_credential_source_is_rejected(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='openai-codex', model='gpt-6-sol', "
        "reason='canonical', switch=True, frontier=False, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "_resolve_physical_runtime",
        lambda *_args: {
            "provider": "openai-codex",
            "api_key": "static-token",
            "source": "env",
        },
    )

    with pytest.raises(bridge.TrpAutoBootstrapError, match="credential source"):
        bridge.resolve_bootstrap_runtime("TRP_AUTO")


@pytest.mark.parametrize(
    ("provider", "source"),
    [
        ("OPENAI-CODEX", "device_code"),
        ("openai-codex", "device_code "),
    ],
)
def test_provider_and_credential_source_require_exact_allowlist_values(
    tmp_path, monkeypatch, provider, source
):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='openai-codex', model='gpt-6-sol', "
        "reason='canonical', switch=True, frontier=False, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "_resolve_physical_runtime",
        lambda *_args: {
            "provider": provider,
            "api_key": "subscription-token",
            "source": source,
        },
    )

    with pytest.raises(bridge.TrpAutoBootstrapError):
        bridge.resolve_bootstrap_runtime("TRP_AUTO")


def test_missing_trp_auto_plugin_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.trp_auto_bridge import TrpAutoBootstrapError

    with pytest.raises(TrpAutoBootstrapError, match="plugin is missing seat.py"):
        resolve_runtime_provider(requested="trp-auto", target_model="TRP_AUTO")


def test_trp_auto_virtual_recursion_fails_closed(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "def bootstrap_seat():\n"
        "    from hermes_cli.trp_auto_bridge import resolve_bootstrap_runtime\n"
        "    return resolve_bootstrap_runtime('TRP_AUTO')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.trp_auto_bridge import TrpAutoBootstrapError, resolve_bootstrap_runtime

    with pytest.raises(TrpAutoBootstrapError, match="recursion"):
        resolve_bootstrap_runtime("TRP_AUTO")


def test_malformed_trp_auto_decision_fails_closed(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='', model='gpt-6-sol')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.trp_auto_bridge import TrpAutoBootstrapError, resolve_bootstrap_runtime

    with pytest.raises(TrpAutoBootstrapError, match="invalid provider"):
        resolve_bootstrap_runtime("TRP_AUTO")


def test_malformed_trp_auto_metadata_is_rejected_before_physical_resolution(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='openai-codex', model='gpt-6-sol', "
        "reason='', frontier=False, switch=True, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    calls = []
    monkeypatch.setattr(bridge, "_resolve_physical_runtime", lambda *args: calls.append(args) or {})
    with pytest.raises(bridge.TrpAutoBootstrapError, match="invalid reason"):
        bridge.resolve_bootstrap_runtime("TRP_AUTO")
    assert calls == []


def test_trp_auto_rejects_openrouter_bootstrap_decision(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='openrouter', model='paid/model', "
        "reason='bad', switch=True, frontier=False, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.trp_auto_bridge import TrpAutoBootstrapError, resolve_bootstrap_runtime

    with pytest.raises(TrpAutoBootstrapError, match="non-physical provider 'openrouter'"):
        resolve_bootstrap_runtime("TRP_AUTO")


def test_unresolvable_subscription_seat_fails_without_leaking_details(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def bootstrap_seat():\n"
        "    return SimpleNamespace(provider='openai-codex', model='gpt-6-sol', "
        "reason='canonical', switch=True, frontier=False, control_digest='digest')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    def fail(_provider, _model):
        raise RuntimeError("secret-token-value")

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", fail)
    with pytest.raises(bridge.TrpAutoBootstrapError) as exc_info:
        bridge.resolve_bootstrap_runtime("TRP_AUTO")
    assert "could not resolve subscription seat openai-codex/gpt-6-sol" in str(exc_info.value)
    assert "secret-token-value" not in str(exc_info.value)
