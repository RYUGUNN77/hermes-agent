from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from hermes_cli.model_switch import ModelSwitchResult


class _SessionDB:
    def __init__(self):
        self.model_updates = []
        self.config_updates = []

    def update_session_model(self, session_id, model):
        self.model_updates.append((session_id, model))

    def patch_session_model_config(self, session_id, config):
        self.config_updates.append((session_id, config))


class _Agent:
    def __init__(self):
        self.model = "gpt-5.6-sol"
        self.provider = "openai-codex"
        self.requested_provider = "openai-codex"
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
        }
        self.calls = []

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]
        self.requested_provider = kwargs["new_provider"]
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
        }


class _CLI(CLIModelSwitchMixin):
    def __init__(self):
        self.model = "gpt-5.6-sol"
        self.provider = "openai-codex"
        self.requested_provider = "openai-codex"
        self.api_key = "old-subscription"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.api_mode = "codex_responses"
        self._explicit_api_key = None
        self._explicit_base_url = None
        self.reasoning_config = {"effort": "medium"}
        self.agent = _Agent()
        self.conversation_history = []
        self._pending_model_switch_note = None
        self._pending_one_turn_model_restore = None
        self._session_db = _SessionDB()
        self.session_id = "session-1"

    def _stage_and_swap_model(self, result, old_model):
        from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin

        return CLIModelSwitchMixin._stage_and_swap_model(self, result, old_model)

    def _confirm_expensive_model_switch(self, result):
        return result.success


def _virtual_result():
    return ModelSwitchResult(
        success=True,
        new_model="TRP_AUTO",
        target_provider="trp-auto",
        provider_changed=True,
        provider_label="TRP_AUTO",
    )


def _physical_runtime():
    return {
        "provider": "openai-codex",
        "requested_provider": "trp-auto",
        "api_key": "validated-subscription-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        "capabilities": {"reasoning": True},
        "source": "device_code",
        "trp_auto": {
            "virtual_provider": "trp-auto",
            "virtual_model": "TRP_AUTO",
            "bootstrap_provider": "openai-codex",
            "bootstrap_model": "gpt-6-sol",
        },
    }


@pytest.mark.parametrize("surface", ["typed", "picker"])
def test_trp_auto_apply_materializes_physical_runtime_but_persists_virtual_selection(
    monkeypatch, surface
):
    import cli as cli_mod
    import hermes_cli.cli_model_switch_mixin as mixin

    cli = _CLI()
    persisted = []
    monkeypatch.setattr(mixin, "_resolve_cli_reasoning", lambda _cli: None)
    monkeypatch.setattr(mixin, "_merge_preflight_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mixin, "_print_switch_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.trp_auto_bridge.resolve_bootstrap_runtime",
        lambda target: _physical_runtime(),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.persist_model_selection", persisted.append
    )

    result = _virtual_result()
    if surface == "typed":
        mixin.CLIModelSwitchMixin._confirm_and_apply_cli_model_switch(
            cli, result, True, False
        )
    else:
        mixin.CLIModelSwitchMixin._apply_model_switch_result(cli, result, True)

    assert (cli.provider, cli.model, cli.requested_provider) == (
        "openai-codex",
        "gpt-6-sol",
        "trp-auto",
    )
    assert cli.agent.calls == [
        {
            "new_model": "gpt-6-sol",
            "new_provider": "openai-codex",
            "api_key": "validated-subscription-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "capabilities": {"reasoning": True},
        }
    ]
    assert cli.agent.requested_provider == "trp-auto"
    assert cli.agent._primary_runtime == {
        "model": "gpt-6-sol",
        "provider": "openai-codex",
        "requested_provider": "trp-auto",
    }
    assert persisted == [result]
    assert cli._session_db.model_updates == [("session-1", "TRP_AUTO")]
    route = cli._session_db.config_updates[0][1]
    assert route["provider"] == "trp-auto"
    assert route["gateway_runtime"]["provider"] == "trp-auto"


def test_trp_auto_bootstrap_failure_leaves_cli_and_agent_unchanged(monkeypatch):
    import cli as cli_mod
    import hermes_cli.cli_model_switch_mixin as mixin

    cli = _CLI()
    before_cli = (
        cli.provider,
        cli.model,
        cli.requested_provider,
        cli.api_key,
        cli.base_url,
        cli.api_mode,
    )
    before_agent = dict(cli.agent._primary_runtime)
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text, *_args, **_kwargs: printed.append(text))
    monkeypatch.setattr(
        "hermes_cli.trp_auto_bridge.resolve_bootstrap_runtime",
        lambda _target: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )

    applied = mixin.CLIModelSwitchMixin._stage_and_swap_model(
        cli, _virtual_result(), cli.model
    )

    assert applied is False
    assert (
        cli.provider,
        cli.model,
        cli.requested_provider,
        cli.api_key,
        cli.base_url,
        cli.api_mode,
    ) == before_cli
    assert cli.agent.calls == []
    assert cli.agent._primary_runtime == before_agent
    assert any("staying on" in line for line in printed)


def test_trp_auto_apply_uses_runtime_provider_disabled_guard(monkeypatch):
    import cli as cli_mod
    import hermes_cli.cli_model_switch_mixin as mixin

    cli = _CLI()
    printed = []
    bridge_calls = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text, *_args, **_kwargs: printed.append(text))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider disabled")),
    )
    monkeypatch.setattr(
        "hermes_cli.trp_auto_bridge.resolve_bootstrap_runtime",
        lambda _target: bridge_calls.append(_target) or _physical_runtime(),
    )

    applied = mixin.CLIModelSwitchMixin._stage_and_swap_model(
        cli, _virtual_result(), cli.model
    )

    assert applied is False
    assert bridge_calls == []
    assert cli.agent.calls == []
    assert any("staying on" in line for line in printed)
