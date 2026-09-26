from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeAgent:
    def __init__(self, *, requested_provider: str = "trp-auto"):
        self.requested_provider = requested_provider
        self.provider = "openai-codex"
        self.model = "bootstrap-codex"
        self.base_url = "https://codex.example"
        self.api_key = "bootstrap-token"
        self.api_mode = "codex_responses"
        self.runtime_capabilities = {"seat": "bootstrap"}
        self._primary_runtime = {
            "provider": self.provider,
            "model": self.model,
            "requested_provider": requested_provider,
        }
        self.switch_calls: list[dict] = []
        self.warnings: list[str] = []

    def switch_model(self, **kwargs):
        self.switch_calls.append(kwargs)
        self.provider = kwargs["new_provider"]
        self.model = kwargs["new_model"]
        self.base_url = kwargs["base_url"]
        self.api_key = kwargs["api_key"]
        self.api_mode = kwargs["api_mode"]
        self.runtime_capabilities = kwargs["capabilities"]
        self.requested_provider = kwargs["new_provider"]
        self._primary_runtime = {
            "provider": self.provider,
            "model": self.model,
            "requested_provider": self.requested_provider,
        }

    def _emit_warning(self, message: str):
        self.warnings.append(message)


def _install_seat_plugin(home: Path) -> None:
    plugin = home / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "from types import SimpleNamespace\n"
        "def resolve_seat(user_text, *, requested_provider, current_provider, current_model):\n"
        "    if 'policy' in user_text:\n"
        "        provider, model = 'anthropic', 'policy-claude'\n"
        "    elif 'research' in user_text:\n"
        "        provider, model = 'xai', 'research-grok'\n"
        "    else:\n"
        "        provider, model = 'openai-codex', 'coding-codex'\n"
        "    return SimpleNamespace(provider=provider, model=model, reason='classified', "
        "switch=(provider, model) != (current_provider, current_model), "
        "frontier=provider == 'xai', control_digest='digest')\n",
        encoding="utf-8",
    )


def _runtime(provider: str, model: str) -> dict:
    physical_provider, source = {
        "openai-codex": ("openai-codex", "device_code"),
        "anthropic": ("anthropic", "claude_code"),
        "xai-oauth": ("xai-oauth", "device_code"),
    }[provider]
    return {
        "provider": physical_provider,
        "model": model,
        "base_url": f"https://{physical_provider}.example",
        "api_key": f"{physical_provider}-subscription",
        "api_mode": "anthropic_messages" if provider == "anthropic" else "codex_responses",
        "capabilities": {"seat": model},
        "source": source,
    }


def test_route_turn_switches_coding_to_codex_with_virtual_provenance(tmp_path, monkeypatch):
    _install_seat_plugin(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", _runtime)
    agent = _FakeAgent()

    assert bridge.route_turn(agent, "implement the parser") is True
    assert agent.switch_calls == [{
        "new_model": "coding-codex",
        "new_provider": "openai-codex",
        "api_key": "openai-codex-subscription",
        "base_url": "https://openai-codex.example",
        "api_mode": "codex_responses",
        "capabilities": {"seat": "coding-codex"},
    }]
    assert (agent.provider, agent.model, agent.requested_provider) == (
        "openai-codex",
        "coding-codex",
        "trp-auto",
    )
    assert agent._primary_runtime == {
        "provider": "openai-codex",
        "model": "coding-codex",
        "requested_provider": "trp-auto",
    }


@pytest.mark.parametrize(
    ("prompt", "provider", "model"),
    [
        ("review this policy", "anthropic", "policy-claude"),
        ("research the latest incident", "xai", "research-grok"),
    ],
)
def test_route_turn_selects_policy_and_research_seats(
    tmp_path, monkeypatch, prompt, provider, model
):
    _install_seat_plugin(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", _runtime)
    agent = _FakeAgent()

    assert bridge.route_turn(agent, prompt) is True
    assert (agent.provider, agent.model, agent.requested_provider) == (
        "xai-oauth" if provider == "xai" else provider,
        model,
        "trp-auto",
    )


@pytest.mark.parametrize("requested_provider", ["anthropic", "moa", "custom"])
def test_direct_physical_and_other_virtual_selections_are_byte_compatible_noops(
    tmp_path, monkeypatch, requested_provider
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.trp_auto_bridge import route_turn

    agent = _FakeAgent(requested_provider=requested_provider)
    before = dict(vars(agent))

    assert route_turn(agent, "research the policy") is False
    assert vars(agent) == before


def test_repeated_route_for_the_current_seat_does_not_switch_again(tmp_path, monkeypatch):
    _install_seat_plugin(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", _runtime)
    agent = _FakeAgent()

    assert bridge.route_turn(agent, "implement the parser") is True
    assert bridge.route_turn(agent, "implement the parser") is False
    assert len(agent.switch_calls) == 1


def test_repeated_grok_route_compares_platform_and_physical_provider_names(tmp_path, monkeypatch):
    _install_seat_plugin(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    monkeypatch.setattr(bridge, "_resolve_physical_runtime", _runtime)
    agent = _FakeAgent()

    assert bridge.route_turn(agent, "research the latest incident") is True
    assert bridge.route_turn(agent, "research the latest incident") is False
    assert len(agent.switch_calls) == 1


def test_plugin_error_preserves_current_seat_and_emits_non_secret_warning(tmp_path, monkeypatch):
    plugin = tmp_path / "plugins" / "trp-auto"
    plugin.mkdir(parents=True)
    (plugin / "seat.py").write_text(
        "def resolve_seat(*args, **kwargs):\n"
        "    raise RuntimeError('secret-token-value')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.trp_auto_bridge import route_turn

    agent = _FakeAgent()
    physical_before = (agent.provider, agent.model, agent.base_url, agent.api_key)

    assert route_turn(agent, "research current pricing") is False
    assert (agent.provider, agent.model, agent.base_url, agent.api_key) == physical_before
    assert len(agent.warnings) == 1
    assert "secret-token-value" not in agent.warnings[0]
    assert "kept the current subscription seat" in agent.warnings[0]


@pytest.mark.parametrize(
    "seat_source",
    [
        None,
        "value = 'no resolve_seat here'\n",
        (
            "from types import SimpleNamespace\n"
            "def resolve_seat(*args, **kwargs):\n"
            "    return SimpleNamespace(provider='openrouter', model='paid/model', "
            "reason='bad', switch=True, frontier=False, control_digest='digest')\n"
        ),
    ],
)
def test_missing_or_malformed_plugin_never_falls_to_openrouter(
    tmp_path, monkeypatch, seat_source
):
    if seat_source is not None:
        plugin = tmp_path / "plugins" / "trp-auto"
        plugin.mkdir(parents=True)
        (plugin / "seat.py").write_text(seat_source, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.trp_auto_bridge as bridge

    physical_resolutions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        bridge,
        "_resolve_physical_runtime",
        lambda provider, model: physical_resolutions.append((provider, model)),
    )
    agent = _FakeAgent()
    before = (agent.provider, agent.model, agent.base_url, agent.api_key)

    assert bridge.route_turn(agent, "research current pricing") is False
    assert (agent.provider, agent.model, agent.base_url, agent.api_key) == before
    assert physical_resolutions == []
    assert len(agent.warnings) == 1


def test_turn_boundary_routes_once_after_restore_before_runtime_publication(monkeypatch):
    import agent.turn_context as turn_context
    import hermes_cli.trp_auto_bridge as bridge

    events: list[tuple] = []
    agent = SimpleNamespace(
        session_id="session",
        _memory_write_origin="assistant_tool",
        _review_attended=False,
        provider="fallback",
        model="fallback-model",
    )

    def restore():
        events.append(("restore",))
        agent.provider = "openai-codex"
        agent.model = "bootstrap-codex"

    def route(routed_agent, user_text):
        events.append(("route", routed_agent.provider, routed_agent.model, user_text))
        routed_agent.provider = "anthropic"
        routed_agent.model = "policy-claude"
        return True

    def publish(published_agent):
        events.append(("publish", published_agent.provider, published_agent.model))

    class _BoundaryReached(Exception):
        pass

    agent._restore_primary_runtime = restore
    monkeypatch.setattr(bridge, "route_turn", route)
    monkeypatch.setattr(turn_context, "_publish_runtime_main", publish)
    monkeypatch.setattr(
        turn_context,
        "_refresh_mcp_tools_between_turns",
        lambda _agent: (_ for _ in ()).throw(_BoundaryReached()),
    )

    with pytest.raises(_BoundaryReached):
        turn_context.build_turn_context(
            agent=agent,
            user_message="review this policy",
            system_message=None,
            conversation_history=None,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda *_a, **_k: events.append(("prompt",)),
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda text: text,
            summarize_user_message_for_log=lambda text: text,
            set_session_context=lambda _session_id: None,
            set_current_write_origin=lambda _origin: None,
            ra=lambda: SimpleNamespace(_set_interrupt=lambda *_a, **_k: None),
        )

    assert events == [
        ("restore",),
        ("route", "openai-codex", "bootstrap-codex", "review this policy"),
        ("publish", "anthropic", "policy-claude"),
    ]
