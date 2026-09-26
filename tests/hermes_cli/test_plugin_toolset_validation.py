"""Regression coverage for dynamic plugin toolset validation."""

from cli import HermesCLI


def _cli_with_messages():
    cli = HermesCLI.__new__(HermesCLI)
    messages = []
    cli._console_print = messages.append
    return cli, messages


def test_plugin_toolset_is_not_reported_as_unknown(monkeypatch):
    import cli as cli_module
    import hermes_cli.tools_config as tools_config

    cli, messages = _cli_with_messages()
    monkeypatch.setitem(cli_module.CLI_CONFIG, "agent", {})
    monkeypatch.setitem(cli_module.CLI_CONFIG, "mcp_servers", {})
    monkeypatch.setattr(cli_module, "validate_toolset", lambda _name: False)
    monkeypatch.setattr(
        tools_config, "_get_plugin_toolset_keys", lambda: {"trp_auto"}
    )

    cli._init_toolsets(["trp_auto"])

    assert messages == []
    assert cli.enabled_toolsets == ["trp_auto"]


def test_truly_unknown_toolset_still_warns(monkeypatch):
    import cli as cli_module
    import hermes_cli.tools_config as tools_config

    cli, messages = _cli_with_messages()
    monkeypatch.setitem(cli_module.CLI_CONFIG, "agent", {})
    monkeypatch.setitem(cli_module.CLI_CONFIG, "mcp_servers", {})
    monkeypatch.setattr(cli_module, "validate_toolset", lambda _name: False)
    monkeypatch.setattr(tools_config, "_get_plugin_toolset_keys", set)

    cli._init_toolsets(["not-a-toolset"])

    assert len(messages) == 1
    assert "Unknown toolsets: not-a-toolset" in messages[0]
