"""Startup endpoint selection stops before any account or exchange setup."""

from copy import deepcopy

import pytest

import passivbot


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,argv,selected,autodiscover",
    [
        (None, [], None, True),
        ("config.json", [], "config.json", False),
        ("none", [], None, False),
        ("off", [], None, False),
        ("disable", [], None, False),
        ("", [], None, False),
        ("config.json", ["--custom-endpoints", "cli.json"], "cli.json", False),
        ("config.json", ["--custom-endpoints", "none"], None, False),
        ("none", ["--custom-endpoints", "cli.json"], "cli.json", False),
        (
            None,
            ["--live.api.custom_endpoints_path", "canonical.json"],
            "canonical.json",
            False,
        ),
    ],
)
async def test_grouped_custom_endpoint_selection_preserves_cli_precedence(
    monkeypatch, configured, argv, selected, autodiscover
):
    config = passivbot.get_template_config()
    config["live"]["account"]["user"] = "offline_test"
    config["live"]["api"]["custom_endpoints_path"] = configured
    monkeypatch.setattr(passivbot.sys, "argv", ["passivbot", *argv])
    monkeypatch.setattr(passivbot, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        passivbot, "load_input_config", lambda *args: (config, "", deepcopy(config))
    )
    monkeypatch.setattr(
        passivbot,
        "resolve_live_log_file_settings",
        lambda *args, **kwargs: {"log_file": None},
    )
    loaded = []
    preloaded = object()

    def load(path):
        loaded.append(path)
        return preloaded

    class EndpointBoundary(Exception):
        pass

    def configure(path, **kwargs):
        assert path == selected
        assert kwargs == {
            "autodiscover": autodiscover,
            "preloaded": preloaded if selected is not None else None,
        }
        raise EndpointBoundary()

    def forbidden(*args, **kwargs):
        pytest.fail("endpoint selection reached account or exchange setup")

    monkeypatch.setattr(passivbot, "load_custom_endpoint_config", load)
    monkeypatch.setattr(passivbot, "configure_custom_endpoint_loader", configure)
    for name in ("load_user_info", "load_markets", "setup_bot"):
        monkeypatch.setattr(passivbot, name, forbidden)
    context = {}
    with pytest.raises(EndpointBoundary):
        await passivbot._run_live(context)
    assert context["stage"] == "custom_endpoints"
    assert loaded == ([] if selected is None else [selected])
