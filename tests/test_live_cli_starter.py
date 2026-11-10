"""Exercise live CLI selection with local fixtures before any exchange action."""

import json
import sys

import pytest

from config.access import require_live_value
from config.schema import CONFIG_SCHEMA_VERSION, STARTER_LIVE_USER, get_starter_config


class ReachedBotSetup(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("file_user", "cli_args", "expected_user", "starter"),
    [
        (None, [], STARTER_LIVE_USER, True),
        (None, ["--live.account.user", STARTER_LIVE_USER], STARTER_LIVE_USER, True),
        (None, ["-u", "bybit_01"], "bybit_01", False),
        (STARTER_LIVE_USER, [], STARTER_LIVE_USER, False),
        ("file_account", ["-u", "other_account"], "other_account", False),
    ],
)
async def test_live_selects_starter_only_for_its_opt_in_account(
    monkeypatch, tmp_path, file_user, cli_args, expected_user, starter
):
    import passivbot

    config_args = []
    if file_user is not None:
        path = tmp_path / "sparse.json"
        path.write_text(
            json.dumps({
                "config_version": CONFIG_SCHEMA_VERSION,
                "live": {"account": {"user": file_user}},
            })
        )
        config_args.append(str(path))
    monkeypatch.setattr(sys, "argv", ["passivbot live", *config_args, *cli_args])
    monkeypatch.setattr(passivbot, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(passivbot, "configure_custom_endpoint_loader", lambda *a, **kw: None)

    def fake_user_info(user):
        assert user == expected_user
        return {"exchange": "fake"}

    async def fake_markets(exchange, **kwargs):
        assert exchange == "fake"

    def reached_setup(config):
        assert require_live_value(config, "user") == expected_user
        assert config["bot"]["long"]["enabled"] is starter
        assert config["bot"]["short"]["enabled"] is False
        approved = get_starter_config()["bot"]["global"]["universe"]["approved_coins"]
        assert config["bot"]["long"]["universe"]["approved_coins"] == (approved if starter else [])
        raise ReachedBotSetup

    monkeypatch.setattr(passivbot, "load_user_info", fake_user_info)
    monkeypatch.setattr(passivbot, "load_markets", fake_markets)
    monkeypatch.setattr(passivbot, "setup_bot", reached_setup)
    with pytest.raises(ReachedBotSetup):
        await passivbot._run_live({})


@pytest.mark.asyncio
async def test_missing_starter_alias_reports_setup_before_market_loading(
    monkeypatch, tmp_path, caplog
):
    import passivbot

    monkeypatch.chdir(tmp_path)
    (tmp_path / "api-keys.json").write_text(json.dumps({"bybit_01": {"exchange": "fake"}}))
    monkeypatch.setattr(sys, "argv", ["passivbot live"])
    monkeypatch.setattr(passivbot, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(passivbot, "configure_custom_endpoint_loader", lambda *a, **kw: None)

    async def forbidden(*args, **kwargs):
        pytest.fail("missing starter account reached market loading")

    monkeypatch.setattr(passivbot, "load_markets", forbidden)
    with pytest.raises(SystemExit) as error:
        await passivbot.main()
    assert error.value.code == 1
    assert f"Configure account {STARTER_LIVE_USER} in local api-keys.json" in caplog.text
