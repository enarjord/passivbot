"""Startup activation reporting uses permission flags independently of capacity."""

import logging
from types import SimpleNamespace

import pytest

from passivbot import Passivbot


@pytest.mark.parametrize(
    "long_enabled,short_enabled,twel,mode",
    [
        (True, False, 1.0, "LONG only"),
        (False, True, 1.0, "SHORT only"),
        (True, True, 1.0, "LONG + SHORT"),
        (False, False, 1.0, "DISABLED"),
        (True, False, 0.0, "LONG only"),
    ],
)
def test_startup_banner_reports_explicit_activation(
    caplog, long_enabled, short_enabled, twel, mode
):
    bot = SimpleNamespace(
        user="offline_fixture",
        exchange="fake",
        config={"bot": {"long": {"enabled": long_enabled},
                        "short": {"enabled": short_enabled}}},
        bot_value=lambda side, key: {
            "total_wallet_exposure_limit": twel, "n_positions": 2,
        }[key],
        _live_market_orders_allowed=lambda: False,
        runtime_identity=SimpleNamespace(
            run_id="r" * 32, passivbot_version="9.0.0", python_git_dirty=False,
            python_git_commit="a" * 40, config_sha256="b" * 64,
            rust_source_sha256="c" * 64, rust_artifact_sha256="d" * 64,
        ),
    )
    with caplog.at_level(logging.INFO):
        Passivbot._log_startup_banner(bot)

    assert f"Mode: {mode} " in caplog.text
    assert "Positions: 2L/2S" in caplog.text
    assert f"TWEL: L:{twel:.0%} S:{twel:.0%}" in caplog.text
    assert (
        f"Long: {'enabled' if long_enabled else 'disabled'}; "
        f"Short: {'enabled' if short_enabled else 'disabled'}"
    ) in caplog.text
    assert ("held-position management follows configured modes" in caplog.text) is (
        not long_enabled and not short_enabled
    )
