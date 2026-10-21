"""Authored millisecond names preserve existing runtime values and event keys."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from config.load import prepare_config
from config.schema import get_template_config


def test_explicit_millisecond_config_reaches_existing_runtime_interfaces(monkeypatch):
    from fill_events_manager import fee_policy_kwargs_from_config
    from passivbot import Passivbot

    source = get_template_config()
    source["live"]["api"]["recv_window_milliseconds"] = 12345
    source["live"]["fills"]["fee_conversion_max_age_milliseconds"] = 987654
    source["live"]["startup"]["startup_phase_budgets"] = {
        "account": {"elapsed_milliseconds": 3456, "since_previous_milliseconds": 789},
    }
    original = deepcopy(source)
    prepared = prepare_config(source, verbose=False)
    bot = SimpleNamespace(config=prepared, _startup_timing_marks={},
                          _startup_timing_started_ms=100, _startup_timing_last_mark_ms=200)
    assert Passivbot._build_ccxt_options(bot)["recvWindow"] == 12345
    assert fee_policy_kwargs_from_config(prepared)["fee_conversion_max_age_ms"] == 987654
    emitted = []
    monkeypatch.setattr("passivbot.utc_ms", lambda: 1000)
    monkeypatch.setattr(Passivbot, "_live_event_console_available", lambda self: True)
    monkeypatch.setattr(Passivbot, "_emit_startup_timing_event", lambda self, **event: emitted.append(event))
    Passivbot._startup_timing_mark(bot, "account")
    assert emitted == [{"phase": "account", "elapsed_ms": 900, "since_previous_ms": 800,
                        "details": "", "elapsed_budget_ms": 3456,
                        "since_previous_budget_ms": 789, "budget_source": "config"}]
    assert source == original


@pytest.mark.parametrize("path,old", [
    (("live", "api"), "recv_window_ms"),
    (("live", "fills"), "fee_conversion_max_age_ms"),
    (("monitor",), "price_tick_min_interval_ms"),
    (("live", "startup", "startup_phase_budgets", "account"), "elapsed_ms"),
    (("live", "startup", "startup_phase_budgets", "account"), "since_previous_ms"),
])
def test_unreleased_abbreviated_millisecond_names_are_rejected(path, old):
    from tools.migrate_config_v9 import migrate

    source = get_template_config()
    node = source
    for key in path:
        node = node.setdefault(key, {})
    node[old] = 123
    with pytest.raises((KeyError, TypeError, ValueError), match="unknown|Unknown|retired|unexpected"):
        prepare_config(source, verbose=False)
    with pytest.raises((KeyError, TypeError, ValueError), match="unknown|Unknown|retired|unexpected"):
        migrate(source)


def test_monitor_millisecond_config_preserves_native_throttling_value(tmp_path):
    from monitor_publisher import MonitorPublisher

    source = get_template_config()
    source["monitor"].update(root_dir=str(tmp_path), price_tick_min_interval_milliseconds=789)
    prepared = prepare_config(source, verbose=False)
    publisher = MonitorPublisher.from_config(exchange="fake", user="synthetic", config=prepared["monitor"])
    assert publisher.price_tick_min_interval_ms == 789
    assert prepared["monitor"]["price_tick_min_interval_milliseconds"] == 789


def test_millisecond_cli_uses_explicit_units_without_retired_dotted_aliases():
    import argparse
    from config_utils import add_config_arguments, project_template_config_for_cli, update_config_with_args

    source = get_template_config()
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, project_template_config_for_cli(source, "live"),
                                command="live", help_all=True)
    args = parser.parse_args(["--fee-conversion-max-age-milliseconds", "987654",
                              "--live.api.recv_window_milliseconds", "12345"])
    update_config_with_args(source, args, allowed_keys=keys)
    assert source["live"]["api"]["recv_window_milliseconds"] == 12345
    assert source["live"]["fills"]["fee_conversion_max_age_milliseconds"] == 987654
    for flag in ("--live.api.recv_window_ms", "--live.fills.fee_conversion_max_age_ms",
                 "--fee-conversion-max-age-ms"):
        with pytest.raises(SystemExit):
            parser.parse_args([flag, "123"])
