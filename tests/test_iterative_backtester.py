import asyncio
from pathlib import Path

import pytest

from tools import iterative_backtester as ib


def test_parse_override_value_supports_common_scalar_types():
    assert ib.parse_override_value("true") is True
    assert ib.parse_override_value("false") is False
    assert ib.parse_override_value("null") is None
    assert ib.parse_override_value("7") == 7
    assert ib.parse_override_value("1.25") == pytest.approx(1.25)
    assert ib.parse_override_value('{"x": 1}') == {"x": 1}
    assert ib.parse_override_value("[1, 2]") == [1, 2]
    assert ib.parse_override_value("now") == "now"


def test_apply_cli_overrides_updates_nested_config_paths():
    config = {
        "backtest": {"start_date": "2023-01-01", "end_date": "2024-01-01"},
        "bot": {
            "long": {"risk": {"n_positions": 1}, "strategy": {"ema_anchor": {"offset": 0.003}}},
            "short": {"risk": {"n_positions": 1}, "strategy": {"ema_anchor": {"offset": 0.003}}},
        },
    }

    overridden = ib.apply_cli_overrides(
        config,
        [
            "backtest.start_date=2022-01-01",
            "backtest.end_date=now",
            "bot.long.strategy.ema_anchor.offset=0.01",
            "bot.short.risk.n_positions=0",
        ],
    )

    assert overridden["backtest"]["start_date"] == "2022-01-01"
    assert overridden["backtest"]["end_date"] == "now"
    assert overridden["bot"]["long"]["strategy"]["ema_anchor"]["offset"] == pytest.approx(0.01)
    assert overridden["bot"]["short"]["risk"]["n_positions"] == 0
    assert config["backtest"]["start_date"] == "2023-01-01"


def test_parse_cli_override_rejects_missing_equals():
    with pytest.raises(ValueError, match="expected dotted.path=value"):
        ib.parse_cli_override("backtest.start_date")


def test_async_main_rejects_quit_after_run_without_auto_run(monkeypatch):
    monkeypatch.setattr(
        ib.sys,
        "argv",
        ["iterative_backtester.py", "configs/examples/ema_anchor.json", "--quit-after-run"],
    )

    with pytest.raises(SystemExit):
        asyncio.run(ib.async_main())


def test_async_main_passes_cli_overrides_to_session(monkeypatch):
    captured = {}

    class FakeSession:
        def __init__(self, config_path, log_level, auto_run, *, cli_overrides, quit_after_run):
            captured["config_path"] = config_path
            captured["log_level"] = log_level
            captured["auto_run"] = auto_run
            captured["cli_overrides"] = cli_overrides
            captured["quit_after_run"] = quit_after_run

        async def initialize(self):
            captured["initialized"] = True

        async def interactive_loop(self):
            captured["looped"] = True

    monkeypatch.setattr(ib, "IterativeBacktestSession", FakeSession)
    monkeypatch.setattr(
        ib.sys,
        "argv",
        [
            "iterative_backtester.py",
            "configs/examples/ema_anchor.json",
            "--auto-run",
            "--quit-after-run",
            "--override",
            "backtest.start_date=2022-01-01",
            "--override",
            "backtest.end_date=now",
        ],
    )

    asyncio.run(ib.async_main())

    assert captured["config_path"] == Path("configs/examples/ema_anchor.json")
    assert captured["auto_run"] is True
    assert captured["quit_after_run"] is True
    assert captured["cli_overrides"] == [
        "backtest.start_date=2022-01-01",
        "backtest.end_date=now",
    ]
    assert captured["initialized"] is True
    assert captured["looped"] is True


def test_dataset_signature_changes_for_dataset_affecting_config():
    base = {
        "backtest": {"start_date": "2023-01-01", "coins": {"combined": ["BTC"]}, "cache_dir": {}},
        "bot": {
            "long": {
                "strategy": {
                    "ema_anchor": {
                        "ema_span_0": 48,
                        "ema_span_1": 60,
                        "entry_volatility_ema_span_hours": 12,
                        "offset": 0.003,
                    }
                }
            },
            "short": {
                "strategy": {
                    "ema_anchor": {
                        "ema_span_0": 48,
                        "ema_span_1": 60,
                        "entry_volatility_ema_span_hours": 12,
                        "offset": 0.003,
                    }
                }
            },
        },
        "live": {
            "approved_coins": {"long": ["BTC"], "short": []},
            "strategy_kind": "ema_anchor",
            "warmup_ratio": 1.0,
            "max_warmup_minutes": 0,
            "minimum_coin_age_days": 365,
        },
        "optimize": {"scoring": [{"metric": "adg_strategy_pnl_rebased", "goal": "max"}]},
    }

    same = {
        **base,
        "backtest": {"start_date": "2023-01-01", "coins": {"combined": ["ETH"]}, "cache_dir": {"combined": "x"}},
    }
    warmup_changed = {
        **base,
        "live": {**base["live"], "max_warmup_minutes": 720},
    }
    bot_quote_changed = {
        **base,
        "bot": {
            "long": {
                "strategy": {
                    "ema_anchor": {
                        "ema_span_0": 48,
                        "ema_span_1": 60,
                        "entry_volatility_ema_span_hours": 12,
                        "offset": 0.01,
                    }
                }
            },
            "short": {
                "strategy": {
                    "ema_anchor": {
                        "ema_span_0": 48,
                        "ema_span_1": 60,
                        "entry_volatility_ema_span_hours": 12,
                        "offset": 0.003,
                    }
                }
            },
        },
    }
    live_changed = {
        **base,
        "live": {**base["live"], "approved_coins": {"long": ["BTC", "SOL"], "short": []}},
    }
    optimize_changed = {
        **base,
        "optimize": {"scoring": [{"metric": "sharpe_ratio_usd", "goal": "max"}]},
    }

    assert ib.make_dataset_signature(base) == ib.make_dataset_signature(same)
    assert ib.make_dataset_signature(base) != ib.make_dataset_signature(warmup_changed)
    assert ib.make_dataset_signature(base) == ib.make_dataset_signature(bot_quote_changed)
    assert ib.make_dataset_signature(base) != ib.make_dataset_signature(live_changed)
    assert ib.make_dataset_signature(base) == ib.make_dataset_signature(optimize_changed)


def test_run_signature_changes_for_optimize_changes():
    base = {
        "backtest": {"start_date": "2023-01-01", "coins": {"combined": ["BTC"]}, "cache_dir": {}},
        "bot": {"long": {"strategy": {"ema_anchor": {"ema_span_0": 48}}}},
        "live": {
            "approved_coins": {"long": ["BTC"], "short": []},
            "strategy_kind": "ema_anchor",
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0,
            "minimum_coin_age_days": 365,
        },
        "optimize": {"scoring": [{"metric": "adg_strategy_pnl_rebased", "goal": "max"}]},
    }
    optimize_changed = {
        **base,
        "optimize": {"scoring": [{"metric": "sharpe_ratio_usd", "goal": "max"}]},
    }

    assert ib.make_run_signature(base) != ib.make_run_signature(optimize_changed)


def test_session_initialize_infers_combined_mode_from_exchange_count(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text("{}", encoding="utf-8")

    async def fake_load_config(self):
        return {"backtest": {"exchanges": ["binance", "bybit"], "base_dir": "backtests"}}

    async def fake_prepare_datasets(self, config):
        assert config["backtest"]["exchanges"] == ["binance", "bybit"]
        return {}

    monkeypatch.setattr(ib.IterativeBacktestSession, "_load_config", fake_load_config)
    monkeypatch.setattr(ib.IterativeBacktestSession, "_prepare_datasets", fake_prepare_datasets)
    monkeypatch.setattr(ib, "make_backtest_signature", lambda config: "sig")
    monkeypatch.setattr(ib, "make_get_filepath", lambda path: path)
    monkeypatch.setattr(ib.time, "strftime", lambda fmt: "iterative_20260404_000000")

    session = ib.IterativeBacktestSession(config_path, None, False)
    asyncio.run(session.initialize())

    assert session.backtest_exchanges == ["binance", "bybit"]
    assert session.combine_ohlcvs is True
