from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

import tools.run_fake_live as run_fake_live_module
from fill_events_manager import FillEvent, FillEventCache
from config_utils import load_config
from exchanges.fake import FakeCCXTClient
from passivbot import setup_bot
from tools.run_fake_live import (
    _async_main,
    _apply_assertions,
    _compare_run_artifacts,
    _extract_hsl_trace,
    _install_fake_user_override,
    _install_runtime_overrides,
    _load_run_artifacts,
    _prime_fake_candles,
    _prime_fake_fill_cache,
    _run_fake_bot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scenario() -> dict:
    return {
        "name": "runner",
        "start_time": "2026-01-01T00:00:00Z",
        "tick_interval_seconds": 60,
        "boot_index": 0,
        "account": {"balance": 1000.0},
        "symbols": {
            "BTC/USDT:USDT": {
                "qty_step": 0.001,
                "price_step": 0.1,
                "min_qty": 0.001,
                "min_cost": 5.0,
            }
        },
        "timeline": [
            {"t": 0, "prices": {"BTC/USDT:USDT": 100.0}},
            {"t": 1, "prices": {"BTC/USDT:USDT": 101.0}},
            {"t": 2, "prices": {"BTC/USDT:USDT": 102.0}},
        ],
    }


class _StubBot:
    def __init__(self) -> None:
        self.loop_calls = 0
        self._equity_hard_stop = {
            "long": {"halted": False, "no_restart_latched": False, "last_metrics": None},
            "short": {"halted": True, "no_restart_latched": False, "last_metrics": None},
        }

    async def update_pos_oos_pnls_ohlcvs(self):
        return True

    def _equity_hard_stop_enabled(self, pside: str | None = None):
        return False if pside is not None else False

    def _equity_hard_stop_runtime_red_latched(self, pside: str):
        return False

    def _hsl_psides(self):
        return ("long", "short")

    async def execute_to_exchange(self):
        self.loop_calls += 1
        return {"cycle": self.loop_calls}

    def _hsl_state(self, pside: str):
        return self._equity_hard_stop[pside]


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_run_fake_bot_advances_until_timeline_end():
    bot = _StubBot()
    client = FakeCCXTClient(_scenario(), quote="USDT")
    summaries = await _run_fake_bot(bot, client, max_steps=None)
    assert bot.loop_calls == 3
    assert [row["step_index"] for row in summaries] == [0, 1, 2]


@pytest.mark.fake_live
def test_apply_assertions_validates_positions_and_halted_psides():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = _StubBot()
    scenario = {
        "assertions": {
            "fill_count": 0,
            "final_balance": {"approx": 1000.0, "tolerance": 1e-9},
            "last_prices": {"BTC/USDT:USDT": 100.0},
            "final_positions": {"BTC/USDT:USDT|long": 0.0},
            "halted_psides": {"long": False, "short": True},
        }
    }
    _apply_assertions(bot, client, scenario, step_summaries=[], log_text="")


@pytest.mark.fake_live
def test_apply_assertions_supports_path_assertions_and_logs():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = _StubBot()
    step_summaries = [{"step_index": 0, "fills": 0, "positions": []}]
    scenario = {
        "assertions": {
            "state_paths": {
                "current_index": 0,
                "prices.BTC/USDT:USDT": {"approx": 100.0, "tolerance": 1e-9},
            },
            "hsl_paths": {"short.halted": True},
            "summary_paths": {"step_count": 1, "last.step_index": 0},
            "log_contains": ["READY", "fake"],
        }
    }
    _apply_assertions(
        bot,
        client,
        scenario,
        step_summaries=step_summaries,
        log_text="READY fake harness\n",
    )


@pytest.mark.fake_live
def test_install_fake_user_override_restores_original_loader():
    import passivbot as passivbot_mod

    original = passivbot_mod.load_user_info
    config = {"live": {"user": "demo"}}
    fake_user, restore = _install_fake_user_override(config, "scenario.hjson", None)
    try:
        payload = passivbot_mod.load_user_info(fake_user)
        assert payload["exchange"] == "fake"
        assert payload["fake_scenario_path"] == "scenario.hjson"
    finally:
        restore()
    assert passivbot_mod.load_user_info is original


@pytest.mark.fake_live
def test_extract_hsl_trace_returns_serializable_state():
    bot = _StubBot()
    trace = _extract_hsl_trace(bot)
    assert trace["long"]["halted"] is False
    assert trace["short"]["halted"] is True


@pytest.mark.fake_live
def test_prime_fake_fill_cache_writes_fake_fill_events(tmp_path):
    scenario = _scenario()
    scenario["account"]["fills"] = [
        {
            "id": "1",
            "order": "1",
            "timestamp": "2026-01-01T00:00:00Z",
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "side": "buy",
            "amount": 1.0,
            "price": 100.0,
            "clientOrderId": "boot_entry",
        }
    ]
    client = FakeCCXTClient(scenario, quote="USDT")
    bot = type("Bot", (), {"exchange": "fake", "user": "runner_test"})()
    cache_path = _prime_fake_fill_cache(bot, client, cache_root=tmp_path)
    payload = next(cache_path.glob("*.json")).read_text(encoding="utf-8")
    assert "boot_entry" in payload


@pytest.mark.fake_live
def test_prime_fake_candles_seeds_candlestick_manager_cache():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    cm = type("CM", (), {"_cache": {}, "_ema_cache": {}, "_current_close_cache": {}, "_tf_range_cache": {}})()
    bot = type("Bot", (), {"cm": cm})()
    _prime_fake_candles(bot, client)
    arr = bot.cm._cache["BTC/USDT:USDT"]
    assert len(arr) == 1
    assert int(arr[0]["ts"]) == 1_767_225_600_000
    assert float(arr[0]["c"]) == pytest.approx(100.0)


@pytest.mark.fake_live
def test_resume_normal_cooldown_does_not_preauthorize_flat_halted_side():
    scenario_path = (
        REPO_ROOT
        / "scenarios"
        / "fake_live"
        / "hsl_long_cooldown_resume_normal_bot_self_entry_bug.hjson"
    )
    cfg = load_config(str(REPO_ROOT / "configs" / "fake_live_hsl_btc.hjson"), verbose=False)
    cfg["bot"]["long"]["hsl_cooldown_minutes_after_red"] = 2.0
    cfg["live"]["hsl_position_during_cooldown_policy"] = "normal"
    cfg["live"]["hsl_signal_mode"] = "unified"
    cfg["bot"]["short"] = json.loads(json.dumps(cfg["bot"]["long"]))
    cfg["bot"]["short"]["hsl_enabled"] = False
    cfg["live"]["approved_coins"]["long"] = ["XMR"]
    cfg["live"]["approved_coins"]["short"] = ["XMR"]
    cfg["live"]["fake_scenario_path"] = str(scenario_path)

    _, restore_user_override = _install_fake_user_override(
        cfg,
        str(scenario_path),
        "fake_hsl_resume_normal_self_entry_bug",
    )
    try:
        bot = setup_bot(cfg)
        state = bot._hsl_state("long")
        state["halted"] = True
        state["cooldown_until_ms"] = 1

        symbol = "XMR/USDT:USDT"
        assert bot._equity_hard_stop_halted_mode("long", symbol) == "graceful_stop"
        assert bot._orchestrator_mode_override("long", symbol) == "graceful_stop"
    finally:
        restore_user_override()


def test_bot_params_to_rust_dict_includes_hsl_fields():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.values = {
                "close_grid_markup_end": 0.01,
                "close_grid_markup_start": 0.005,
                "close_grid_qty_pct": 1.0,
                "close_trailing_retracement_pct": 0.001,
                "close_trailing_grid_ratio": 0.0,
                "close_trailing_qty_pct": 0.0,
                "close_trailing_threshold_pct": 0.001,
                "entry_grid_double_down_factor": 1.0,
                "entry_grid_inflation_enabled": True,
                "entry_grid_spacing_volatility_weight": 0.0,
                "entry_grid_spacing_we_weight": 0.0,
                "entry_grid_spacing_pct": 0.01,
                "entry_volatility_ema_span_hours": 0.0,
                "entry_initial_ema_dist": -0.001,
                "entry_initial_qty_pct": 0.1,
                "entry_trailing_double_down_factor": 1.0,
                "entry_trailing_retracement_pct": 0.001,
                "entry_trailing_retracement_we_weight": 0.0,
                "entry_trailing_retracement_volatility_weight": 0.0,
                "entry_trailing_grid_ratio": 0.0,
                "entry_trailing_threshold_pct": 0.001,
                "entry_trailing_threshold_we_weight": 0.0,
                "entry_trailing_threshold_volatility_weight": 0.0,
                "filter_volatility_ema_span": 0.0,
                "filter_volume_ema_span": 1.0,
                "forager_volume_drop_pct": 0.0,
                "forager_score_weights": {
                    "volume": 1.0,
                    "ema_readiness": 0.0,
                    "volatility": 0.0,
                },
                "ema_span_0": 2.0,
                "ema_span_1": 4.0,
                "hsl_enabled": True,
                "hsl_red_threshold": 0.05,
                "hsl_ema_span_minutes": 1.0,
                "hsl_cooldown_minutes_after_red": 1.0,
                "hsl_no_restart_drawdown_threshold": 0.9,
                "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
                "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
                "hsl_panic_close_order_type": "market",
                "n_positions": 1.0,
                "total_wallet_exposure_limit": 5.0,
                "wallet_exposure_limit": 5.0,
                "risk_wel_enforcer_threshold": 1.0,
                "risk_twel_enforcer_threshold": 1.0,
                "risk_we_excess_allowance_pct": 0.0,
                "unstuck_close_pct": 0.01,
                "unstuck_ema_dist": 0.0,
                "unstuck_loss_allowance_pct": 0.1,
                "unstuck_threshold": 1.0,
            }

        def bot_value(self, _pside, key):
            if key == "hsl_tier_ratios.yellow":
                return self.values["hsl_tier_ratios"]["yellow"]
            if key == "hsl_tier_ratios.orange":
                return self.values["hsl_tier_ratios"]["orange"]
            return self.values.get(key, 0.0)

        def bp(self, _pside, key, _symbol=None):
            return self.values.get(key, 0.0)

    out = Passivbot._bot_params_to_rust_dict(_Stub(), "long", None)
    assert out["hsl_enabled"] is True
    assert out["hsl_red_threshold"] == pytest.approx(0.05)
    assert out["hsl_tier_ratio_yellow"] == pytest.approx(0.5)
    assert out["hsl_tier_ratio_orange"] == pytest.approx(0.75)
    assert out["hsl_orange_tier_mode"] == "tp_only_with_active_entry_cancellation"
    assert out["hsl_panic_close_order_type"] == "market"
    assert out["entry_grid_inflation_enabled"] is True
    assert out["forager_score_weights"] == {
        "volume": pytest.approx(1.0),
        "ema_readiness": pytest.approx(0.0),
        "volatility": pytest.approx(0.0),
    }


def test_bot_params_to_rust_dict_respects_coin_override_entry_grid_inflation_flag():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.config = {
                "bot": {
                    "long": {
                        "close_grid_markup_end": 0.01,
                        "close_grid_markup_start": 0.005,
                        "close_grid_qty_pct": 1.0,
                        "close_trailing_retracement_pct": 0.001,
                        "close_trailing_grid_ratio": 0.0,
                        "close_trailing_qty_pct": 0.0,
                        "close_trailing_threshold_pct": 0.001,
                        "entry_grid_double_down_factor": 1.0,
                        "entry_grid_inflation_enabled": True,
                        "entry_grid_spacing_volatility_weight": 0.0,
                        "entry_grid_spacing_we_weight": 0.0,
                        "entry_grid_spacing_pct": 0.01,
                        "entry_volatility_ema_span_hours": 0.0,
                        "entry_initial_ema_dist": -0.001,
                        "entry_initial_qty_pct": 0.1,
                        "entry_trailing_double_down_factor": 1.0,
                        "entry_trailing_retracement_pct": 0.001,
                        "entry_trailing_retracement_we_weight": 0.0,
                        "entry_trailing_retracement_volatility_weight": 0.0,
                        "entry_trailing_grid_ratio": 0.0,
                        "entry_trailing_threshold_pct": 0.001,
                        "entry_trailing_threshold_we_weight": 0.0,
                        "entry_trailing_threshold_volatility_weight": 0.0,
                        "forager_volatility_ema_span": 0.0,
                        "forager_volume_ema_span": 1.0,
                        "forager_volume_drop_pct": 0.0,
                        "forager_score_weights": {
                            "volume": 1.0,
                            "ema_readiness": 0.0,
                            "volatility": 0.0,
                        },
                        "ema_span_0": 2.0,
                        "ema_span_1": 4.0,
                        "hsl_enabled": True,
                        "hsl_red_threshold": 0.05,
                        "hsl_ema_span_minutes": 1.0,
                        "hsl_cooldown_minutes_after_red": 1.0,
                        "hsl_no_restart_drawdown_threshold": 0.9,
                        "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
                        "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
                        "hsl_panic_close_order_type": "market",
                        "n_positions": 1.0,
                        "total_wallet_exposure_limit": 5.0,
                        "wallet_exposure_limit": 5.0,
                        "risk_wel_enforcer_threshold": 1.0,
                        "risk_twel_enforcer_threshold": 1.0,
                        "risk_we_excess_allowance_pct": 0.0,
                        "unstuck_close_pct": 0.01,
                        "unstuck_ema_dist": 0.0,
                        "unstuck_loss_allowance_pct": 0.1,
                        "unstuck_threshold": 1.0,
                    },
                    "short": {},
                }
            }
            self.coin_overrides = {
                "BTC/USDT:USDT": {"bot": {"long": {"entry_grid_inflation_enabled": False}}}
            }

        def bot_value(self, _pside, key):
            if key == "hsl_tier_ratios.yellow":
                return self.config["bot"]["long"]["hsl_tier_ratios"]["yellow"]
            if key == "hsl_tier_ratios.orange":
                return self.config["bot"]["long"]["hsl_tier_ratios"]["orange"]
            return self.config["bot"]["long"][key]

        def bp(self, pside, key, symbol=None):
            if symbol in self.coin_overrides:
                override = (
                    self.coin_overrides[symbol].get("bot", {}).get(pside, {}).get(key, None)
                )
                if override is not None:
                    return override
            return self.config["bot"][pside][key]

    out = Passivbot._bot_params_to_rust_dict(_Stub(), "long", "BTC/USDT:USDT")

    assert out["entry_grid_inflation_enabled"] is False


def test_install_runtime_overrides_sets_exchange_time_override():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = type("Bot", (), {"cca": client})()
    _install_runtime_overrides(bot, {})
    assert bot.get_exchange_time() == client.now_ms


def test_main_parses_authoritative_refresh_mode(monkeypatch):
    seen = {}

    async def fake_async_main(args):
        seen["mode"] = args.authoritative_refresh_mode
        return 0

    monkeypatch.setattr(run_fake_live_module, "_async_main", fake_async_main)
    monkeypatch.setattr(
        run_fake_live_module.sys,
        "argv",
        [
            "run_fake_live.py",
            "config.json",
            "scenario.hjson",
            "--authoritative-refresh-mode",
            "staged",
        ],
    )

    assert run_fake_live_module.main() == 0
    assert seen["mode"] == "staged"


def test_compare_run_artifacts_reports_no_diff_for_matching_payloads():
    payload = {
        "step_summaries": [{"step_index": 0, "fills": 0}],
        "fake_exchange_state": {"balance_total": 1000.0},
        "fills": [],
        "positions": [],
        "hsl_trace": {"long": {"halted": False}},
        "run_metadata": {"authoritative_refresh_mode": "legacy"},
    }
    report = _compare_run_artifacts(payload, {**payload, "run_metadata": {"authoritative_refresh_mode": "staged"}})
    assert report["match"] is True
    assert report["diff_count"] == 0


def test_compare_run_artifacts_ignores_nondeterministic_fields():
    legacy = {
        "step_summaries": [{"step_index": 0, "fills": 1}],
        "fake_exchange_state": {
            "balance_total": 1000.0,
            "fills": [
                {
                    "id": "1",
                    "order": "1",
                    "clientOrderId": "legacy-oid",
                    "symbol": "BTC/USDT:USDT",
                    "position_side": "long",
                    "side": "buy",
                    "price": 100.0,
                    "amount": 0.01,
                    "timestamp": 1,
                    "pnl": 0.0,
                    "reduceOnly": False,
                    "info": {"clientOrderId": "legacy-oid", "positionSide": "LONG"},
                }
            ],
        },
        "fills": [
            {
                "id": "1",
                "order": "1",
                "clientOrderId": "legacy-oid",
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "side": "buy",
                "price": 100.0,
                "amount": 0.01,
                "timestamp": 1,
                "pnl": 0.0,
                "reduceOnly": False,
                "info": {"clientOrderId": "legacy-oid", "positionSide": "LONG"},
            }
        ],
        "positions": [],
        "hsl_trace": {
            "long": {
                "halted": False,
                "last_stop_event": {"triggered_at": "a", "user": "legacy_user", "tier": "red"},
            }
        },
        "run_metadata": {"authoritative_refresh_mode": "legacy"},
    }
    staged = {
        "step_summaries": [{"step_index": 0, "fills": 1}],
        "fake_exchange_state": {
            "balance_total": 1000.0,
            "fills": [
                {
                    "id": "99",
                    "order": "99",
                    "clientOrderId": "staged-oid",
                    "symbol": "BTC/USDT:USDT",
                    "position_side": "long",
                    "side": "buy",
                    "price": 100.0,
                    "amount": 0.01,
                    "timestamp": 1,
                    "pnl": 0.0,
                    "reduceOnly": False,
                    "info": {"clientOrderId": "staged-oid", "positionSide": "LONG"},
                }
            ],
        },
        "fills": [
            {
                "id": "99",
                "order": "99",
                "clientOrderId": "staged-oid",
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "side": "buy",
                "price": 100.0,
                "amount": 0.01,
                "timestamp": 1,
                "pnl": 0.0,
                "reduceOnly": False,
                "info": {"clientOrderId": "staged-oid", "positionSide": "LONG"},
            }
        ],
        "positions": [],
        "hsl_trace": {
            "long": {
                "halted": False,
                "last_stop_event": {"triggered_at": "b", "user": "staged_user", "tier": "red"},
            }
        },
        "run_metadata": {"authoritative_refresh_mode": "staged"},
    }

    report = _compare_run_artifacts(legacy, staged)

    assert report["match"] is True
    assert report["diff_count"] == 0


def test_load_run_artifacts_reads_expected_files(tmp_path):
    (tmp_path / "step_summaries.json").write_text(json.dumps([{"step_index": 0}]), encoding="utf-8")
    (tmp_path / "fake_exchange_state.json").write_text(json.dumps({"balance_total": 1}), encoding="utf-8")
    (tmp_path / "fills.json").write_text("[]", encoding="utf-8")
    (tmp_path / "positions.json").write_text("[]", encoding="utf-8")
    (tmp_path / "hsl_trace.json").write_text(json.dumps({"long": {"halted": False}}), encoding="utf-8")
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"authoritative_refresh_mode": "legacy"}), encoding="utf-8"
    )
    (tmp_path / "fake_live.log").write_text("hello\n", encoding="utf-8")

    loaded = _load_run_artifacts(tmp_path)

    assert loaded["step_summaries"][0]["step_index"] == 0
    assert loaded["fake_exchange_state"]["balance_total"] == 1
    assert loaded["log_text"] == "hello\n"


def test_main_parses_compare_authoritative_refresh_modes(monkeypatch):
    seen = {}

    async def fake_async_main(args):
        seen["compare"] = args.compare_authoritative_refresh_modes
        return 0

    monkeypatch.setattr(run_fake_live_module, "_async_main", fake_async_main)
    monkeypatch.setattr(
        run_fake_live_module.sys,
        "argv",
        [
            "run_fake_live.py",
            "config.json",
            "scenario.hjson",
            "--compare-authoritative-refresh-modes",
        ],
    )

    assert run_fake_live_module.main() == 0
    assert seen["compare"] is True


def test_refresh_halted_runtime_forced_modes_keeps_active_red_pside_in_panic():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.positions = {"BTC/USDT:USDT": {"long": {"size": 5.0}, "short": {"size": 0.0}}}
            self.open_orders = {}
            self.active_symbols = {"BTC/USDT:USDT"}
            self._runtime_forced_modes = {"long": {}, "short": {}}
            self._states = {
                "long": {"halted": False},
                "short": {"halted": False},
            }

        def _hsl_psides(self):
            return ("long", "short")

        def _equity_hard_stop_enabled(self, pside=None):
            return pside == "long"

        def _hsl_state(self, pside):
            return self._states[pside]

        def _equity_hard_stop_runtime_red_latched(self, pside):
            return pside == "long"

        def _equity_hard_stop_set_red_runtime_forced_modes(self, pside):
            self._runtime_forced_modes[pside] = {"BTC/USDT:USDT": "panic"}

        def _equity_hard_stop_halted_mode(self, pside, symbol):
            del pside
            return "panic" if symbol == "BTC/USDT:USDT" else "graceful_stop"

        def _equity_hard_stop_clear_runtime_forced_modes(self, pside=None):
            if pside is None:
                self._runtime_forced_modes = {"long": {}, "short": {}}
            else:
                self._runtime_forced_modes[pside] = {}

    stub = _Stub()
    Passivbot._equity_hard_stop_refresh_halted_runtime_forced_modes(stub)
    assert stub._runtime_forced_modes["long"]["BTC/USDT:USDT"] == "panic"


def test_refresh_halted_runtime_forced_modes_keeps_halted_pside_in_panic_or_graceful_stop():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.positions = {"BTC/USDT:USDT": {"long": {"size": 5.0}, "short": {"size": 0.0}}}
            self.open_orders = {}
            self.active_symbols = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
            self._runtime_forced_modes = {"long": {}, "short": {}}
            self._states = {
                "long": {"halted": True},
                "short": {"halted": False},
            }

        def _hsl_psides(self):
            return ("long", "short")

        def _equity_hard_stop_enabled(self, pside=None):
            return pside == "long"

        def _hsl_state(self, pside):
            return self._states[pside]

        def _equity_hard_stop_runtime_red_latched(self, pside):
            return pside == "long"

        def _equity_hard_stop_clear_runtime_forced_modes(self, pside=None):
            if pside is None:
                self._runtime_forced_modes = {"long": {}, "short": {}}
            else:
                self._runtime_forced_modes[pside] = {}

        def _equity_hard_stop_set_red_runtime_forced_modes(self, pside):
            raise AssertionError("halted pside should use halted-mode mapping, not active RED panic map")

        def _equity_hard_stop_halted_mode(self, pside, symbol):
            del pside
            return "panic" if symbol == "BTC/USDT:USDT" else "graceful_stop"

    stub = _Stub()
    Passivbot._equity_hard_stop_refresh_halted_runtime_forced_modes(stub)
    assert stub._runtime_forced_modes["long"]["BTC/USDT:USDT"] == "panic"
    assert stub._runtime_forced_modes["long"]["ETH/USDT:USDT"] == "graceful_stop"


@pytest.mark.asyncio
@pytest.mark.fake_live
@pytest.mark.parametrize(
    (
        "scenario_rel",
        "user",
        "expected_steps",
        "expected_log_fragment",
        "cooldown_override",
        "policy_override",
    ),
    [
        (
            "scenarios/fake_live/hsl_long_red_restart.hjson",
            "fake_hsl_restart_test",
            4,
            "RED cooldown elapsed; trading resumed",
            None,
            None,
        ),
        (
            "scenarios/fake_live/hsl_long_terminal_no_restart.hjson",
            "fake_hsl_terminal_test",
            3,
            "RED stop finalized (terminal)",
            None,
            None,
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_repanic_reset.hjson",
            "fake_hsl_manual_cooldown_test",
            5,
            "cooldown violation repanic flattened; cooldown reset",
            2.0,
            "panic",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_resume_normal.hjson",
            "fake_hsl_manual_resume_normal_test",
            5,
            "operator override during RED cooldown: resumed normal operation and reset drawdown tracker",
            2.0,
            "normal",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_manual_quarantine.hjson",
            "fake_hsl_manual_quarantine_test",
            5,
            "detected non-flat position during RED cooldown | policy=manual",
            2.0,
            "manual",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_tp_only.hjson",
            "fake_hsl_manual_tp_only_test",
            5,
            "detected non-flat position during RED cooldown | policy=tp_only",
            2.0,
            "tp_only",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_graceful_stop.hjson",
            "fake_hsl_manual_graceful_stop_test",
            5,
            "detected non-flat position during RED cooldown | policy=graceful_stop",
            2.0,
            "graceful_stop",
        ),
    ],
)
async def test_hsl_replay_scenarios_run_end_to_end(
    tmp_path,
    scenario_rel,
    user,
    expected_steps,
    expected_log_fragment,
    cooldown_override,
    policy_override,
):
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    config_path = REPO_ROOT / "configs" / "fake_live_hsl_btc.hjson"
    if cooldown_override is not None:
        cfg = load_config(str(config_path), verbose=False)
        cfg["bot"]["long"]["hsl_cooldown_minutes_after_red"] = float(cooldown_override)
        if policy_override is not None:
            cfg["live"]["hsl_position_during_cooldown_policy"] = str(policy_override)
        config_path = tmp_path / "fake_live_hsl_btc_override.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")

    args = argparse.Namespace(
        config=str(config_path),
        scenario=str(REPO_ROOT / scenario_rel),
        user=user,
        max_steps=None,
        output_dir=str(tmp_path),
        log_level=1,
        snapshot_each_step=False,
    )
    assert await _async_main(args) == 0

    output_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(output_dirs) == 1
    run_dir = output_dirs[0]

    step_summaries = json.loads((run_dir / "step_summaries.json").read_text(encoding="utf-8"))
    assert len(step_summaries) == expected_steps
    assert (run_dir / "hsl_trace.json").exists()
    assert expected_log_fragment in (run_dir / "fake_live.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_live_all_lookback_backfills_narrow_fill_cache_once(tmp_path, monkeypatch):
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    user = "fake_hsl_pnls_lookback_all_test"
    scenario_path = REPO_ROOT / "scenarios" / "fake_live" / "hsl_long_red_restart.hjson"
    cache_dir = REPO_ROOT / "caches" / "fill_events" / "fake" / user
    shutil.rmtree(cache_dir, ignore_errors=True)

    cfg = load_config(str(REPO_ROOT / "configs" / "fake_live_hsl_btc.hjson"), verbose=False)
    cfg["live"]["pnls_max_lookback_days"] = "all"
    config_path = tmp_path / "fake_live_hsl_btc_all_lookback.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    def _prime_narrow_window_cache(bot, fake_client, cache_root=None):
        root = Path(cache_root) if cache_root is not None else Path("caches") / "fill_events"
        cache_path = root / str(bot.exchange) / str(bot.user)
        shutil.rmtree(cache_path, ignore_errors=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        all_events = [FillEvent.from_dict(event) for event in fake_client.get_fill_events(None, None)]
        narrow_events = [event for event in all_events if str(event.id) != "10"]
        cache = FillEventCache(cache_path)
        cache.save(narrow_events)
        cache.update_metadata_from_events(narrow_events)
        cache.set_history_scope("window")
        return cache_path

    monkeypatch.setattr(run_fake_live_module, "_prime_fake_fill_cache", _prime_narrow_window_cache)

    args = argparse.Namespace(
        config=str(config_path),
        scenario=str(scenario_path),
        user=user,
        max_steps=None,
        output_dir=str(tmp_path),
        log_level=1,
        snapshot_each_step=False,
    )

    try:
        assert await _async_main(args) == 0
        output_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
        assert len(output_dirs) == 1
        run_dir = output_dirs[0]

        log_text = (run_dir / "fake_live.log").read_text(encoding="utf-8")
        assert "[fills] refresh: events=3 (+1)" in log_text
        assert "initial_entry_boot" in log_text
        assert log_text.count("id=10") == 1

        cache = FillEventCache(cache_dir)
        cached_ids = [str(event.id) for event in cache.load()]
        assert cache.get_history_scope() == "all"
        assert "10" in cached_ids
        assert {"10", "11", "12", "13"}.issubset(set(cached_ids))
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_live_min_effective_cost_blocks_zero_min_qty_integer_step_symbol(tmp_path):
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    scenario = {
        "name": "min_effective_cost_zero_min_qty_guard",
        "start_time": "2026-03-31T15:18:00Z",
        "tick_interval_seconds": 60,
        "boot_index": 4,
        "account": {"balance": 363.52606},
        "symbols": {
            "SOL/USDT:USDT": {
                "qty_step": 1.0,
                "price_step": 0.01,
                "min_qty": 0.0,
                "min_cost": 0.1,
                "contractSize": 1.0,
                "maker_fee": 0.0002,
                "taker_fee": 0.00055,
            }
        },
        "timeline": [
            {"t": 0, "prices": {"SOL/USDT:USDT": 88.165}},
            {"t": 1, "prices": {"SOL/USDT:USDT": 88.165}},
            {"t": 2, "prices": {"SOL/USDT:USDT": 88.165}},
            {"t": 3, "prices": {"SOL/USDT:USDT": 88.165}},
            {"t": 4, "prices": {"SOL/USDT:USDT": 88.165}},
        ],
    }
    scenario_path = tmp_path / "fake_min_effective_cost.hjson"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    cfg = load_config(str(REPO_ROOT / "configs" / "fake_live_hsl_btc.hjson"), verbose=False)
    cfg["bot"]["long"]["hsl_enabled"] = False
    cfg["bot"]["short"]["hsl_enabled"] = False
    cfg["bot"]["long"]["entry_initial_qty_pct"] = 0.0276
    cfg["bot"]["long"]["n_positions"] = 5.0
    cfg["bot"]["long"]["total_wallet_exposure_limit"] = 1.8
    cfg["bot"]["long"]["risk_we_excess_allowance_pct"] = 0.37
    cfg["bot"]["short"]["n_positions"] = 0.0
    cfg["bot"]["short"]["total_wallet_exposure_limit"] = 0.0
    cfg["live"]["approved_coins"]["long"] = ["SOL"]
    cfg["live"]["approved_coins"]["short"] = []
    cfg["live"]["ignored_coins"]["long"] = []
    cfg["live"]["ignored_coins"]["short"] = []
    cfg["live"]["filter_by_min_effective_cost"] = True
    cfg["live"]["market_orders_allowed"] = False
    cfg["live"]["fake_scenario_path"] = str(scenario_path)
    config_path = tmp_path / "fake_live_min_effective_cost.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    args = argparse.Namespace(
        config=str(config_path),
        scenario=str(scenario_path),
        user="fake_min_effective_cost_guard",
        max_steps=1,
        output_dir=str(tmp_path),
        log_level=1,
        snapshot_each_step=False,
    )
    assert await _async_main(args) == 0

    output_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(output_dirs) == 1
    run_dir = output_dirs[0]

    step_summaries = json.loads((run_dir / "step_summaries.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "fake_exchange_state.json").read_text(encoding="utf-8"))
    positions = json.loads((run_dir / "positions.json").read_text(encoding="utf-8"))
    fills = json.loads((run_dir / "fills.json").read_text(encoding="utf-8"))
    log_text = (run_dir / "fake_live.log").read_text(encoding="utf-8")

    assert len(step_summaries) == 1
    assert step_summaries[0]["open_orders"] == 0
    assert step_summaries[0]["fills"] == 0
    assert step_summaries[0]["positions"] == []
    assert state["open_orders"] == []
    assert positions == []
    assert fills == []
    assert "[order]   post SOL" not in log_text
