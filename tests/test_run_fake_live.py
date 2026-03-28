from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from config_utils import load_config
from exchanges.fake import FakeCCXTClient
from tools.run_fake_live import (
    _async_main,
    _apply_assertions,
    _extract_hsl_trace,
    _install_fake_user_override,
    _install_runtime_overrides,
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
        self._equity_hard_stop_halted = False
        self._equity_hard_stop_no_restart_latched = False
        self._equity_hard_stop_halted_until_ms = None
        self._equity_hard_stop_pending_red_since_ms = None
        self._equity_hard_stop_red_flat_confirmations = 0
        self._equity_hard_stop_cooldown_intervention_active = False
        self._equity_hard_stop_cooldown_repanic_reset_pending = False
        self._equity_hard_stop_last_metrics = None
        self._equity_hard_stop_last_stop_event = None

    async def update_pos_oos_pnls_ohlcvs(self):
        return True

    def _equity_hard_stop_enabled(self):
        return False

    def _equity_hard_stop_runtime_red_latched(self):
        return False

    def _equity_hard_stop_runtime_tier(self):
        return "green"

    async def execute_to_exchange(self):
        self.loop_calls += 1
        return {"cycle": self.loop_calls}


@pytest.mark.asyncio
async def test_run_fake_bot_advances_until_timeline_end():
    bot = _StubBot()
    client = FakeCCXTClient(_scenario(), quote="USDT")
    summaries = await _run_fake_bot(bot, client, max_steps=None)
    assert bot.loop_calls == 3
    assert [row["step_index"] for row in summaries] == [0, 1, 2]


def test_apply_assertions_validates_positions_and_hsl_state():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = _StubBot()
    scenario = {
        "assertions": {
            "fill_count": 0,
            "final_balance": {"approx": 1000.0, "tolerance": 1e-9},
            "last_prices": {"BTC/USDT:USDT": 100.0},
            "final_positions": {"BTC/USDT:USDT|long": 0.0},
            "hsl_paths": {"halted": False, "runtime_tier": "green"},
        }
    }
    _apply_assertions(bot, client, scenario, step_summaries=[], log_text="")


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
            "hsl_paths": {"halted": False, "runtime_tier": "green"},
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


def test_extract_hsl_trace_returns_serializable_state():
    bot = _StubBot()
    trace = _extract_hsl_trace(bot)
    assert trace["halted"] is False
    assert trace["runtime_tier"] == "green"


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


def test_prime_fake_candles_seeds_candlestick_manager_cache():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    cm = type("CM", (), {"_cache": {}, "_ema_cache": {}, "_current_close_cache": {}, "_tf_range_cache": {}})()
    bot = type("Bot", (), {"cm": cm})()
    _prime_fake_candles(bot, client)
    arr = bot.cm._cache["BTC/USDT:USDT"]
    assert len(arr) == 1
    assert int(arr[0]["ts"]) == 1_767_225_600_000
    assert float(arr[0]["c"]) == pytest.approx(100.0)


def test_bot_params_to_rust_dict_includes_core_orchestrator_fields():
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
            return self.values.get(key, 0.0)

        def bp(self, _pside, key, _symbol=None):
            return self.values.get(key, 0.0)

    out = Passivbot._bot_params_to_rust_dict(_Stub(), "long", None)
    assert out["n_positions"] == 1
    assert out["wallet_exposure_limit"] == pytest.approx(5.0)
    assert out["ema_span_0"] == pytest.approx(2.0)
    assert out["ema_span_1"] == pytest.approx(4.0)


def test_install_runtime_overrides_sets_exchange_time_override():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = type("Bot", (), {"cca": client})()
    _install_runtime_overrides(bot, {})
    assert bot.get_exchange_time() == client.now_ms


def test_refresh_halted_runtime_forced_modes_uses_halted_mode_for_all_symbols():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.positions = {"BTC/USDT:USDT": {"long": {"size": 5.0}, "short": {"size": 0.0}}}
            self.open_orders = {}
            self.active_symbols = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
            self._runtime_forced_modes = {"long": {}, "short": {}}
            self._equity_hard_stop_halted = True

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
    assert stub._runtime_forced_modes["long"]["ETH/USDT:USDT"] == "graceful_stop"


@pytest.mark.asyncio
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
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_panic.hjson",
            "fake_hsl_manual_panic_test",
            5,
            "cooldown violation repanic flattened; cooldown reset",
            2.0,
            "panic",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_normal.hjson",
            "fake_hsl_manual_normal_test",
            5,
            "operator override during RED cooldown: resumed normal operation and reset drawdown tracker",
            2.0,
            "normal",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_graceful_stop.hjson",
            "fake_hsl_manual_graceful_stop_test",
            5,
            "detected non-flat position during RED cooldown | policy=graceful_stop",
            2.0,
            "graceful_stop",
        ),
        (
            "scenarios/fake_live/hsl_long_cooldown_manual_entry_manual.hjson",
            "fake_hsl_manual_manual_test",
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
        cfg["bot"]["common"]["equity_hard_stop_loss"]["cooldown_minutes_after_red"] = float(
            cooldown_override
        )
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
