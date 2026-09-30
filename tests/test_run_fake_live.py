from __future__ import annotations

import argparse
import hashlib
import hjson
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_fake_live as run_fake_live_module
from fill_events_manager import FillEvent, FillEventCache
from config_utils import load_config
from exchanges.fake import FakeCCXTClient
from live.event_bus import EventTypes, LiveEvent
from passivbot import setup_bot
from tools.run_fake_live import (
    _async_main,
    _apply_assertions,
    _compare_run_artifacts,
    _extract_hsl_trace,
    _install_candle_remote_fetch_trace,
    _install_fake_user_override,
    _install_runtime_overrides,
    _load_run_artifacts,
    _prime_fake_candles,
    _prime_fake_fill_cache,
    _run_fake_bot,
    _summarize_remote_calls,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_candle_remote_fetch_trace_sanitizes_hostile_payload():
    forwarded = []
    bot = SimpleNamespace(cm=SimpleNamespace(_remote_fetch_callback=forwarded.append))
    events, restore = _install_candle_remote_fetch_trace(bot)
    url = "https://api.example.invalid/ohlcv?apiKey=SECRET&signature=abc"

    try:
        bot.cm._remote_fetch_callback(
            {
                "kind": "ccxt_fetch_ohlcv",
                "stage": "error",
                "url": url,
                "params": {"until": 123, "apiKey": "SECRET"},
                "error_type": "AuthError",
                "error": f"GET {url}",
                "error_repr": f"AuthError({url!r})",
            }
        )
    finally:
        restore()

    assert len(events) == 1
    assert events[0]["event_index"] == 0
    assert events[0]["param_keys"] == ["apiKey", "until"]
    assert len(events[0]["url_hash"]) == 64
    assert "url" not in events[0]
    assert "params" not in events[0]
    assert "error" not in events[0]
    assert "error_repr" not in events[0]
    assert "SECRET" not in str(events[0])
    assert forwarded == [
        {key: value for key, value in events[0].items() if key != "event_index"}
    ]


def _cleanup_fake_user_state(user: str) -> None:
    shutil.rmtree(
        REPO_ROOT / "caches" / "fill_events" / "fake" / user, ignore_errors=True
    )


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
        self.config = {"live": {"hsl_signal_mode": "coin"}}
        self._hsl_live = SimpleNamespace(cycle=self.cycle)

    async def cycle(self):
        self.loop_calls += 1
        return {"cycle": self.loop_calls, "updated": True, "ordinary_executed": True}


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_run_fake_bot_advances_until_timeline_end():
    bot = _StubBot()
    client = FakeCCXTClient(_scenario(), quote="USDT")
    summaries = await _run_fake_bot(bot, client, max_steps=None)
    assert bot.loop_calls == 3
    assert [row["step_index"] for row in summaries] == [0, 1, 2]


@pytest.mark.fake_live
def test_live_event_capture_uses_isolated_pipeline_without_default_pipeline():
    class _NoDefaultPipelineBot:
        exchange = "fake"
        user = "fake_capture"
        bot_id = "fake_capture_bot"
        _live_event_pipeline = None

    bot = _NoDefaultPipelineBot()
    sink, restore = run_fake_live_module._install_live_event_capture(bot)
    pipeline = bot._live_event_pipeline
    try:
        assert pipeline.console_sink is None
        assert pipeline.monitor_sinks == ()
        assert pipeline.structured_sinks == (sink,)
        assert pipeline.emit(LiveEvent(EventTypes.BOT_STARTED)) is not None
        assert pipeline.flush(timeout=2.0) is True
        assert [event.event_type for event in sink.events] == [EventTypes.BOT_STARTED]
    finally:
        restore()
    assert bot._live_event_pipeline is None


@pytest.mark.fake_live
def test_live_event_capture_serialization_reports_truncation():
    sink = run_fake_live_module._BoundedLiveEventSink(max_retained=2)
    for event_type in (
        EventTypes.BOT_STARTED,
        EventTypes.BOT_READY,
        EventTypes.CYCLE_COMPLETED,
    ):
        sink.write(LiveEvent(event_type, data={"token": "secret-token"}))

    events, metadata = run_fake_live_module._serialize_live_event_capture(sink)

    assert [event["event_type"] for event in events] == [
        EventTypes.BOT_READY,
        EventTypes.CYCLE_COMPLETED,
    ]
    assert all(event["data"]["token"] == "[redacted]" for event in events)
    assert metadata == {"total": 3, "retained": 2, "truncated": 1, "max_retained": 2}


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_live_persists_events_when_console_pipeline_is_disabled(
    tmp_path, monkeypatch
):
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    monkeypatch.setenv("PASSIVBOT_LIVE_EVENT_CONSOLE", "0")
    user = f"fake_capture_only_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    scenario = hjson.loads(
        (
            REPO_ROOT / "scenarios" / "fake_live" / "hsl_long_red_restart.hjson"
        ).read_text(encoding="utf-8")
    )
    scenario.pop("assertions", None)
    scenario["run_initial_cycle"] = False
    scenario_path = tmp_path / "capture_only.hjson"
    scenario_path.write_text(hjson.dumps(scenario), encoding="utf-8")

    try:
        args = argparse.Namespace(
            config=str(REPO_ROOT / "configs" / "examples" / "fake_live_hsl.json"),
            scenario=str(scenario_path),
            user=user,
            max_steps=0,
            output_dir=str(tmp_path),
            log_level=1,
            snapshot_each_step=False,
        )
        assert await _async_main(args) == 0
        run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
        artifacts = _load_run_artifacts(run_dir)
        assert any(
            event.get("event_type") == EventTypes.BOT_STARTED
            for event in artifacts["live_events"]
        )
        assert any(
            event.get("event_type") == EventTypes.BOT_STOPPING
            for event in artifacts["live_events"]
        )
        assert any(
            event.get("event_type") == EventTypes.BOT_STOPPED
            for event in artifacts["live_events"]
        )
        assert "[monitor] failed building monitor snapshot" not in artifacts["log_text"]
        capture = artifacts["run_metadata"]["live_event_capture"]
        assert capture["total"] == capture["retained"] == len(artifacts["live_events"])
        assert capture["truncated"] == 0
        assert capture["max_retained"] == run_fake_live_module.MAX_CAPTURED_LIVE_EVENTS
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.fake_live
def test_apply_assertions_validates_positions():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = _StubBot()
    scenario = {
        "assertions": {
            "fill_count": 0,
            "final_balance": {"approx": 1000.0, "tolerance": 1e-9},
            "last_prices": {"BTC/USDT:USDT": 100.0},
            "final_positions": {"BTC/USDT:USDT|long": 0.0},
        }
    }
    _apply_assertions(bot, client, scenario, step_summaries=[], log_text="")


@pytest.mark.fake_live
def test_apply_assertions_supports_path_assertions_and_logs():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = _StubBot()
    bot.config = {"live": {"hsl_signal_mode": "coin"}}
    bot.get_exchange_time = lambda: 0
    step_summaries = [{"step_index": 0, "fills": 0, "positions": []}]
    scenario = {
        "assertions": {
            "state_paths": {
                "current_index": 0,
                "prices.BTC/USDT:USDT": {"approx": 100.0, "tolerance": 1e-9},
            },
            "hsl_paths": {
                "hsl.engine": "hsl",
                "hsl.observation_status": "not_evaluated",
                "hsl.scope_count": 0,
            },
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
def test_apply_assertions_supports_remote_call_paths():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    bot = _StubBot()
    remote_calls = [
        {"method": "fetch_balance", "step_index": 0},
        {"method": "fetch_positions", "step_index": 0, "rows": 1},
        {
            "method": "fetch_ohlcv",
            "step_index": 1,
            "symbol": "BTC/USDT:USDT",
            "rows": 2,
        },
    ]
    remote_summary = _summarize_remote_calls(remote_calls)
    scenario = {
        "assertions": {
            "remote_call_paths": {
                "summary.total_calls": 3,
                "summary.by_category.account_state": 2,
                "summary.by_category.market_data": 1,
                "summary.max_per_step_by_method.fetch_balance": 1,
                "calls.2.symbol": "BTC/USDT:USDT",
            }
        }
    }
    _apply_assertions(
        bot,
        client,
        scenario,
        step_summaries=[],
        log_text="",
        remote_calls=remote_calls,
        remote_call_summary=remote_summary,
    )


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_client_request_log_counts_order_writes(monkeypatch):
    from live.hsl_live import Owner

    monkeypatch.setattr(Owner, "admit", lambda self, order: True)
    from exchanges.ccxt_bot import CCXTBot

    client = FakeCCXTClient(_scenario(), quote="USDT")
    emitted = []
    bot = CCXTBot.__new__(CCXTBot)
    bot.exchange = "fake"
    bot.user = "fake_01"
    bot.bot_id = "fake_bot"
    bot.cca = client
    bot.open_orders = {}
    bot.recent_order_cancellations = []
    bot.log_order_action = lambda *args, **kwargs: None
    bot._log_order_action_summary = lambda *args, **kwargs: None
    bot._build_order_params = lambda _order: {
        "positionSide": "LONG",
        "clientOrderId": "pb-test",
    }
    bot._emit_live_event = lambda event_type, **kwargs: emitted.append(
        (event_type, kwargs)
    )
    requested = {
        "symbol": "BTC/USDT:USDT",
        "type": "limit",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "price": 90.0,
        "reduce_only": False,
        "custom_id": "pb-test",
    }

    order = await bot.execute_order(requested)
    await bot.execute_cancellation({**requested, "id": order["id"]})

    summary = _summarize_remote_calls(client.export_request_log())

    assert summary["by_method"]["create_order"] == 1
    assert summary["by_method"]["cancel_order"] == 1
    assert summary["by_category"]["order_write"] == 2
    assert [event_type for event_type, _kwargs in emitted] == [
        "execution.create_connector_call_started",
        "execution.create_sent",
        "execution.confirmation_requested",
        "execution.cancel_connector_call_started",
        "execution.cancel_sent",
        "execution.confirmation_requested",
    ]


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
    assert trace["hsl"]["observation_status"] == "not_evaluated"
    assert trace["hsl"]["scopes"] == []
    json.dumps(trace)


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
    cache = FillEventCache(cache_path)
    cached_events = cache.load()
    assert len(cached_events) == 1
    assert cached_events[0].client_order_id == "boot_entry"
    assert (
        cache.get_covered_start_ms()
        == client.get_fill_events(None, None)[0]["timestamp"]
    )
    assert cache.get_history_scope() == "window"


@pytest.mark.fake_live
def test_prime_fake_candles_seeds_candlestick_manager_cache():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    cm = type(
        "CM",
        (),
        {
            "_cache": {},
            "_ema_cache": {},
            "_current_close_cache": {},
            "_tf_range_cache": {},
        },
    )()
    bot = type("Bot", (), {"cm": cm})()
    _prime_fake_candles(bot, client)
    arr = bot.cm._cache["BTC/USDT:USDT"]
    assert len(arr) == 1
    assert int(arr[0]["ts"]) == 1_767_225_600_000
    assert float(arr[0]["c"]) == pytest.approx(100.0)


def test_bot_params_to_rust_dict_includes_hsl_fields():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.config = {
                "live": {"hsl_signal_mode": "coin"},
                "bot": {
                    "long": {
                        "hsl": {"enabled": True, "panic_close_order_type": "market"}
                    }
                },
            }
            self.coin_overrides = {}
            self.values = {
                "close_grid_qty_pct": 1.0,
                "close_trailing_retracement_pct": 0.001,
                "close_trailing_qty_pct": 0.0,
                "close_trailing_threshold_pct": 0.001,
                "close_weight_volatility_1h": 0.0,
                "close_weight_volatility_1m": 0.0,
                "entry_grid_double_down_factor": 1.0,
                "entry_weight_volatility_1h": 0.0,
                "entry_weight_volatility_1m": 0.0,
                "entry_we_weight": 0.0,
                "entry_grid_spacing_pct": 0.01,
                "entry_volatility_ema_span_1h": 0.0,
                "entry_volatility_ema_span_1m": 60.0,
                "entry_initial_ema_dist": -0.001,
                "entry_initial_qty_pct": 0.1,
                "entry_trailing_double_down_factor": 1.0,
                "entry_trailing_retracement_pct": 0.001,
                "entry_trailing_threshold_pct": 0.001,
                "filter_volatility_ema_span_1m": 0.0,
                "filter_volume_ema_span_1m": 1.0,
                "forager_volume_drop_pct": 0.0,
                "forager_score_weights": {
                    "volume": 1.0,
                    "ema_readiness": 0.0,
                    "volatility": 0.0,
                },
                "ema_span_0": 2.0,
                "ema_span_1": 4.0,
                "hsl_enabled": True,
                "hsl_panic_close_order_type": "market",
                "n_positions": 1.0,
                "total_wallet_exposure_limit": 5.0,
                "wallet_exposure_limit": 5.0,
                "risk_entry_cooldown_minutes": 0.0,
                "risk_wel_enforcer_threshold": 1.0,
                "risk_twel_enforcer_policy": "REDUCE_PORTFOLIO",
                "risk_twel_enforcer_threshold": 1.0,
                "risk_we_excess_allowance_pct": 0.0,
                "risk_we_excess_allowance_mode": "LEGACY_RAW",
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
    assert out["hsl_enabled"] is True
    assert out["hsl_panic_close_order_type"] == "market"
    assert out["risk_twel_enforcer_policy"] == "reduce_portfolio"
    assert out["risk_we_excess_allowance_mode"] == "legacy_raw"
    assert "entry_grid_inflation_enabled" not in out
    assert out["forager_score_weights"] == {
        "volume": pytest.approx(1.0),
        "ema_readiness": pytest.approx(0.0),
        "volatility": pytest.approx(0.0),
    }

    stub = _Stub()
    stub.values["risk_twel_enforcer_policy"] = "invalid_policy"
    with pytest.raises(ValueError, match="total_exposure_enforcer_policy"):
        Passivbot._bot_params_to_rust_dict(stub, "long", None)


def test_bot_params_to_rust_dict_ignores_removed_entry_grid_inflation_flag():
    from passivbot import Passivbot

    class _Stub:
        def __init__(self):
            self.config = {
                "bot": {
                    "long": {
                        "close_grid_qty_pct": 1.0,
                        "close_trailing_retracement_pct": 0.001,
                        "close_trailing_qty_pct": 0.0,
                        "close_trailing_threshold_pct": 0.001,
                        "close_weight_volatility_1h": 0.0,
                        "close_weight_volatility_1m": 0.0,
                        "entry_grid_double_down_factor": 1.0,
                        "entry_weight_volatility_1h": 0.0,
                        "entry_weight_volatility_1m": 0.0,
                        "entry_we_weight": 0.0,
                        "entry_grid_spacing_pct": 0.01,
                        "entry_volatility_ema_span_1h": 0.0,
                        "entry_volatility_ema_span_1m": 60.0,
                        "entry_initial_ema_dist": -0.001,
                        "entry_initial_qty_pct": 0.1,
                        "entry_trailing_double_down_factor": 1.0,
                        "entry_trailing_retracement_pct": 0.001,
                        "entry_trailing_threshold_pct": 0.001,
                        "forager_volatility_ema_span_1m": 0.0,
                        "forager_volume_ema_span_1m": 1.0,
                        "forager_volume_drop_pct": 0.0,
                        "forager_score_weights": {
                            "volume": 1.0,
                            "ema_readiness": 0.0,
                            "volatility": 0.0,
                        },
                        "ema_span_0": 2.0,
                        "ema_span_1": 4.0,
                        "hsl_enabled": True,
                        "hsl_panic_close_order_type": "market",
                        "n_positions": 1.0,
                        "total_wallet_exposure_limit": 5.0,
                        "wallet_exposure_limit": 5.0,
                        "risk_entry_cooldown_minutes": 0.0,
                        "risk_wel_enforcer_enabled": True,
                        "risk_wel_enforcer_threshold": 1.0,
                        "risk_twel_enforcer_enabled": True,
                        "risk_twel_enforcer_policy": "reduce_overweight",
                        "risk_twel_entry_gate_enabled": True,
                        "risk_twel_enforcer_threshold": 1.0,
                        "risk_we_excess_allowance_pct": 0.0,
                        "risk_we_excess_allowance_mode": "bounded",
                        "unstuck_close_pct": 0.01,
                        "unstuck_ema_dist": 0.0,
                        "unstuck_ema_span_0": 2.0,
                        "unstuck_ema_span_1": 4.0,
                        "unstuck_enabled": True,
                        "unstuck_ema_gating_enabled": True,
                        "unstuck_loss_allowance_pct": 0.1,
                        "unstuck_threshold": 1.0,
                    },
                    "short": {},
                }
            }
            self.config["live"] = {"hsl_signal_mode": "coin"}
            self.config["bot"]["long"]["hsl"] = {
                "enabled": True,
                "panic_close_order_type": "market",
            }
            self.coin_overrides = {
                "BTC/USDT:USDT": {
                    "bot": {
                        "long": {
                            "entry_grid_inflation_enabled": False,
                            "unstuck_loss_allowance_pct": 0.025,
                        }
                    }
                }
            }

        def bot_value(self, _pside, key):
            return self.config["bot"]["long"][key]

        def bp(self, pside, key, symbol=None):
            if symbol in self.coin_overrides:
                override = (
                    self.coin_overrides[symbol]
                    .get("bot", {})
                    .get(pside, {})
                    .get(key, None)
                )
                if override is not None:
                    return override
            return self.config["bot"][pside][key]

    out = Passivbot._bot_params_to_rust_dict(_Stub(), "long", "BTC/USDT:USDT")

    assert "entry_grid_inflation_enabled" not in out
    assert out["unstuck_loss_allowance_pct"] == pytest.approx(0.025)


def test_install_runtime_overrides_keep_scenario_time_after_client_shutdown():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    cm = type("CandlestickManager", (), {})()
    bot = type("Bot", (), {"cca": client, "cm": cm})()
    _install_runtime_overrides(bot, {})
    bot.cca = None

    assert bot.get_exchange_time() == client.now_ms
    assert bot.cm._now_ms_callback() == client.now_ms


def test_compare_run_artifacts_reports_no_diff_for_matching_payloads():
    payload = {
        "step_summaries": [{"step_index": 0, "fills": 0}],
        "fake_exchange_state": {"balance_total": 1000.0},
        "fills": [],
        "positions": [],
        "hsl_trace": {"long": {"halted": False}},
        "remote_call_summary": {"total_calls": 3, "by_method": {"fetch_balance": 1}},
    }
    report = _compare_run_artifacts(
        payload,
        {
            **payload,
            "remote_call_summary": {
                "total_calls": 5,
                "by_method": {"fetch_balance": 1, "fetch_tickers": 2},
            },
        },
    )
    assert report["match"] is True
    assert report["diff_count"] == 0
    assert report["remote_call_delta"] == 2
    assert report["remote_call_delta_by_method"] == {
        "fetch_balance": 0,
        "fetch_tickers": 2,
    }


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
                "last_stop_event": {
                    "triggered_at": "a",
                    "user": "legacy_user",
                    "tier": "red",
                },
            }
        },
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
                "last_stop_event": {
                    "triggered_at": "b",
                    "user": "staged_user",
                    "tier": "red",
                },
            }
        },
    }

    report = _compare_run_artifacts(legacy, staged)

    assert report["match"] is True
    assert report["diff_count"] == 0


def test_load_run_artifacts_reads_expected_files(tmp_path):
    (tmp_path / "step_summaries.json").write_text(
        json.dumps([{"step_index": 0}]), encoding="utf-8"
    )
    (tmp_path / "fake_exchange_state.json").write_text(
        json.dumps({"balance_total": 1}), encoding="utf-8"
    )
    (tmp_path / "fills.json").write_text("[]", encoding="utf-8")
    (tmp_path / "positions.json").write_text("[]", encoding="utf-8")
    (tmp_path / "hsl_trace.json").write_text(
        json.dumps({"long": {"halted": False}}), encoding="utf-8"
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"user": "fake"}), encoding="utf-8"
    )
    (tmp_path / "remote_calls.json").write_text(
        json.dumps([{"method": "fetch_balance"}]), encoding="utf-8"
    )
    (tmp_path / "remote_call_summary.json").write_text(
        json.dumps({"total_calls": 1}), encoding="utf-8"
    )
    (tmp_path / "candle_remote_fetches.json").write_text(
        json.dumps([{"kind": "ccxt_fetch_ohlcv"}]), encoding="utf-8"
    )
    (tmp_path / "live_events.json").write_text(
        json.dumps([{"event_type": "snapshot.built"}]), encoding="utf-8"
    )
    (tmp_path / "fake_live.log").write_text("hello\n", encoding="utf-8")

    loaded = _load_run_artifacts(tmp_path)

    assert loaded["step_summaries"][0]["step_index"] == 0
    assert loaded["fake_exchange_state"]["balance_total"] == 1
    assert loaded["remote_calls"][0]["method"] == "fetch_balance"
    assert loaded["remote_call_summary"]["total_calls"] == 1
    assert loaded["candle_remote_fetches"][0]["kind"] == "ccxt_fetch_ohlcv"
    assert loaded["live_events"][0]["event_type"] == "snapshot.built"
    assert loaded["log_text"] == "hello\n"


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_coin_overrides_resolve_in_offline_restart_harness(tmp_path, monkeypatch):
    """Exercise file and inline coin patches through a real fake-live restart scenario."""
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    user = f"fake_coin_overrides_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    base_config = hjson.loads(
        (REPO_ROOT / "configs" / "examples" / "fake_live_hsl.json").read_text(
            encoding="utf-8"
        )
    )
    global_threshold = base_config["bot"]["long"]["strategy"]["trailing_martingale"][
        "entry"
    ]["threshold_base_pct"]
    override_path = tmp_path / "btc_override.hjson"
    override_path.write_text(
        hjson.dumps(
            {
                "bot": {
                    "long": {
                        "risk": {"entry_cooldown_minutes": 3.0},
                        "unstuck": {
                            "ema_gating_enabled": False,
                            "loss_allowance_pct": 0.1,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    base_config["coin_overrides"] = {
        "BTC": {
            "override_config_path": override_path.name,
            "bot": {
                "long": {
                    "risk": {"entry_cooldown_minutes": 0.05},
                    "strategy": {
                        "trailing_martingale": {
                            "entry": {"threshold_base_pct": global_threshold}
                        }
                    },
                }
            },
        }
    }
    config_path = tmp_path / "fake_live_coin_overrides.hjson"
    config_path.write_text(hjson.dumps(base_config), encoding="utf-8")

    captured = {}
    real_setup_bot = run_fake_live_module.setup_bot

    def capture_setup_bot(config):
        bot = real_setup_bot(config)
        captured["bot"] = bot
        return bot

    monkeypatch.setattr(run_fake_live_module, "setup_bot", capture_setup_bot)
    try:
        args = argparse.Namespace(
            config=str(config_path),
            scenario=str(
                REPO_ROOT / "scenarios" / "fake_live" / "hsl_long_red_restart.hjson"
            ),
            user=user,
            max_steps=None,
            output_dir=str(tmp_path / "artifacts"),
            log_level=1,
            snapshot_each_step=False,
        )
        assert await _async_main(args) == 0

        bot = captured["bot"]
        override = bot.coin_overrides["BTC/USDT:USDT"]
        assert override["bot"]["long"]["risk"]["entry_cooldown_minutes"] == 0.05
        assert override["bot"]["long"]["risk_entry_cooldown_minutes"] == 0.05
        assert override["bot"]["long"]["unstuck"]["ema_gating_enabled"] is False
        assert override["bot"]["long"]["unstuck_ema_gating_enabled"] is False
        assert override["bot"]["long"]["unstuck"]["loss_allowance_pct"] == 0.1
        assert (
            override["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
                "threshold_base_pct"
            ]
            == global_threshold
        )
        transform_steps = [item["step"] for item in bot.config["_transform_log"]]
        assert transform_steps.index("parse_overrides") < transform_steps.index(
            "compile_runtime_config"
        )
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_live_all_lookback_backfills_narrow_fill_cache_once(
    tmp_path, monkeypatch
):
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    user = "fake_hsl_pnls_lookback_all_test"
    user = f"{user}_{tmp_path.name}"
    scenario_path = REPO_ROOT / "scenarios" / "fake_live" / "hsl_long_red_restart.hjson"
    cache_dir = REPO_ROOT / "caches" / "fill_events" / "fake" / user
    _cleanup_fake_user_state(user)

    cfg = load_config(
        str(REPO_ROOT / "configs" / "examples" / "fake_live_hsl.json"), verbose=False
    )
    cfg["bot"]["long"]["hsl"]["enabled"] = False
    cfg["live"]["pnls_max_lookback_days"] = "all"
    cfg["live"]["hsl_signal_mode"] = "pside"
    config_path = tmp_path / "fake_live_hsl_btc_all_lookback.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    scenario_copy_path = tmp_path / "hsl_long_red_restart_no_assertions.hjson"
    scenario_cfg = hjson.loads(scenario_path.read_text(encoding="utf-8"))
    scenario_cfg.pop("assertions", None)
    scenario_copy_path.write_text(hjson.dumps(scenario_cfg), encoding="utf-8")

    def _prime_narrow_window_cache(bot, fake_client, cache_root=None):
        root = (
            Path(cache_root)
            if cache_root is not None
            else Path("caches") / "fill_events"
        )
        cache_path = root / str(bot.exchange) / str(bot.user)
        shutil.rmtree(cache_path, ignore_errors=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        all_events = [
            FillEvent.from_dict(event)
            for event in fake_client.get_fill_events(None, None)
        ]
        narrow_events = [event for event in all_events if str(event.id) != "10"]
        cache = FillEventCache(cache_path)
        cache.save(narrow_events)
        cache.update_metadata_from_events(narrow_events)
        cache.set_history_scope("window")
        return cache_path

    monkeypatch.setattr(
        run_fake_live_module, "_prime_fake_fill_cache", _prime_narrow_window_cache
    )

    args = argparse.Namespace(
        config=str(config_path),
        scenario=str(scenario_copy_path),
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
        fill_ref = hashlib.sha256(b"10").hexdigest()[:12]
        assert (
            sum(
                "[fill]" in line and f" id={fill_ref}" in line
                for line in log_text.splitlines()
            )
            == 1
        )
        assert not any(
            "[fill]" in line and " id=10" in line for line in log_text.splitlines()
        )

        cache = FillEventCache(cache_dir)
        cached_ids = [str(event.id) for event in cache.load()]
        assert cache.get_history_scope() == "all"
        assert "10" in cached_ids
        assert {"10", "11", "12"}.issubset(set(cached_ids))
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_live_min_effective_cost_blocks_zero_min_qty_integer_step_symbol(
    tmp_path,
):
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

    cfg = load_config(
        str(REPO_ROOT / "configs" / "examples" / "fake_live_hsl.json"), verbose=False
    )
    cfg["bot"]["long"]["hsl_enabled"] = False
    cfg["bot"]["short"]["hsl_enabled"] = False
    cfg["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "initial_qty_pct"
    ] = 0.0276
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

    output_dirs = sorted(
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and (path / "remote_calls.json").exists()
    )
    assert len(output_dirs) == 1
    run_dir = output_dirs[0]

    step_summaries = json.loads(
        (run_dir / "step_summaries.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (run_dir / "fake_exchange_state.json").read_text(encoding="utf-8")
    )
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


@pytest.mark.asyncio
@pytest.mark.fake_live
async def test_fake_live_writes_remote_call_artifacts(tmp_path, monkeypatch):
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    monkeypatch.chdir(tmp_path)
    scenario = hjson.loads(
        (
            REPO_ROOT / "scenarios" / "fake_live" / "hsl_long_red_restart.hjson"
        ).read_text(encoding="utf-8")
    )
    scenario["assertions"] = {
        "remote_call_paths": {
            "summary.total_calls": {"min": 1},
            "summary.by_method.fetch_balance": {"min": 1},
            "summary.by_method.fetch_positions": {"min": 1},
            "summary.by_method.fetch_open_orders": {"min": 1},
            "summary.by_method.fetch_ohlcv": {"min": 1},
            "summary.by_category.account_state": {"min": 3},
            "summary.by_category.market_data": {"min": 1},
        }
    }
    scenario_path = tmp_path / "fake_remote_call_trace.hjson"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    config_path = REPO_ROOT / "configs" / "examples" / "fake_live_hsl.json"

    args = argparse.Namespace(
        config=str(config_path),
        scenario=str(scenario_path),
        user="fake_remote_call_trace",
        max_steps=1,
        output_dir=str(tmp_path),
        log_level=1,
        snapshot_each_step=False,
    )
    assert await _async_main(args) == 0

    output_dirs = sorted(
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and (path / "remote_calls.json").exists()
    )
    assert len(output_dirs) == 1
    run_dir = output_dirs[0]

    remote_calls = json.loads(
        (run_dir / "remote_calls.json").read_text(encoding="utf-8")
    )
    remote_summary = json.loads(
        (run_dir / "remote_call_summary.json").read_text(encoding="utf-8")
    )
    candle_fetches = json.loads(
        (run_dir / "candle_remote_fetches.json").read_text(encoding="utf-8")
    )

    methods = {entry["method"] for entry in remote_calls}
    assert "fetch_balance" in methods
    assert "fetch_positions" in methods
    assert "fetch_open_orders" in methods
    assert "fetch_ohlcv" in methods

    assert remote_summary["total_calls"] == len(remote_calls)
    assert remote_summary["by_method"]["fetch_balance"] >= 1
    assert remote_summary["by_method"]["fetch_positions"] >= 1
    assert remote_summary["by_method"]["fetch_open_orders"] >= 1
    assert remote_summary["by_method"]["fetch_ohlcv"] >= 1

    assert any(entry["kind"] == "ccxt_fetch_ohlcv" for entry in candle_fetches)


@pytest.mark.fake_live
@pytest.mark.parametrize("declared", [False, True])
def test_fake_boot_fill_preserves_only_explicit_position_chain_evidence(declared):
    scenario = _scenario()
    fill = {
        "id": "1",
        "timestamp": "2026-01-01T00:00:00Z",
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": "sell",
        "amount": 1.0,
        "price": 100.0,
    }
    if declared:
        fill["info"] = {"startPosition": "1.0"}
    scenario["account"]["fills"] = [fill]
    client = FakeCCXTClient(scenario, quote="USDT")
    info = client.fills[0]["info"]
    assert ("startPosition" in info) is declared
    if declared:
        assert info["startPosition"] == "1.0"


def test_replay_comparison_ignores_only_hsl_scheduler_pass_count():
    from copy import deepcopy

    result = dict(
        updated=True,
        ordinary_completed=True,
        ordinary_executed=True,
        protective_work=False,
        engine="hsl",
        passes=2,
    )
    left = {
        "step_summaries": [
            dict(step_index=1, result=str(result), fills=1, open_orders=0, positions=[])
        ]
    }
    left.update(fake_exchange_state={}, fills=[], positions=[], hsl_trace={})
    right = deepcopy(left)
    right["step_summaries"][0]["result"] = str(dict(result, passes=3))
    original = deepcopy(right)
    assert _compare_run_artifacts(left, right)["match"]
    assert right == original  # Full diagnostics remain in the saved artifacts.
    for key, value in [
        ("updated", False),
        ("ordinary_completed", False),
        ("ordinary_executed", False),
        ("protective_work", True),
        ("preparation_pending", True),
        ("current_io_unavailable", True),
    ]:
        right["step_summaries"][0]["result"] = str(dict(result, **{key: value}))
        assert not _compare_run_artifacts(left, right)["match"], key
    right["step_summaries"][0]["result"] = str(dict(result, passes=3))
    right["step_summaries"][0]["fills"] = 2
    assert not _compare_run_artifacts(left, right)["match"]


def test_replay_comparison_retains_nonhsl_and_malformed_results():
    for result in ["opaque", "{bad", "{'engine': 'legacy', 'passes': 2}"]:
        left = {"step_summaries": [{"result": result}]}
        right = {"step_summaries": [{"result": result + " changed"}]}
        for payload in (left, right):
            payload.update(fake_exchange_state={}, fills=[], positions=[], hsl_trace={})
        assert not _compare_run_artifacts(left, right)["match"]


def test_fake_assertions_reject_retired_halt_state_instead_of_reading_legacy_controller():
    with pytest.raises(ValueError, match="use hsl_paths"):
        _apply_assertions(
            _StubBot(),
            FakeCCXTClient(_scenario(), quote="USDT"),
            {"assertions": {"halted_psides": {"long": False}}},
            step_summaries=[],
            log_text="",
        )
