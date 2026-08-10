import asyncio
import logging
import math
from types import MethodType, SimpleNamespace

import pytest

from fill_events_manager import FillEvent, FillEventsManager
from passivbot import Passivbot
from passivbot_exceptions import RestartBotException
from live.event_bus import ReasonCodes

pbr = pytest.importorskip("passivbot_rust", reason="passivbot_rust extension not available")
hsl = pytest.importorskip("passivbot_hsl", reason="live HSL dependencies not available")

if bool(getattr(pbr, "__is_stub__", False)):
    pytest.skip("passivbot_rust extension not available", allow_module_level=True)


class FakeHslBot(SimpleNamespace):
    pass


def test_hsl_event_emitter_failure_logs_type_without_secret(caplog):
    secret = "api_secret=hsl-event-secret"
    url = "https://private.example.invalid/v1/hsl?token=hsl-event-token"

    class HslEventEmitterFailure(RuntimeError):
        pass

    def fail_emit(*_args, **_kwargs):
        raise HslEventEmitterFailure(f"request failed {secret} {url}")

    bot = SimpleNamespace(
        _live_event_pipeline=object(),
        _live_event_current_cycle_id="cy_hsl_redaction",
        _emit_live_event=fail_emit,
    )

    with caplog.at_level(logging.DEBUG):
        emitted = hsl._emit_hsl_event(
            bot,
            "hsl.status",
            ("hsl", "risk"),
            {},
            pside="long",
            symbol="BTC/USDT:USDT",
        )

    assert emitted is None
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text
    assert url not in caplog.text

    sensitive_identifier = "ApiKey_prod_super_secret_HSL123"
    sensitive_identifier_type = type(sensitive_identifier, (RuntimeError,), {})

    def fail_with_sensitive_identifier(*_args, **_kwargs):
        raise sensitive_identifier_type("safe")

    bot._emit_live_event = fail_with_sensitive_identifier
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert (
            hsl._emit_hsl_event(
                bot,
                "hsl.status",
                ("hsl", "risk"),
                {},
                pside="long",
                symbol="BTC/USDT:USDT",
            )
            is None
        )

    assert "Error" in caplog.text
    assert sensitive_identifier not in caplog.text

    camelcase_sensitive_identifier = "ApiKeyProdSecretHSL123"
    camelcase_sensitive_type = type(
        camelcase_sensitive_identifier, (RuntimeError,), {}
    )

    def fail_with_camelcase_sensitive_identifier(*_args, **_kwargs):
        raise camelcase_sensitive_type("safe")

    bot._emit_live_event = fail_with_camelcase_sensitive_identifier
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert (
            hsl._emit_hsl_event(
                bot,
                "hsl.status",
                ("hsl", "risk"),
                {},
                pside="long",
                symbol="BTC/USDT:USDT",
            )
            is None
        )

    assert "Error" in caplog.text
    assert camelcase_sensitive_identifier not in caplog.text

    class HostileName(str):
        def __getitem__(self, _key):
            return "api_secret=hostile-hsl-slice-secret"

    class HostileSliceMeta(type):
        @property
        def __name__(cls):
            return HostileName("SafeLookingHslFailure")

    hostile_slice_type = HostileSliceMeta(
        "HostileSliceHslFailure", (RuntimeError,), {}
    )

    def fail_with_hostile_slice(*_args, **_kwargs):
        raise hostile_slice_type("safe")

    bot._emit_live_event = fail_with_hostile_slice
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert (
            hsl._emit_hsl_event(
                bot,
                "hsl.status",
                ("hsl", "risk"),
                {},
                pside="long",
                symbol="BTC/USDT:USDT",
            )
            is None
        )

    assert "Error" in caplog.text
    assert "hostile-hsl-slice-secret" not in caplog.text

    invalid_suffix = "H" * 80 + "\napi_secret=tail-hsl-class-name-secret"
    invalid_suffix_type = type(invalid_suffix, (RuntimeError,), {})

    def fail_with_invalid_suffix(*_args, **_kwargs):
        raise invalid_suffix_type("safe")

    bot._emit_live_event = fail_with_invalid_suffix
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert (
            hsl._emit_hsl_event(
                bot,
                "hsl.status",
                ("hsl", "risk"),
                {},
                pside="long",
                symbol="BTC/USDT:USDT",
            )
            is None
        )

    assert "Error" in caplog.text
    assert "tail-hsl-class-name-secret" not in caplog.text

    class HostileExceptionMeta(type):
        @property
        def __name__(cls):
            raise KeyboardInterrupt("api_secret=hostile-hsl-property-secret")

    hostile_type = HostileExceptionMeta(
        "HostileHslEventEmitterFailure", (RuntimeError,), {}
    )

    def fail_with_hostile_type(*_args, **_kwargs):
        raise hostile_type("api_secret=hostile-hsl-metadata-secret")

    bot._emit_live_event = fail_with_hostile_type
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert (
            hsl._emit_hsl_event(
                bot,
                "hsl.status",
                ("hsl", "risk"),
                {},
                pside="long",
                symbol="BTC/USDT:USDT",
            )
            is None
        )

    assert "Error" in caplog.text
    assert "hostile-hsl-metadata-secret" not in caplog.text
    assert "hostile-hsl-property-secret" not in caplog.text


class FakeRiskCache:
    def __init__(
        self,
        covered_start_ms=1,
        history_scope="all",
        oldest_event_ts=0,
        newest_event_ts=0,
    ):
        self.covered_start_ms = covered_start_ms
        self.history_scope = history_scope
        self.oldest_event_ts = oldest_event_ts
        self.newest_event_ts = newest_event_ts

    def get_known_gaps(self):
        return []

    def get_covered_start_ms(self):
        return self.covered_start_ms

    def get_history_scope(self):
        return self.history_scope

    def load_metadata(self):
        return {
            "known_gaps": [],
            "covered_start_ms": self.covered_start_ms,
            "history_scope": self.history_scope,
            "oldest_event_ts": self.oldest_event_ts,
            "newest_event_ts": self.newest_event_ts,
        }


def make_fake_pnls_manager(events, *, covered_start_ms=1, history_scope="all"):
    timestamps = [int(getattr(event, "timestamp", 0) or 0) for event in events]
    cache = FakeRiskCache(
        covered_start_ms=covered_start_ms,
        history_scope=history_scope,
        oldest_event_ts=min(timestamps, default=0),
        newest_event_ts=max(timestamps, default=0),
    )
    manager = SimpleNamespace(
        _events=list(events),
        get_events=lambda: events,
        cache=cache,
        get_history_scope=cache.get_history_scope,
    )
    manager.get_coverage_status = MethodType(
        FillEventsManager.get_coverage_status,
        manager,
    )
    return manager


def test_hsl_signal_mode_requires_normalized_live_config():
    bot = FakeHslBot(config={"live": {"hsl_signal_mode": "coin"}})

    assert hsl._equity_hard_stop_signal_mode(bot) == "coin"

    bot.config = {"live": {}}
    with pytest.raises(KeyError, match="live.hsl_signal_mode"):
        hsl._equity_hard_stop_signal_mode(bot)


def test_hsl_coin_replay_candidate_batches_are_frozen_and_scope_prioritized():
    active = [
        ("long", "A"),
        ("long", "Z"),
        ("short", "B"),
        ("long", "C"),
    ]
    held = {("long", "Z"), ("short", "B")}
    cooldown = {("long", "Z"), ("long", "C")}

    batches = hsl._hsl_coin_replay_candidate_batches(active, held, cooldown)

    active.append(("short", "NEW"))
    held.clear()
    cooldown.clear()
    assert batches == (
        (("long", "Z"), ("short", "B")),
        (("long", "C"),),
        (("long", "A"),),
    )


def test_coin_hsl_fill_index_preserves_pair_order_and_event_identity():
    long_first = {
        "timestamp": 30,
        "symbol": "A",
        "pside": "long",
        "action": "increase",
        "qty": 1.0,
    }
    short_fill = {
        "timestamp": 20,
        "symbol": "A",
        "pside": "short",
        "action": "increase",
        "qty": 2.0,
    }
    long_second = {
        "timestamp": 10,
        "symbol": "A",
        "pside": "long",
        "action": "decrease",
        "qty": 1.0,
    }

    indexed = hsl._equity_hard_stop_index_coin_fill_events(
        [long_first, short_fill, long_second]
    )

    assert indexed == {
        ("long", "A"): [long_first, long_second],
        ("short", "A"): [short_fill],
    }
    assert indexed[("long", "A")][0] is long_first
    assert indexed[("long", "A")][1] is long_second


def test_coin_hsl_fill_index_rejects_conflicting_pside_aliases():
    with pytest.raises(ValueError, match="conflicting pside aliases"):
        hsl._equity_hard_stop_index_coin_fill_events(
            [
                {
                    "timestamp": 1,
                    "symbol": "A",
                    "pside": "long",
                    "position_side": "short",
                    "action": "increase",
                    "qty": 1.0,
                }
            ]
        )


def test_coin_hsl_fill_index_keeps_replay_and_contract_results_identical():
    fills = [
        {
            "timestamp": 1,
            "symbol": "A",
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
        },
        {
            "timestamp": 2,
            "symbol": "B",
            "pside": "short",
            "action": "increase",
            "qty": 3.0,
        },
        {
            "timestamp": 3,
            "symbol": "A",
            "pside": "long",
            "action": "decrease",
            "qty": 1.0,
            "pb_order_type": "panic_close",
        },
        {
            "timestamp": 4,
            "symbol": "A",
            "pside": "long",
            "action": "increase",
            "qty": 0.5,
        },
    ]
    pair_fills = hsl._equity_hard_stop_index_coin_fill_events(fills)[("long", "A")]
    bot = FakeHslBot(
        hsl={"long": {"cooldown_minutes_after_red": 1.0}},
        positions={"A": {"long": {"size": 0.5}}},
    )
    bot._equity_hard_stop_cooldown_position_policy = lambda: "normal"
    bot._equity_hard_stop_has_open_position_symbol = lambda pside, symbol: True

    assert hsl._equity_hard_stop_coin_replay_events(
        pair_fills, "long", "A"
    ) == hsl._equity_hard_stop_coin_replay_events(fills, "long", "A")
    assert hsl._equity_hard_stop_infer_coin_replay_contract(
        bot, "long", "A", pair_fills, 30_000
    ) == hsl._equity_hard_stop_infer_coin_replay_contract(
        bot, "long", "A", fills, 30_000
    )


def test_compact_sparse_replay_indices_keep_run_and_explicit_boundaries():
    import numpy as np

    timestamps = np.arange(20, dtype=np.int64) * 60_000
    constant = np.zeros(20, dtype=np.float64)

    indices = hsl._hsl_compact_sparse_replay_indices(
        timestamps,
        np.full(20, 100.0, dtype=np.float64),
        constant,
        constant,
        lookback_ms=30 * 24 * 60 * 60 * 1_000,
        boundary_timestamps=(10 * 60_000,),
    )

    assert indices.tolist() == [0, 9, 10, 19]


def test_compact_sparse_replay_indices_keep_rolling_window_expiry_boundary():
    import numpy as np

    timestamps = np.arange(10, dtype=np.int64) * 60_000
    realized = np.asarray([0.0, 0.0, 0.0] + [1.0] * 7, dtype=np.float64)

    indices = hsl._hsl_compact_sparse_replay_indices(
        timestamps,
        np.full(10, 100.0, dtype=np.float64),
        realized,
        np.zeros(10, dtype=np.float64),
        lookback_ms=2 * 60_000,
    )

    assert indices.tolist() == [0, 2, 3, 4, 5, 9]


def test_parse_hsl_config_logs_compact_complete_startup_summary(caplog):
    bot = FakeHslBot(config={"live": {"hsl_signal_mode": "unified"}})
    values = {
        "hsl_enabled": True,
        "hsl_red_threshold": 0.123456789,
        "hsl_ema_span_minutes": 1.23456789e308,
        "hsl_cooldown_minutes_after_red": 1.23456789e308,
        "hsl_no_restart_drawdown_threshold": 0.987654321,
        "hsl_tier_ratios": {"yellow": 0.3456789, "orange": 0.8765432},
        "hsl_tier_ratios.yellow": 0.3456789,
        "hsl_tier_ratios.orange": 0.8765432,
        "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
        "hsl_panic_close_order_type": "market",
        "hsl_restart_after_red_policy": "threshold",
    }
    bot._hsl_psides = lambda: ["short"]
    bot._equity_hard_stop_signal_mode = MethodType(
        hsl._equity_hard_stop_signal_mode,
        bot,
    )
    bot.bot_value = lambda pside, key: values[key]

    with caplog.at_level(logging.INFO):
        parsed = hsl._parse_hsl_config(bot)

    assert parsed == {
        "short": {
            "enabled": True,
            "red_threshold": 0.123456789,
            "ema_span_minutes": 1.23456789e308,
            "cooldown_minutes_after_red": 1.23456789e308,
            "no_restart_drawdown_threshold": 0.987654321,
            "tier_ratios": {"yellow": 0.3456789, "orange": 0.8765432},
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
            "restart_after_red_policy": "threshold",
        }
    }
    messages = [record.getMessage() for record in caplog.records]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    warning = warnings[0].getMessage()
    assert warning == (
        "[risk] HSL[short] enabled; review docs/equity_hard_stop_loss_risks.md. "
        "Deposits, withdrawals, balance overrides, and HSL mode/budget/threshold "
        "changes can reinterpret reconstructed history."
    )
    startup = next(message for message in messages if message.startswith("[risk] HSL[short] on"))
    assert startup == (
        "[risk] HSL[short] on | red=0.123457 ema=1.23457e+308 cd=1.23457e+308 "
        "no-r=0.987654 mode=unified tiers=0.345679/0.876543 "
        "orange=tp_only_with_active_entry_cancellation panic=market restart=threshold"
    )
    warning_prefix = "2026-07-15T12:34:56Z WARNING  [hyperliquid] "
    info_prefix = "2026-07-15T12:34:56Z INFO     [hyperliquid] "
    assert len(warning_prefix + warning) <= 240
    assert len(info_prefix + startup) <= 240


def test_coin_panic_supervision_requires_red_active_now():
    # B2.1 red split: a latched red episode authorizes panic supervision only
    # while the CURRENT sample is in RED.
    bot = make_coin_bot()
    symbol = "A"
    state = bot._hsl_coin_state("long", symbol)
    state["runtime"].apply_sample(
        timestamp_ms=60_000, equity=100.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    state["runtime"].apply_sample(
        timestamp_ms=120_000, equity=70.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    assert state["runtime"].red_latched() is True

    # No metrics yet against the latched state: stay protective.
    state["last_metrics"] = None
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is True
    )
    # Current sample in RED: panic authorized.
    state["last_metrics"] = {"red_active_now": True}
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is True
    )
    # Current sample recovered: no new panic orders for this episode.
    state["last_metrics"] = {"red_active_now": False}
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is False
    )
    # Halted repanic-reset supervision is unaffected by the split.
    state["halted"] = True
    state["cooldown_repanic_reset_pending"] = True
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is True
    )


@pytest.mark.asyncio
async def test_recovered_red_episode_finalizes_from_check_path(caplog):
    # Codex blocker regression on the red split: RED latches, the sample
    # recovers while the position is open (panic pauses, tp-only holds), the
    # position later flattens normally, and the episode MUST still be
    # finalized (halt + cooldown) by the regular check path without RED ever
    # re-activating and without the panic supervisor running.
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }
    bot.open_orders = {}
    bot.active_symbols = [symbol]
    upnl_box = {"value": 0.0}

    async def calc_upnl(pside=None, sym=None):
        return upnl_box["value"]

    bot._calc_upnl_sum_strict = calc_upnl
    now_box = {"ts": 60_000}
    bot.get_exchange_time = lambda: now_box["ts"]

    # Warmup green sample, then crash through RED (slot budget 50, red 0.5).
    await bot._equity_hard_stop_check_coin()
    now_box["ts"] = 120_000
    upnl_box["value"] = -30.0
    out = await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["tier"] == "red"
    assert state["runtime"].red_latched() is True
    assert bot._runtime_forced_modes["long"][symbol] == "panic"

    # Recovery while the position is still open: panic pauses, tp-only holds,
    # and the panic supervisor is NOT needed.
    now_box["ts"] = 180_000
    upnl_box["value"] = 0.0
    out = await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["red_active_now"] is False
    assert (
        bot._runtime_forced_modes["long"][symbol]
        == "tp_only_with_active_entry_cancellation"
    )
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is False
    )
    assert state["halted"] is False
    assert state["pending_red_since_ms"] == 120_000

    # The position closes normally under tp-only. Flat exchange state without
    # its close fill is not enough to invent a cooldown anchor.
    bot.positions = {
        symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0}}
    }
    bot.active_symbols = []
    now_box["ts"] = 240_000
    await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert state["red_flat_confirmations"] == 0
    assert state["halted"] is False
    assert state["pending_red_since_ms"] == 120_000
    assert "flatten-fill evidence is unavailable" in caplog.text
    bot._pnls_manager = make_fake_pnls_manager(
        [
            {
                "timestamp": 230_000,
                "symbol": symbol,
                "pside": "long",
                "pb_order_type": "close_grid_long",
                "pnl": 0.0,
                "fee_paid": 0.0,
            }
        ]
    )
    now_box["ts"] = 300_000
    await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert state["red_flat_confirmations"] == 1
    assert state["halted"] is False
    now_box["ts"] = 360_000
    await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] is not None
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 230_000


def test_forced_mode_refresher_preserves_paused_red(monkeypatch):
    # Hermes finding on the red split: the centralized refresher used to
    # overwrite the paused tp-only modes back to panic for any latched
    # non-halted pside. It must derive panic vs paused from the latest
    # sample's red_active_now, staying protective when no sample exists.
    bot = make_coin_bot()
    bot.positions = {"A": {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.active_symbols = ["A"]
    state = bot._hsl_state("long")
    state["runtime"].apply_sample(
        timestamp_ms=60_000, equity=100.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    state["runtime"].apply_sample(
        timestamp_ms=120_000, equity=70.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    assert state["runtime"].red_latched() is True

    # No sample recorded: protective panic.
    state["last_metrics"] = None
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    assert bot._runtime_forced_modes["long"]["A"] == "panic"

    # Active sample: panic.
    state["last_metrics"] = {"red_active_now": True}
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    assert bot._runtime_forced_modes["long"]["A"] == "panic"

    # Recovered sample: the paused tp-only modes survive the refresher.
    state["last_metrics"] = {"red_active_now": False}
    bot._equity_hard_stop_set_red_paused_runtime_forced_modes("long")
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    assert (
        bot._runtime_forced_modes["long"]["A"]
        == "tp_only_with_active_entry_cancellation"
    )


def _incomplete_history_bot(*, policy="threshold", override=False, fills=(), position_size=1.0):
    bot = make_coin_bot()
    bot.get_exchange_time = lambda: 1_000_000
    bot.hsl["long"]["restart_after_red_policy"] = policy
    if override:
        bot.config["live"]["hsl_accept_incomplete_history"] = True
    bot.positions = {
        "A": {"long": {"size": position_size, "price": 100.0}, "short": {"size": 0.0}}
    }

    class _Cache:
        def load_metadata(self):
            return {
                "covered_start_ms": 1,
                "oldest_event_ts": 1,
                "newest_event_ts": 1,
                "known_gaps": [],
            }

        def get_covered_start_ms(self):
            return 1

        def get_known_gaps(self):
            return []

        def get_history_scope(self):
            return "window"

    class _Manager:
        cache = _Cache()

        def __init__(self, events):
            self._events = list(events)

        def get_events(self, start_ms=None):
            if start_ms is None:
                return list(self._events)
            return [
                e for e in self._events if getattr(e, "timestamp", 0) >= start_ms
            ]

        def get_history_scope(self):
            return "window"

    bot._pnls_manager = _Manager(fills)
    bot._pnls_manager.get_coverage_status = MethodType(
        FillEventsManager.get_coverage_status,
        bot._pnls_manager,
    )
    return bot


def _episode_fill(ts, action, qty):
    return SimpleNamespace(
        position_side="long",
        symbol="A",
        timestamp=ts,
        action=action,
        qty=qty,
        pnl=0.0,
    )


def _set_coin_hsl_override(bot, symbol, **changes):
    effective = dict(bot.hsl["long"])
    effective["tier_ratios"] = dict(effective["tier_ratios"])
    effective.update(changes)
    bot.coin_overrides = {symbol: {}}
    values = {
        "hsl_cooldown_minutes_after_red": effective["cooldown_minutes_after_red"],
        "hsl_ema_span_minutes": effective["ema_span_minutes"],
        "hsl_enabled": effective["enabled"],
        "hsl_no_restart_drawdown_threshold": effective[
            "no_restart_drawdown_threshold"
        ],
        "hsl_orange_tier_mode": effective["orange_tier_mode"],
        "hsl_panic_close_order_type": effective["panic_close_order_type"],
        "hsl_red_threshold": effective["red_threshold"],
        "hsl_restart_after_red_policy": effective["restart_after_red_policy"],
        "hsl_tier_ratios": effective["tier_ratios"],
    }
    bot.bp = lambda pside, key, scope_symbol=None: (
        values[key]
        if pside == "long" and scope_symbol == symbol
        else bot.hsl[pside][key.removeprefix("hsl_")]
    )


def test_incomplete_history_policy_gates_hsl_coverage(caplog):
    import logging as logging_module

    # Coverage is unproven in every case below (covered_start_ms=0, window).
    # Fills reconstruct the current episode: flat -> 1.0 long.
    episode_fills = [_episode_fill(60_000, "increase", 1.0)]

    # threshold: hard-fail preserved.
    bot = _incomplete_history_bot(policy="threshold", fills=episode_fills)
    with pytest.raises(Exception):
        bot._equity_hard_stop_coin_realized_pnl_peak_last("long", "A", 120_000)

    # never: hard-fail preserved.
    bot = _incomplete_history_bot(policy="never", fills=episode_fills)
    with pytest.raises(Exception):
        bot._equity_hard_stop_coin_realized_pnl_peak_last("long", "A", 120_000)

    # always + provable episode start: proceeds with a critical log.
    bot = _incomplete_history_bot(policy="always", fills=episode_fills)
    with caplog.at_level(logging_module.CRITICAL):
        peak, last = bot._equity_hard_stop_coin_realized_pnl_peak_last(
            "long", "A", 120_000
        )
    assert "INCOMPLETE fill history" in caplog.text
    caplog.clear()

    # always + unprovable episode (fills never reach flat): hard-fail.
    partial_fills = [_episode_fill(60_000, "increase", 0.4)]
    bot = _incomplete_history_bot(policy="always", fills=partial_fills)
    with pytest.raises(Exception):
        bot._equity_hard_stop_coin_realized_pnl_peak_last("long", "A", 120_000)

    # Explicit per-run override: proceeds loudly regardless of policy.
    bot = _incomplete_history_bot(
        policy="never", override=True, fills=partial_fills
    )
    with caplog.at_level(logging_module.CRITICAL):
        bot._equity_hard_stop_coin_realized_pnl_peak_last("long", "A", 120_000)
    assert "INCOMPLETE fill history" in caplog.text


def test_coin_always_fill_requirement_covers_held_episode_and_flat_cooldown():
    pnl_start_ms = 100_000
    now_ms = 1_000_000

    # Flat scopes need only the still-relevant cooldown horizon when no fill
    # in that horizon suggests an episode requiring replay.
    bot = _incomplete_history_bot(
        policy="always", fills=(), position_size=0.0
    )
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, 700_000)

    # A recent fill for an authoritatively flat scope may own an active RED
    # cooldown, so the configured lookback remains mandatory.
    bot = _incomplete_history_bot(
        policy="always",
        fills=[_episode_fill(800_000, "decrease", 1.0)],
        position_size=0.0,
    )
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, pnl_start_ms)

    # With no held position and no configured cooldown, HSL has no historical
    # fill consumer under the always policy.
    bot = _incomplete_history_bot(
        policy="always", fills=(), position_size=0.0
    )
    bot.hsl["long"]["cooldown_minutes_after_red"] = 0.0
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (False, None)

    # A held scope uses the one canonical fill-proven episode boundary plus
    # the flat-scope cooldown horizon.
    bot = _incomplete_history_bot(
        policy="always",
        fills=[_episode_fill(650_000, "increase", 1.0)],
    )
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, 650_000)

    # Incomplete or ambiguous held reconstruction falls back to the full
    # configured lookback.
    bot = _incomplete_history_bot(
        policy="always",
        fills=[_episode_fill(650_000, "increase", 0.4)],
    )
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, pnl_start_ms)


def test_coin_fill_requirement_uses_effective_hsl_override_policy_and_cooldown():
    pnl_start_ms = 100_000
    now_ms = 1_000_000

    # A coin override can be the only enabled HSL scope.
    bot = _incomplete_history_bot(
        policy="threshold", fills=(), position_size=0.0
    )
    bot.hsl["long"]["enabled"] = False
    _set_coin_hsl_override(
        bot,
        "A",
        enabled=True,
        restart_after_red_policy="always",
        cooldown_minutes_after_red=10.0,
    )
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, 400_000)

    # A strict effective policy keeps the full lookback even if the global
    # side policy is always.
    bot = _incomplete_history_bot(
        policy="always", fills=(), position_size=0.0
    )
    _set_coin_hsl_override(bot, "A", restart_after_red_policy="threshold")
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, pnl_start_ms)

    # Fills for a disabled override do not create a false flat-scope blocker.
    bot = _incomplete_history_bot(
        policy="always",
        fills=[_episode_fill(800_000, "decrease", 1.0)],
        position_size=0.0,
    )
    _set_coin_hsl_override(bot, "A", enabled=False)
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        now_ms, pnl_start_ms=pnl_start_ms
    ) == (True, 700_000)


def test_coin_fill_requirement_treats_unknown_pside_as_recent_ambiguity():
    bot = _incomplete_history_bot(
        policy="always",
        fills=[
            SimpleNamespace(
                position_side="unknown",
                symbol="A",
                timestamp=800_000,
                action="decrease",
                qty=1.0,
                pnl=0.0,
            )
        ],
        position_size=0.0,
    )
    bot.hsl["long"]["enabled"] = False
    bot.hsl["short"]["enabled"] = True
    bot.hsl["short"]["restart_after_red_policy"] = "always"

    assert bot._equity_hard_stop_required_fill_history_start_ms(
        1_000_000, pnl_start_ms=100_000
    ) == (True, 100_000)


def test_coin_required_pnl_events_preserve_each_held_episode_boundary():
    def fill(ts, symbol, action, *, pnl_status="complete"):
        return SimpleNamespace(
            position_side="long",
            symbol=symbol,
            timestamp=ts,
            action=action,
            qty=1.0,
            pnl=0.0,
            pnl_status=pnl_status,
        )

    fills = [
        fill(650_000, "A", "increase"),
        fill(660_000, "B", "increase"),
        fill(700_000, "B", "decrease", pnl_status="pending"),
        fill(800_000, "B", "increase"),
    ]
    bot = _incomplete_history_bot(policy="always", fills=fills)
    bot.hsl["long"]["cooldown_minutes_after_red"] = 1.0
    bot.positions["B"] = {
        "long": {"size": 1.0, "price": 100.0},
        "short": {"size": 0.0},
    }

    assert bot._equity_hard_stop_required_fill_history_start_ms(
        1_000_000, pnl_start_ms=100_000
    ) == (True, 650_000)
    relevant = bot._equity_hard_stop_required_pnl_events(
        fills,
        1_000_000,
        pnl_start_ms=100_000,
    )

    assert [(event.symbol, event.timestamp) for event in relevant] == [
        ("A", 650_000),
        ("B", 800_000),
    ]


@pytest.mark.parametrize("policy", ["threshold", "never"])
def test_coin_non_always_fill_requirement_remains_full_lookback(policy):
    bot = _incomplete_history_bot(
        policy=policy,
        fills=[_episode_fill(650_000, "increase", 1.0)],
    )
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        1_000_000, pnl_start_ms=100_000
    ) == (True, 100_000)


@pytest.mark.parametrize("signal_mode", ["pside", "unified"])
def test_non_coin_hsl_fill_requirement_remains_full_lookback(signal_mode):
    bot = _incomplete_history_bot(
        policy="always",
        fills=[_episode_fill(650_000, "increase", 1.0)],
    )
    bot.config["live"]["hsl_signal_mode"] = signal_mode
    assert bot._equity_hard_stop_required_fill_history_start_ms(
        1_000_000, pnl_start_ms=100_000
    ) == (True, 100_000)


def test_cooldown_anchor_uses_scope_flattening_fill():
    # B2.1: cooldown anchors at the fill that flattened the scope, by any
    # means. A manual close after the last panic fill must win; with no fill
    # evidence the anchor remains unavailable rather than becoming "now".
    bot = make_coin_bot()
    events = [
        SimpleNamespace(
            position_side="long", symbol="A", timestamp=120_500,
            pb_order_type="close_panic_long",
        ),
        SimpleNamespace(
            position_side="long", symbol="A", timestamp=150_000,
            pb_order_type="close_manual_long",
        ),
    ]
    bot._pnls_manager = SimpleNamespace(get_events=lambda: events)
    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
            "long", symbol="A", since_ms=60_000
        )
        == 150_000
    )
    # Window that excludes both fills: anchor unavailable.
    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
            "long", symbol="A", since_ms=200_000
        )
        is None
    )
    # Other-pair fills never leak into the anchor.
    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
            "long", symbol="B", since_ms=60_000
        )
        is None
    )


@pytest.mark.parametrize(
    ("pside", "side"),
    [("long", "sell"), ("short", "buy")],
)
def test_repanic_replay_infers_decrease_from_durable_fill_fields(pside, side):
    bot = make_coin_bot()
    event = FillEvent.from_dict(
        {
            "id": f"{pside}-close",
            "timestamp": 150_000,
            "symbol": "A",
            "side": side,
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
            "pb_order_type": f"close_panic_{pside}",
            "position_side": pside,
            "client_order_id": "",
        }
    )
    bot._pnls_manager = make_fake_pnls_manager([event])

    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
            pside,
            symbol="A",
            since_ms=150_000,
            replay_start_sizes={"A": 1.0},
        )
        == 150_000
    )


@pytest.mark.asyncio
async def test_coin_repanic_replay_includes_intervention_millisecond():
    bot = make_coin_bot(policy="panic")
    symbol = "A"
    state = bot._hsl_coin_state("long", symbol)
    state["halted"] = True
    state["cooldown_until_ms"] = 200_000
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }

    changed = await bot._equity_hard_stop_handle_coin_position_during_cooldown(
        "long", symbol, 150_000
    )

    assert changed is False
    assert state["cooldown_repanic_since_ms"] == 150_000
    assert state["cooldown_repanic_start_sizes"] == {symbol: 1.0}


@pytest.mark.asyncio
async def test_flatten_confirmation_refreshes_fills_and_rejects_pre_episode_anchor():
    bot = make_coin_bot()
    symbol = "A"
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    events = [
        SimpleNamespace(
            position_side="long",
            symbol=symbol,
            timestamp=90_000,
            pb_order_type="close_manual_long",
        )
    ]
    bot._pnls_manager = SimpleNamespace(get_events=lambda: events)
    refresh_sources = []

    async def update_pnls(*, source, since_ms=None):
        refresh_sources.append(source)
        assert since_ms == 120_000
        events.append(
            SimpleNamespace(
                position_side="long",
                symbol=symbol,
                timestamp=170_000,
                pb_order_type="close_grid_long",
            )
        )
        return True

    bot.update_pnls = update_pnls

    stop_ts_ms = await bot._equity_hard_stop_flatten_fill_timestamp_with_refresh(
        "long",
        180_000,
        symbol=symbol,
        since_ms=state["pending_red_since_ms"],
    )

    assert stop_ts_ms == 170_000
    assert refresh_sources == ["hsl_flatten_confirmation"]
    assert state["last_missing_flatten_fill_refresh_ms"] == 0


@pytest.mark.asyncio
async def test_flatten_confirmation_refresh_failure_keeps_scope_protective_without_secret(
    caplog,
):
    bot = make_coin_bot()
    symbol = "A"
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    bot._pnls_manager = make_fake_pnls_manager([])
    secret = "api_key=flatten-refresh-secret"

    async def update_pnls(*, source, since_ms=None):
        assert source == "hsl_flatten_confirmation"
        assert since_ms == 120_000
        raise RuntimeError(secret)

    bot.update_pnls = update_pnls

    with caplog.at_level(logging.WARNING):
        stop_ts_ms = await bot._equity_hard_stop_flatten_fill_timestamp_with_refresh(
            "long",
            180_000,
            symbol=symbol,
            since_ms=state["pending_red_since_ms"],
        )

    assert stop_ts_ms is None
    assert state["pending_stop_event"] is None
    assert state["red_flat_confirmations"] == 0
    assert state["last_missing_flatten_fill_log_ms"] == 180_000
    assert "error_type=RuntimeError" in caplog.text
    assert secret not in caplog.text


def test_red_paused_forced_modes_block_entries_without_panic():
    bot = make_coin_bot()
    bot.positions = {"A": {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.open_orders = {"B": []}
    bot.active_symbols = ["C"]

    bot._equity_hard_stop_set_red_paused_runtime_forced_modes("long")

    forced = bot._runtime_forced_modes["long"]
    assert set(forced) == {"A", "B", "C"}
    assert set(forced.values()) == {"tp_only_with_active_entry_cancellation"}


def bind_hsl_methods(bot):
    for name in (
        "_hsl_psides",
        "_hsl_state",
        "_equity_hard_stop_enabled",
        "_equity_hard_stop_runtime_red_latched",
        "_equity_hard_stop_runtime_tier",
        "_equity_hard_stop_signal_mode",
        "_equity_hard_stop_cooldown_position_policy",
        "_calc_upnl_sum_strict",
        "_equity_hard_stop_apply_coin_sample",
        "_equity_hard_stop_apply_coin_metrics_sample",
        "_equity_hard_stop_maybe_emit_raw_red_pending",
        "_equity_hard_stop_activate_coin_red_from_metrics",
        "_equity_hard_stop_coin_active_pside",
        "_equity_hard_stop_coin_realized_pnl_peak_last",
        "_equity_hard_stop_coin_needs_panic_supervision",
        "_equity_hard_stop_coin_red_active",
        "_equity_hard_stop_coin_symbols",
        "_equity_hard_stop_handle_coin_position_during_cooldown",
        "_equity_hard_stop_refresh_coin_cooldown_after_repanic",
        "_equity_hard_stop_has_open_position_symbol",
        "_equity_hard_stop_count_blocking_open_orders_symbol",
        "_equity_hard_stop_history_coin_value",
        "_equity_hard_stop_initialize_coin_from_history",
        "_equity_hard_stop_start_coin_history_replay",
        "_equity_hard_stop_initialize_from_history",
        "_equity_hard_stop_infer_replay_contract",
        "_equity_hard_stop_apply_sample",
        "_equity_hard_stop_reset_state",
        "_equity_hard_stop_reset_after_restart",
        "_equity_hard_stop_red_episode_finalization",
        "_equity_hard_stop_validate_balance_source_for_history_replay",
        "_equity_hard_stop_balance_override_active",
        "_equity_hard_stop_position_symbols",
        "_equity_hard_stop_latest_flatten_fill_timestamp_optional_ms",
        "_equity_hard_stop_defer_missing_flatten_fill",
        "_equity_hard_stop_flatten_fill_timestamp_with_refresh",
        "_equity_hard_stop_signal_values",
        "_equity_hard_stop_refresh_halted_runtime_forced_modes",
        "_equity_hard_stop_infer_coin_replay_contract",
        "_equity_hard_stop_lookback_ms",
        "_equity_hard_stop_log_transition",
        "_equity_hard_stop_set_red_paused_runtime_forced_modes",
        "_equity_hard_stop_latest_flatten_fill_timestamp_optional_ms",
        "_equity_hard_stop_coverage_allow_incomplete",
        "_equity_hard_stop_required_fill_history_start_ms",
        "_equity_hard_stop_required_pnl_events",
        "_equity_hard_stop_refresh_halted_runtime_forced_modes",
        "_equity_hard_stop_set_red_runtime_forced_modes",
        "_equity_hard_stop_runtime_red_latched",
        "_equity_hard_stop_clear_runtime_forced_modes",
        "_equity_hard_stop_halted_mode",
        "_equity_hard_stop_build_latch_payload",
        "_equity_hard_stop_check_coin",
        "_equity_hard_stop_clear_coin_runtime_forced_mode",
        "_equity_hard_stop_compute_coin_stop_event",
        "_equity_hard_stop_finalize_coin_red_stop",
        "_equity_hard_stop_latest_panic_fill_timestamp_ms",
        "_equity_hard_stop_latest_panic_fill_timestamp_optional_ms",
        "_equity_hard_stop_log_coin_cooldown_status",
        "_equity_hard_stop_emit_coin_status",
        "_equity_hard_stop_make_state",
        "_equity_hard_stop_prime_coin_runtime_for_replay",
        "_equity_hard_stop_reset_coin_after_restart",
        "_equity_hard_stop_set_coin_runtime_forced_mode",
        "_equity_hard_stop_symbol_supported_for_coin_replay",
        "_hsl_coin_state",
    ):
        setattr(bot, name, MethodType(getattr(hsl, name), bot))
    for name in (
        "_assert_no_pending_pnl_events",
        "_fill_history_coverage_status",
        "_pnl_event_preview",
        "_assert_pnl_history_safe_for_risk",
        "_assert_pnl_history_coverage_for_risk",
    ):
        setattr(bot, name, MethodType(getattr(Passivbot, name), bot))


def make_coin_bot(policy="panic"):
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot.user = "test_user"
    bot.exchange = "test_exchange"
    bot._equity_hard_stop = {
        "long": bot._equity_hard_stop_make_state(),
        "short": bot._equity_hard_stop_make_state(),
    }
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._runtime_forced_modes = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.positions = {}
    bot.open_orders = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.c_mults = {}
    bot.config = {
        "live": {
            "hsl_signal_mode": "coin",
            "hsl_position_during_cooldown_policy": policy,
            "pnls_max_lookback_days": 30.0,
        }
    }
    bot.hsl = {
        "long": {
            "enabled": True,
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 0.9,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
        "short": {
            "enabled": False,
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 0.9,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
    }
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._equity_hard_stop_write_latch = lambda pside, payload, symbol=None: "/tmp/hsl_coin.json"
    bot._equity_hard_stop_remove_latch_file = lambda pside, symbol=None: None
    bot.get_raw_balance = lambda: 100.0
    bot.get_exchange_time = lambda: 180_000
    bot.live_value = lambda key: bot.config["live"][key]
    bot._equity_hard_stop_realized_pnl_now = lambda pside=None: 0.0

    def bot_value(pside, key):
        values = {
            "n_positions": 2,
            "total_wallet_exposure_limit": 2.0,
        }
        return values[key]

    bot.bot_value = bot_value

    async def calc_upnl(pside=None, symbol=None):
        return 0.0

    bot._calc_upnl_sum_strict = calc_upnl
    return bot


def test_passivbot_binds_coin_hsl_replay_support_helper():
    assert hasattr(Passivbot, "_equity_hard_stop_symbol_supported_for_coin_replay")


@pytest.mark.asyncio
async def test_coin_hsl_initializer_uses_canonical_required_history_start():
    bot = make_coin_bot()
    captured = {}
    bot._equity_hard_stop_required_fill_history_start_ms = (
        lambda now_ms, pnl_start_ms=None: (True, 120_000)
    )

    async def fake_history(**kwargs):
        captured.update(kwargs)
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert captured["hsl_replay_signal_mode"] == "coin"
    assert captured["hsl_coin_compact_replay"] is True
    assert captured["hsl_replay_start_ms"] == 120_000


def test_coin_hsl_restart_reset_preserves_persistent_no_restart_peak():
    bot = make_coin_bot()
    state = bot._hsl_coin_state("long", "BTC/USDT:USDT")
    state["no_restart_peak_strategy_equity"] = 1.25
    state["pnl_reset_timestamp_ms"] = 123_456
    state["halted"] = True

    bot._equity_hard_stop_reset_coin_after_restart("long", "BTC/USDT:USDT")

    state = bot._hsl_coin_state("long", "BTC/USDT:USDT")
    assert state["no_restart_peak_strategy_equity"] == pytest.approx(1.25)
    assert state["pnl_reset_timestamp_ms"] == 123_456
    assert state["halted"] is False


@pytest.mark.asyncio
async def test_coin_hsl_repanic_fill_confirmation_blocks_past_old_deadline():
    bot = make_coin_bot()
    symbol = "A"
    state = bot._hsl_coin_state("long", symbol)
    state["halted"] = True
    state["cooldown_until_ms"] = 200_000
    state["cooldown_intervention_active"] = True
    state["cooldown_repanic_reset_pending"] = True
    state["cooldown_repanic_since_ms"] = 150_001
    state["cooldown_repanic_start_sizes"] = {symbol: 1.0}
    bot.positions = {
        symbol: {
            "long": {"size": 0.0, "price": 0.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = make_fake_pnls_manager([])
    bot.get_exchange_time = lambda: 210_000
    refreshes = []

    async def update_pnls(*, source, since_ms=None):
        refreshes.append((source, since_ms))
        return False

    bot.update_pnls = update_pnls

    await bot._equity_hard_stop_check_coin()

    assert refreshes == [("hsl_flatten_confirmation", 150_001)]
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 200_000
    assert state["cooldown_repanic_reset_pending"] is True
    assert state["cooldown_repanic_start_sizes"] == {symbol: 1.0}


@pytest.mark.asyncio
async def test_coin_hsl_replay_cancels_when_shutdown_requested_after_history_load():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    bot = make_coin_bot()
    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fake_history(current_balance=None, **kwargs):
        bot.stop_signal_received = True
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    async def fail_calc_upnl(*_args, **_kwargs):
        raise AssertionError("shutdown should cancel before live upnl fetch")

    bot.get_balance_equity_history = fake_history
    bot._calc_upnl_sum_strict = fail_calc_upnl

    with pytest.raises(asyncio.CancelledError, match="hsl_coin_history_replay_history_loaded"):
        await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot.stop_signal_received is True
    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is False
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type.startswith("hsl.replay.")]
    assert [event.event_type for event in events] == [
        EventTypes.HSL_REPLAY_STARTED,
        EventTypes.HSL_REPLAY_FAILED,
    ]
    assert events[0].status == "started"
    assert events[0].reason_code == "coin_history_replay"
    assert events[1].status == "failed"
    assert events[1].reason_code == "shutdown_cancelled"
    assert events[1].data["elapsed_s"] is not None
    assert events[1].data["history_fetch_elapsed_s"] is not None
    assert events[1].data["pre_replay_elapsed_s"] is None
    assert events[1].data["replay_loop_elapsed_s"] is None
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_replay_failure_redacts_state_and_event_but_preserves_exception():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    secret = "api_secret=hsl-coin-replay-before-ready-secret"
    unsafe_type = type("ApiKeyHslCoinReplayBeforeReadySecret", (RuntimeError,), {})
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fail_history(*_args, **_kwargs):
        raise unsafe_type(secret)

    bot.get_balance_equity_history = fail_history

    with pytest.raises(unsafe_type) as raised:
        await bot._equity_hard_stop_start_coin_history_replay()

    assert str(raised.value) == secret
    assert bot._equity_hard_stop_coin_replay_failure == "RuntimeError"
    assert bot._equity_hard_stop_coin_protective_ready is False
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    failed_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_FAILED
    ]
    assert len(failed_events) == 1
    assert failed_events[0].data["error_type"] == "RuntimeError"
    assert secret not in str(failed_events[0].data)
    assert unsafe_type.__name__ not in str(failed_events[0].data)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_emits_lifecycle_events():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_hsl_replay"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fake_history(current_balance=None, **kwargs):
        await asyncio.sleep(0.01)
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 0.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": -1.0, "short": 0.0}
                    },
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 1.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 1.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": -0.5, "short": 0.0}
                    },
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type.startswith("hsl.replay.")]
    assert [event.event_type for event in events] == [
        EventTypes.HSL_REPLAY_STARTED,
        EventTypes.HSL_REPLAY_PROGRESS,
        EventTypes.HSL_REPLAY_PROGRESS,
        EventTypes.HSL_REPLAY_PROGRESS,
        EventTypes.HSL_REPLAY_PROGRESS,
        EventTypes.HSL_REPLAY_COMPLETED,
    ]
    assert {event.cycle_id for event in events} == {"cy_hsl_replay"}
    assert events[0].status == "started"
    assert events[0].reason_code == "coin_history_replay"
    assert events[1].reason_code == "history_loaded"
    assert events[1].data["symbols"] == 1
    assert events[1].data["pairs"] == 1
    assert events[1].data["held_pairs"] == 1
    assert events[1].data["cooldown_pairs"] == 0
    assert events[1].data["required_pairs"] == 1
    assert events[1].data["timeline_rows"] == 2
    assert events[1].data["history_fetch_elapsed_s"] is not None
    assert events[1].data["pre_replay_elapsed_s"] is not None
    assert events[1].data["elapsed_s"] is not None
    assert events[2].status == "started"
    assert events[2].reason_code == "pair_replay_progress"
    assert events[2].data["pair_idx"] == 1
    assert events[2].data["is_held_pair"] is True
    assert events[2].data["applied_rows"] == 0
    assert events[2].data["scanned_rows"] == 0
    assert events[2].data["total_applied_rows"] == 0
    assert events[2].data["total_scanned_rows"] == 0
    assert events[2].data["rows_per_second"] is not None
    assert events[2].data["scanned_rows_per_second"] is not None
    assert events[2].data["pair_elapsed_s"] is not None
    assert events[3].reason_code == "pair_replay_progress"
    assert events[3].data["applied_rows"] == 2
    assert events[3].data["scanned_rows"] == 2
    assert events[3].data["total_applied_rows"] == 2
    assert events[3].data["total_scanned_rows"] == 2
    assert events[3].data["rows_per_second"] is not None
    assert events[3].data["scanned_rows_per_second"] is not None
    assert events[3].data["pair_elapsed_s"] is not None
    assert events[4].status == "succeeded"
    assert events[4].reason_code == ReasonCodes.HSL_HELD_PROTECTIVE_READY
    assert events[4].data["stage"] == "held_protective_ready"
    assert events[4].data["ready_pairs"] == 1
    assert events[4].data["pending_pairs"] == 0
    assert events[4].data["protective_elapsed_s"] is not None
    completed = events[5]
    assert completed.status == "succeeded"
    assert completed.reason_code == "coin_history_replay_completed"
    assert completed.data["rows"] == 2
    assert completed.data["applied_rows"] == 2
    assert completed.data["total_applied_rows"] == 2
    assert completed.data["total_scanned_rows"] == 2
    assert completed.data["total_scanned_rows"] == completed.data["candidate_rows"]
    assert completed.data["pairs"] == 1
    assert completed.data["held_pairs"] == 1
    assert completed.data["cooldown_pairs"] == 0
    assert completed.data["required_pairs"] == 1
    assert completed.data["skipped_pairs"] == 0
    assert completed.data["timeline_rows"] == 2
    assert completed.data["fill_events"] == 0
    assert completed.data["panic_events"] == 0
    assert completed.data["rows_per_second"] is not None
    assert completed.data["scanned_rows_per_second"] is not None
    assert completed.data["history_fetch_elapsed_s"] is not None
    assert completed.data["pre_replay_elapsed_s"] is not None
    assert completed.data["replay_loop_elapsed_s"] is not None
    assert completed.data["full_elapsed_s"] is not None
    assert completed.data["protective_elapsed_s"] is not None
    assert completed.data["startup_blocking_elapsed_s"] is not None
    assert completed.data["elapsed_s"] is not None
    assert completed.data["history_fetch_elapsed_s"] > 0.0
    phase_elapsed_s = (
        completed.data["history_fetch_elapsed_s"]
        + completed.data["pre_replay_elapsed_s"]
        + completed.data["replay_loop_elapsed_s"]
    )
    assert phase_elapsed_s <= completed.data["startup_blocking_elapsed_s"] + 0.006
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_replay_forced_pair_events_do_not_bypass_console_cadence(
    monkeypatch, caplog
):
    from live.event_bus import EventTypes

    bot = make_coin_bot()
    symbols = ("A", "B", "C", "D")
    emitted_events = []
    clock = {"now_s": 0.0}

    def emit_live_event(event_type, **kwargs):
        emitted_events.append(SimpleNamespace(event_type=event_type, **kwargs))
        return object()

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 0.0, "short": 0.0} for symbol in symbols
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": 0.0, "short": 0.0} for symbol in symbols
                    },
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    async def advance_clock_for_pair(pside=None, symbol=None):
        clock["now_s"] = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}[symbol]
        return 0.0

    bot._live_event_pipeline = object()
    bot._emit_live_event = emit_live_event
    bot.get_balance_equity_history = fake_history
    bot._calc_upnl_sum_strict = advance_clock_for_pair
    monkeypatch.setattr(hsl, "time", SimpleNamespace(monotonic=lambda: clock["now_s"]))

    with caplog.at_level(logging.INFO):
        await bot._equity_hard_stop_initialize_coin_from_history()

    pair_events = [
        event
        for event in emitted_events
        if event.event_type == EventTypes.HSL_REPLAY_PROGRESS
        and event.reason_code == "pair_replay_progress"
    ]
    assert [(event.symbol, event.data["pair_idx"]) for event in pair_events] == [
        ("A", 1),
        ("A", 1),
        ("B", 2),
        ("C", 3),
        ("D", 4),
    ]
    progress_logs = [
        record.getMessage()
        for record in caplog.records
        if "HSL coin history reconstruction progress" in record.getMessage()
    ]
    assert len(progress_logs) == 2
    assert "pair=1/4 pside=long symbol=A" in progress_logs[0]
    assert "pair=3/4 pside=long symbol=C" in progress_logs[1]
    assert any(
        "HSL coin history reconstruction completed" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_reports_scanned_optional_rows_without_apply():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot._hsl_coin_state("long", symbol)
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history
    now_ms = bot.get_exchange_time()

    def active_cooldown_contract(self, pside, replay_symbol, fill_events, replay_now_ms):
        del self, pside, replay_symbol, fill_events, replay_now_ms
        return {
            "policy": "normal",
            "latest_panic_ts": 60_000,
            "cooldown_until_ms": now_ms + 60_000,
            "intervention_entry_ts": None,
            "active_cooldown_now": True,
            "intervention_active": False,
            "unresolved_residue": False,
        }

    bot._equity_hard_stop_infer_coin_replay_contract = MethodType(
        active_cooldown_contract, bot
    )

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    pair_event = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_REPLAY_PROGRESS
        and event.reason_code == "pair_replay_progress"
    ][-1]
    assert pair_event.data["applied_rows"] == 0
    assert pair_event.data["scanned_rows"] == 2
    assert pair_event.data["total_applied_rows"] == 0
    assert pair_event.data["total_scanned_rows"] == 2
    assert pair_event.data["rows_per_second"] is not None
    assert pair_event.data["scanned_rows_per_second"] is not None
    assert pair_event.data["pair_elapsed_s"] is not None
    assert bot._hsl_coin_state("long", symbol)["halted"] is True

    completed = next(
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_REPLAY_COMPLETED
    )
    assert completed.data["applied_rows"] == 0
    assert completed.data["total_applied_rows"] == 0
    assert completed.data["total_scanned_rows"] == 2
    assert completed.data["total_scanned_rows"] == completed.data["candidate_rows"]
    assert completed.data["rows_per_second"] is not None
    assert completed.data["scanned_rows_per_second"] is not None
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_processes_held_late_symbol_first():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    bot.positions = {
        "Z": {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0},
                        "Z": {"long": 0.0, "short": 0.0},
                    },
                    "unrealized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0},
                        "Z": {"long": -1.0, "short": 0.0},
                    },
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    pair_events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_REPLAY_PROGRESS
        and event.reason_code == "pair_replay_progress"
    ]
    assert len(pair_events) == 3
    assert pair_events[0].symbol == "Z"
    assert pair_events[0].pside == "long"
    assert pair_events[0].data["pair_idx"] == 1
    assert pair_events[0].data["pairs"] == 2
    assert pair_events[0].data["held_pairs"] == 1
    assert pair_events[0].data["is_held_pair"] is True
    assert pair_events[1].symbol == "Z"
    assert pair_events[1].data["scanned_rows"] == 1
    assert pair_events[2].symbol == "A"
    assert set(bot._equity_hard_stop_coin["long"]) == {"A", "Z"}
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_start_releases_after_held_pair_and_owns_one_continuation():
    bot = make_coin_bot()
    bot.positions = {
        "Z": {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    flat_pair_started = asyncio.Event()
    release_flat_pair = asyncio.Event()

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0},
                        "Z": {"long": 0.0, "short": 0.0},
                    },
                    "unrealized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0},
                        "Z": {"long": -1.0, "short": 0.0},
                    },
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    async def gated_upnl(pside=None, symbol=None):
        if symbol == "A":
            flat_pair_started.set()
            await release_flat_pair.wait()
        return 0.0

    bot.get_balance_equity_history = fake_history
    bot._calc_upnl_sum_strict = gated_upnl

    await bot._equity_hard_stop_start_coin_history_replay()

    assert bot._equity_hard_stop_coin_protective_ready is True
    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is False
    assert bot._equity_hard_stop_coin_replay_ready_pairs == {("long", "Z")}
    assert bot._equity_hard_stop_coin_replay_pending_pairs == {("long", "A")}
    assert bot._equity_hard_stop_coin_replay_task.done() is False
    assert bot.maintainers["hsl_coin_replay"] is bot._equity_hard_stop_coin_replay_task
    await asyncio.wait_for(flat_pair_started.wait(), timeout=1.0)

    same_task = bot._equity_hard_stop_coin_replay_task
    await bot._equity_hard_stop_start_coin_history_replay()
    assert bot._equity_hard_stop_coin_replay_task is same_task

    release_flat_pair.set()
    await asyncio.wait_for(same_task, timeout=1.0)
    assert bot._equity_hard_stop_coin_initialized is True
    assert bot._equity_hard_stop_coin_replay_pending_pairs == set()
    assert bot._equity_hard_stop_coin_replay_ready_pairs == {
        ("long", "A"),
        ("long", "Z"),
    }


@pytest.mark.asyncio
async def test_coin_hsl_background_replay_yields_before_one_thousand_rows():
    bot = make_coin_bot()
    bot.positions = {}
    bot.get_exchange_time = lambda: 10_000_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": minute * 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0}
                    },
                }
                for minute in range(1, 151)
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_start_coin_history_replay()

    replay_task = bot._equity_hard_stop_coin_replay_task
    assert bot._equity_hard_stop_coin_protective_ready is True
    assert replay_task.done() is False
    assert bot._equity_hard_stop_coin_replay_pending_pairs == {("long", "A")}

    await asyncio.wait_for(replay_task, timeout=1.0)
    assert bot._equity_hard_stop_coin_initialized is True
    assert bot._equity_hard_stop_coin_replay_pending_pairs == set()


@pytest.mark.asyncio
async def test_coin_hsl_partial_replay_restarts_if_pending_pair_becomes_held():
    bot = make_coin_bot()
    bot._equity_hard_stop_coin_protective_ready = True
    bot._equity_hard_stop_coin_initialized = False
    bot._equity_hard_stop_coin_replay_ready_pairs = set()
    bot._equity_hard_stop_coin_replay_pending_pairs = {("long", "A")}
    bot.positions = {
        "A": {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }

    with pytest.raises(
        RestartBotException,
        match="restart required for held-first reconstruction: long:A",
    ):
        await bot._equity_hard_stop_check_coin()


@pytest.mark.asyncio
async def test_coin_hsl_background_failure_keeps_pending_pair_blocked_without_retry():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    bot.positions = {
        "Z": {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0},
                        "Z": {"long": 0.0, "short": 0.0},
                    },
                    "unrealized_pnl_by_coin_pside": {
                        "A": {"long": 0.0, "short": 0.0},
                        "Z": {"long": -1.0, "short": 0.0},
                    },
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    secret = "api_secret=hsl-coin-replay-failure-secret"
    unsafe_type = type("ApiKeyHslCoinReplayFailureSecret", (RuntimeError,), {})
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def failing_upnl(pside=None, symbol=None):
        if symbol == "A":
            raise unsafe_type(secret)
        return 0.0

    bot.get_balance_equity_history = fake_history
    bot._calc_upnl_sum_strict = failing_upnl

    await bot._equity_hard_stop_start_coin_history_replay()
    replay_task = bot._equity_hard_stop_coin_replay_task
    await asyncio.wait_for(replay_task, timeout=1.0)

    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is False
    assert bot._equity_hard_stop_coin_replay_ready_pairs == {("long", "Z")}
    assert bot._equity_hard_stop_coin_replay_pending_pairs == {("long", "A")}
    assert bot._equity_hard_stop_coin_replay_failure == "RuntimeError"
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    failed_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_FAILED
    ]
    assert len(failed_events) == 1
    assert failed_events[0].data["error_type"] == "RuntimeError"
    assert secret not in str(failed_events[0].data)
    assert unsafe_type.__name__ not in str(failed_events[0].data)
    await bot._equity_hard_stop_start_coin_history_replay()
    assert bot._equity_hard_stop_coin_replay_task is replay_task

    metrics = await bot._equity_hard_stop_check_coin()
    assert "long:Z" in metrics
    assert "long:A" not in metrics
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_data_maintainers_own_active_coin_hsl_replay_task():
    bot = Passivbot.__new__(Passivbot)
    bot.ws_enabled = False
    blocker = asyncio.Event()

    async def maintain_hourly_cycle():
        await blocker.wait()

    async def continue_replay():
        await blocker.wait()

    bot.maintain_hourly_cycle = maintain_hourly_cycle
    replay_task = asyncio.create_task(continue_replay())
    bot._equity_hard_stop_coin_replay_task = replay_task

    await bot.start_data_maintainers()

    assert bot.maintainers["hsl_coin_replay"] is replay_task
    hourly_task = bot.maintainers["maintain_hourly_cycle"]
    bot.stop_data_maintainers(verbose=False)
    await asyncio.gather(replay_task, hourly_task, return_exceptions=True)
    assert replay_task.cancelled()
    assert hourly_task.cancelled()


async def _run_parity_history(
    monkeypatch,
    *,
    compact: bool = False,
    fill_events_override: list[dict] | None = None,
    hsl_replay_start_ms: int | None = None,
    candle_calls: list[dict] | None = None,
):
    """Run the real coin-mode collection over a small two-close scenario."""
    import numpy as np
    from unittest.mock import AsyncMock

    import passivbot as passivbot_module
    from live.event_bus import ListEventSink, LiveEventPipeline

    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 120_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_parity"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            if candle_calls is not None:
                candle_calls.append({"symbol": sym, **kwargs})
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 84.0, 100.0, 60.0, 60.0, 1.0),
                    (base_ts + 120_000, 60.0, 102.0, 60.0, 101.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 90_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 101.0,
            "pnl": 0.5,
        },
    ]
    if fill_events_override is not None:
        fill_events = fill_events_override
    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
        hsl_coin_compact_replay=compact,
        hsl_replay_start_ms=hsl_replay_start_ms,
    )
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    return history, symbol


@pytest.mark.asyncio
async def test_hsl_compact_coin_history_matches_authoritative_values(monkeypatch):
    history, symbol = await _run_parity_history(monkeypatch)
    compact_history, compact_symbol = await _run_parity_history(
        monkeypatch, compact=True
    )
    assert compact_symbol == symbol
    assert "timeline" not in compact_history
    assert "balances" not in compact_history
    assert "equities" not in compact_history
    assert history["metadata"]["history_format"] == "timeline"
    assert compact_history["metadata"]["history_format"] == "compact"
    compact = compact_history["hsl_coin_compact_replay"]
    pair = compact["pair_values"][("long", symbol)]
    assert compact["timestamps"].tolist() == [
        int(row["timestamp"]) for row in history["timeline"]
    ]
    assert compact["balances"].tolist() == pytest.approx(
        [float(row["balance"]) for row in history["timeline"]], abs=1e-9
    )
    assert compact["realized_pnl"].tolist() == pytest.approx(
        [float(row["realized_pnl"]) for row in history["timeline"]], abs=1e-9
    )
    for idx, row in enumerate(history["timeline"]):
        realized = row["realized_pnl_by_coin_pside"].get(symbol, {}).get("long")
        unrealized = row["unrealized_pnl_by_coin_pside"].get(symbol, {}).get("long")
        if realized is None:
            assert math.isnan(pair["realized_pnl"][idx])
        else:
            assert pair["realized_pnl"][idx] == pytest.approx(realized, abs=1e-9)
        if unrealized is None:
            assert math.isnan(pair["unrealized_pnl"][idx])
        else:
            assert pair["unrealized_pnl"][idx] == pytest.approx(unrealized, abs=1e-9)


@pytest.mark.asyncio
async def test_hsl_compact_coin_history_zero_fill_shape(monkeypatch):
    history, _symbol = await _run_parity_history(
        monkeypatch,
        compact=True,
        fill_events_override=[],
    )

    assert set(history) == {
        "hsl_coin_compact_replay",
        "panic_flatten_events",
        "fill_events",
        "metadata",
    }
    compact = history["hsl_coin_compact_replay"]
    assert compact["timestamps"].shape == (1,)
    assert compact["balances"].tolist() == [100.0]
    assert compact["realized_pnl"].tolist() == [0.0]
    assert compact["pair_values"] == {}
    assert history["metadata"]["history_format"] == "compact"


@pytest.mark.asyncio
async def test_hsl_compact_coin_history_bounds_materialization_but_seeds_prior_fills(
    monkeypatch,
):
    base_ts = 1_800_000_000_000
    symbol = "BTC/USDT:USDT"
    candle_calls = []
    fill_events = [
        {
            "timestamp": base_ts - 120_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts - 60_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 1.0,
            "price": 95.0,
            "pnl": -5.0,
            "pb_order_type": "close_panic_long",
        },
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 90_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 101.0,
            "pnl": 0.5,
        },
    ]

    history, _symbol = await _run_parity_history(
        monkeypatch,
        compact=True,
        fill_events_override=fill_events,
        hsl_replay_start_ms=base_ts,
        candle_calls=candle_calls,
    )

    compact = history["hsl_coin_compact_replay"]
    assert compact["timestamps"].tolist() == [
        base_ts,
        base_ts + 60_000,
        base_ts + 120_000,
    ]
    assert compact["balances"].tolist() == pytest.approx([99.5, 100.0, 100.0])
    assert compact["realized_pnl"].tolist() == pytest.approx([0.0, 0.5, 0.5])
    pair = compact["pair_values"][("long", symbol)]
    assert pair["realized_pnl"].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert [event["timestamp"] for event in history["fill_events"]] == [
        base_ts,
        base_ts + 90_000,
    ]
    assert history["panic_flatten_events"] == []
    assert history["metadata"]["events_used"] == 4
    assert history["metadata"]["replay_events"] == 2
    assert history["metadata"]["pre_window_events_applied"] == 2
    assert history["metadata"]["materialized_start_ms"] == base_ts
    assert history["metadata"]["bounded_history"] is True
    assert candle_calls
    assert {call["start_ts"] for call in candle_calls} == {base_ts}


@pytest.mark.asyncio
async def test_hsl_replay_start_requires_compact_coin_mode_and_present_time(monkeypatch):
    base_ts = 1_800_000_000_000

    with pytest.raises(ValueError, match="requires hsl_coin_compact_replay"):
        await _run_parity_history(
            monkeypatch,
            hsl_replay_start_ms=base_ts,
        )

    with pytest.raises(ValueError, match="cannot be later than exchange time"):
        await _run_parity_history(
            monkeypatch,
            compact=True,
            hsl_replay_start_ms=base_ts + 180_000,
        )


def test_calc_upnl_sum_strict_preserves_symbol_filter():
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot.fetched_positions = [
        {"symbol": "A", "position_side": "long", "price": 100.0, "size": 1.0},
        {"symbol": "B", "position_side": "long", "price": 100.0, "size": 2.0},
    ]
    bot.c_mults = {"A": 1.0, "B": 1.0}

    async def get_live_last_prices(symbols, max_age_ms, context):
        return {"A": 90.0, "B": 80.0}

    bot._get_live_last_prices = get_live_last_prices

    assert asyncio.run(bot._calc_upnl_sum_strict("long")) == pytest.approx(-50.0)
    assert asyncio.run(bot._calc_upnl_sum_strict("long", "A")) == pytest.approx(-10.0)


def test_coin_hsl_slot_budget_rejects_zero_n_positions():
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.config = {"live": {"pnls_max_lookback_days": 30.0}}
    bot.hsl = {
        "long": {
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
        }
    }

    def bot_value(pside, key):
        values = {
            "n_positions": 0,
            "total_wallet_exposure_limit": 1.0,
        }
        return values[key]

    bot.bot_value = bot_value

    with pytest.raises(ValueError, match="n_positions"):
        bot._equity_hard_stop_apply_coin_sample("long", "A", 60_000, 100.0, -1.0)


def test_hsl_transition_falls_back_to_monitor_when_pipeline_absent():
    bot = make_coin_bot()
    captured = []
    bot._live_event_pipeline = None
    bot._live_event_current_cycle_id = "cy_absent_pipeline"
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    def record_event(kind, tags, payload, *, pside=None, symbol=None, ts=None):
        captured.append(
            {
                "kind": kind,
                "tags": tuple(tags),
                "payload": payload,
                "pside": pside,
                "symbol": symbol,
                "ts": ts,
            }
        )

    bot._monitor_record_event = record_event
    metrics = {
        "pside": "long",
        "signal_mode": "pside",
        "timestamp_ms": 180_000,
        "balance": 100.0,
        "strategy_equity": 98.0,
        "peak_strategy_equity": 100.0,
        "rolling_peak_strategy_equity": 100.0,
        "drawdown_raw": 0.02,
        "drawdown_ema": 0.01,
        "drawdown_score": 0.01,
        "strategy_pnl": -2.0,
        "peak_strategy_pnl": 0.0,
        "red_threshold": 0.5,
        "tier": "yellow",
        "changed": True,
    }

    bot._equity_hard_stop_log_transition("long", metrics, "green")

    assert len(captured) == 1
    event = captured[0]
    assert event["kind"] == "hsl.transition"
    assert event["tags"] == ("hsl", "risk", "transition")
    assert event["pside"] == "long"
    assert event["ts"] == 180_000
    assert event["payload"]["previous_tier"] == "green"
    assert event["payload"]["tier"] == "yellow"
    assert event["payload"]["metrics"]["tier"] == "yellow"
    assert event["payload"]["metrics"]["changed"] is True


@pytest.mark.asyncio
async def test_coin_hsl_check_skips_enabled_side_with_zero_budget():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["short"]["enabled"] = True
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_hsl"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    def bot_value(pside, key):
        values = {
            "long": {"n_positions": 2, "total_wallet_exposure_limit": 2.0},
            "short": {"n_positions": 3, "total_wallet_exposure_limit": 0.0},
        }
        return values[pside][key]

    bot.bot_value = bot_value

    out = await bot._equity_hard_stop_check_coin()

    assert set(out) == {f"long:{symbol}"}
    assert symbol in bot._equity_hard_stop_coin["long"]
    assert bot._equity_hard_stop_coin["short"] == {}
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_STATUS]
    assert len(events) == 1
    assert events[0].cycle_id == "cy_coin_hsl"
    assert events[0].symbol == symbol
    assert events[0].pside == "long"
    assert events[0].data["signal_mode"] == "coin"
    assert events[0].data["dist_to_red"] == pytest.approx(0.5)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_check_emits_raw_red_pending_event_with_bounded_payload():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_raw_red_pending"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    bot._equity_hard_stop_status_log_interval_ms = 15 * 60 * 1000
    bot.get_exchange_time = lambda: 1_000_000

    def pending_metrics(_pside, _symbol, timestamp_ms, _balance, _current_upnl):
        return {
            "pside": "long",
            "symbol": symbol,
            "signal_mode": "coin",
            "timestamp_ms": int(timestamp_ms),
            "drawdown_raw": 0.20,
            "drawdown_ema": 0.05,
            "drawdown_score": 0.05,
            "red_threshold": 0.10,
            "tier": "orange",
            "changed": False,
            "elapsed_minutes": 1,
            "slot_budget": 100.0,
            "realized_pnl": 0.0,
            "peak_realized_pnl": 20.0,
            "unrealized_pnl": 0.0,
        }

    bot._equity_hard_stop_apply_coin_sample = pending_metrics

    await bot._equity_hard_stop_check_coin()
    await bot._equity_hard_stop_check_coin()

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_RAW_RED_PENDING
    ]
    assert len(events) == 1
    event = events[0]
    assert event.level == "warning"
    assert event.status == "degraded"
    assert event.reason_code == ReasonCodes.HSL_RAW_RED_PENDING_EMA_CONFIRMATION
    assert event.cycle_id == "cy_coin_raw_red_pending"
    assert event.symbol == symbol
    assert event.pside == "long"
    assert event.data["signal_mode"] == "coin"
    assert event.data["drawdown_raw"] == pytest.approx(0.20)
    assert event.data["drawdown_ema"] == pytest.approx(0.05)
    assert event.data["dist_to_red"] == pytest.approx(0.05)
    assert event.data["raw_excess"] == pytest.approx(0.10)
    assert event.data["balance_override_active"] is False
    assert "balance" not in event.data
    assert "slot_budget" not in event.data
    assert "realized_pnl" not in event.data
    assert "peak_realized_pnl" not in event.data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_coin_hsl_runtime_forced_mode_changes_emit_risk_events():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_risk_mode"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._current_live_event_cycle_id = MethodType(Passivbot._current_live_event_cycle_id, bot)
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    bot._emit_risk_mode_changed_event = MethodType(
        Passivbot._emit_risk_mode_changed_event,
        bot,
    )

    bot._equity_hard_stop_set_coin_runtime_forced_mode("long", symbol, "panic")
    bot._equity_hard_stop_set_coin_runtime_forced_mode("long", symbol, "panic")
    bot._equity_hard_stop_clear_coin_runtime_forced_mode("long", symbol)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.RISK_MODE_CHANGED]
    assert len(events) == 2
    assert events[0].cycle_id == "cy_risk_mode"
    assert events[0].pside == "long"
    assert events[0].symbol == symbol
    assert events[0].reason_code == "hsl_runtime_forced_mode_set"
    assert events[0].data["action"] == "set"
    assert events[0].data["mode"] == "panic"
    assert "previous_mode" not in events[0].data
    assert events[1].reason_code == "hsl_runtime_forced_mode_clear"
    assert events[1].data["action"] == "clear"
    assert events[1].data["previous_mode"] == "panic"
    assert "mode" not in events[1].data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_skips_enabled_side_with_zero_budget():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["short"]["enabled"] = True
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    def bot_value(pside, key):
        values = {
            "long": {"n_positions": 2, "total_wallet_exposure_limit": 2.0},
            "short": {"n_positions": 3, "total_wallet_exposure_limit": 0.0},
        }
        return values[pside][key]

    bot.bot_value = bot_value

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert symbol in bot._equity_hard_stop_coin["long"]
    assert bot._equity_hard_stop_coin["short"] == {}


@pytest.mark.parametrize(
    "n_positions,total_wallet_exposure_limit,match",
    [
        (-1, 1.0, "n_positions"),
        (float("nan"), 1.0, "n_positions"),
        (0.4, 1.0, "round to > 0"),
        (1, -1.0, "total_wallet_exposure_limit"),
        (1, float("inf"), "total_wallet_exposure_limit"),
    ],
)
def test_coin_hsl_active_side_rejects_invalid_budget_config(
    n_positions, total_wallet_exposure_limit, match
):
    bot = make_coin_bot()

    def bot_value(pside, key):
        values = {
            "n_positions": n_positions,
            "total_wallet_exposure_limit": total_wallet_exposure_limit,
        }
        return values[key]

    bot.bot_value = bot_value

    with pytest.raises(ValueError, match=match):
        bot._equity_hard_stop_coin_active_pside("long")


@pytest.mark.parametrize("total_wallet_exposure_limit", [0.5, 1.0, 5.0])
def test_coin_hsl_live_slot_budget_ignores_twel(total_wallet_exposure_limit):
    bot = make_coin_bot()
    symbol = "A"

    def bot_value(pside, key):
        values = {
            "n_positions": 2,
            "total_wallet_exposure_limit": total_wallet_exposure_limit,
        }
        return values[key]

    bot.bot_value = bot_value

    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        0,
        100.0,
        0.0,
        0.0,
        0.0,
    )
    metrics = bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        60_000,
        100.0,
        0.0,
        -25.0,
        0.0,
    )

    assert metrics["slot_budget"] == pytest.approx(50.0)
    assert metrics["drawdown_usd"] == pytest.approx(25.0)
    assert metrics["drawdown_raw"] == pytest.approx(0.5)


def _red_episode_history(symbol, *, flatten_pnl, rows_upnl, flatten_ts=150_000):
    """History where a coin episode enters at 60_000, draws down, and is
    flattened by an ORDINARY close fill (no panic_flatten_events)."""
    timeline = []
    realized = 0.0
    for idx, upnl in enumerate(rows_upnl):
        ts = 60_000 * (idx + 1)
        if flatten_ts < ts + 60_000:
            realized = flatten_pnl
        timeline.append(
            {
                "timestamp": ts,
                "balance": 100.0,
                "realized_pnl": realized,
                "realized_pnl_by_coin_pside": {symbol: {"long": realized, "short": 0.0}},
                "unrealized_pnl_by_coin_pside": {symbol: {"long": upnl, "short": 0.0}},
            }
        )
    fills = [
        {
            "timestamp": 60_000,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "pnl": 0.0,
        },
        {
            "timestamp": flatten_ts,
            "symbol": symbol,
            "pside": "long",
            "action": "decrease",
            "qty": 1.0,
            "pnl": flatten_pnl,
        },
    ]
    return {"timeline": timeline, "panic_flatten_events": [], "fill_events": fills}


def _coin_history_as_compact(history, symbol):
    import numpy as np

    rows = history["timeline"]
    return {
        "hsl_coin_compact_replay": {
            "timestamps": np.asarray(
                [row["timestamp"] for row in rows], dtype=np.int64
            ),
            "balances": np.asarray(
                [row["balance"] for row in rows], dtype=np.float64
            ),
            "realized_pnl": np.asarray(
                [row["realized_pnl"] for row in rows], dtype=np.float64
            ),
            "pair_values": {
                ("long", symbol): {
                    "realized_pnl": np.asarray(
                        [
                            row["realized_pnl_by_coin_pside"][symbol]["long"]
                            for row in rows
                        ],
                        dtype=np.float64,
                    ),
                    "unrealized_pnl": np.asarray(
                        [
                            row["unrealized_pnl_by_coin_pside"][symbol]["long"]
                            for row in rows
                        ],
                        dtype=np.float64,
                    ),
                }
            },
        },
        "panic_flatten_events": history["panic_flatten_events"],
        "fill_events": history["fill_events"],
    }


def _flat_position_bot(symbol):
    bot = make_coin_bot()
    bot.positions = {
        symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0, "price": 0.0}}
    }
    return bot


@pytest.mark.asyncio
async def test_compact_sparse_replay_matches_dense_state_across_flat_gaps():
    symbol = "A"
    rows = []
    for minute in range(1, 21):
        balance = 100.0 if minute < 10 else 90.0
        realized = 0.0 if minute < 3 else -1.0
        upnl = -1.0 if minute == 2 else 0.0
        rows.append(
            {
                "timestamp": minute * 60_000,
                "balance": balance,
                "realized_pnl": realized,
                "realized_pnl_by_coin_pside": {
                    symbol: {"long": realized, "short": 0.0}
                },
                "unrealized_pnl_by_coin_pside": {
                    symbol: {"long": upnl, "short": 0.0}
                },
            }
        )
    history = {
        "timeline": rows,
        "panic_flatten_events": [],
        "fill_events": [
            {
                "timestamp": 120_001,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
            },
            {
                "timestamp": 180_001,
                "symbol": symbol,
                "pside": "long",
                "action": "decrease",
                "qty": 1.0,
                "pnl": -1.0,
            },
        ],
    }

    async def run(payload):
        bot = _flat_position_bot(symbol)
        bot.hsl["long"]["ema_span_minutes"] = 5.0
        bot.get_exchange_time = lambda: rows[-1]["timestamp"]

        async def fake_history(current_balance=None, **kwargs):
            return payload

        bot.get_balance_equity_history = fake_history
        calls = 0
        original_apply = bot._equity_hard_stop_apply_coin_metrics_sample

        def count_apply(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_apply(*args, **kwargs)

        bot._equity_hard_stop_apply_coin_metrics_sample = count_apply
        await bot._equity_hard_stop_initialize_coin_from_history()
        state = bot._hsl_coin_state("long", symbol)
        return state, calls

    dense_state, dense_calls = await run(history)
    sparse_state, sparse_calls = await run(_coin_history_as_compact(history, symbol))

    assert sparse_calls < dense_calls
    assert sparse_state["halted"] == dense_state["halted"]
    assert sparse_state["no_restart_latched"] == dense_state["no_restart_latched"]
    assert sparse_state["cooldown_until_ms"] == dense_state["cooldown_until_ms"]
    assert sparse_state["pnl_reset_timestamp_ms"] == dense_state["pnl_reset_timestamp_ms"]
    assert sparse_state["runtime"].red_seen_in_episode() == (
        dense_state["runtime"].red_seen_in_episode()
    )
    for key in (
        "timestamp_ms",
        "tier",
        "red_active_now",
        "red_seen_in_episode",
        "balance",
        "slot_budget",
        "peak_realized_pnl",
        "realized_pnl",
        "unrealized_pnl",
        "drawdown_raw",
        "drawdown_ema",
        "drawdown_score",
    ):
        if isinstance(dense_state["last_metrics"][key], float):
            assert sparse_state["last_metrics"][key] == pytest.approx(
                dense_state["last_metrics"][key], abs=1e-12
            )
        else:
            assert sparse_state["last_metrics"][key] == dense_state["last_metrics"][key]


@pytest.mark.asyncio
async def test_compact_replay_resets_flat_episode_when_historical_upnl_is_omitted():
    """Regression for queen_bee XLM: a flat historical pair has no candle replay.

    Fill-derived non-flat/flat transitions still own the episode boundary. The
    RED episode ended on July 1, so after its 36-hour cooldown expires its
    realized loss must not leak into the empty current episode.
    """
    import numpy as np

    symbol = "XLM/USDC:USDC"
    entry_ts = 1_782_900_182_624
    flatten_ts = 1_782_903_432_626
    now_ms = flatten_ts + 2_160 * 60_000 + 120_000
    first_minute = entry_ts // 60_000 * 60_000
    last_minute = now_ms // 60_000 * 60_000
    timestamps = np.arange(
        first_minute,
        last_minute + 60_000,
        60_000,
        dtype=np.int64,
    )
    flatten_minute = flatten_ts // 60_000 * 60_000
    realized = np.where(timestamps < flatten_minute, 0.0, -30.0)
    unrealized = np.where(timestamps < flatten_minute, np.nan, 0.0)
    history = {
        "hsl_coin_compact_replay": {
            "timestamps": timestamps,
            "balances": np.full(len(timestamps), 100.0, dtype=np.float64),
            "realized_pnl": realized.copy(),
            "pair_values": {
                ("long", symbol): {
                    "realized_pnl": realized,
                    # Coin replay omits prices/UPnL while a historical pair is
                    # open when it is flat now and has no panic marker.
                    "unrealized_pnl": unrealized,
                }
            },
        },
        "panic_flatten_events": [],
        "fill_events": [
            {
                "timestamp": entry_ts,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 62.0,
                "pnl": 0.0,
            },
            {
                "timestamp": flatten_ts,
                "symbol": symbol,
                "pside": "long",
                "action": "decrease",
                "qty": 62.0,
                "pnl": -30.0,
                "pb_order_type": "close_grid_long",
            },
        ],
    }
    bot = _flat_position_bot(symbol)
    bot.hsl["long"]["cooldown_minutes_after_red"] = 2_160.0
    bot.get_exchange_time = lambda: now_ms

    async def fake_history(current_balance=None, **kwargs):
        return history

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert state["pnl_reset_timestamp_ms"] == flatten_ts + 1
    assert state["last_metrics"]["tier"] == "green"
    assert state["last_metrics"]["realized_pnl"] == pytest.approx(0.0)
    assert symbol not in bot._runtime_forced_modes["long"]


@pytest.mark.parametrize(
    ("restart_policy", "cooldown_minutes", "should_raise"),
    [
        ("always", 5.0, False),
        ("always", 20.0, True),
        ("threshold", 5.0, True),
    ],
)
@pytest.mark.asyncio
async def test_held_coin_replay_bounds_missing_prices_only_after_proven_cooldown_gap(
    restart_policy, cooldown_minutes, should_raise
):
    """Old closed episodes must not strand an ``always`` held position.

    GateIO exposes only a recent 1m window.  A held pair may therefore have
    fill/PnL history for an old closed episode whose candles are no longer
    fetchable, while its current episode has complete price history.  The old
    episode is irrelevant only when a proven flat gap exceeds every possible
    cooldown bridge.
    """
    import numpy as np

    symbol = "UNI/USDT:USDT"
    timestamps = np.arange(1, 21, dtype=np.int64) * 60_000
    old_entry_ts = 60_001
    old_flatten_ts = 120_001
    current_entry_ts = 900_001
    old_entry_fee = -5.0
    old_close_pnl = -55.0
    old_realized_pnl = -60.0
    realized = np.zeros(len(timestamps), dtype=np.float64)
    realized[0] = old_entry_fee
    realized[1:] = old_realized_pnl
    balances = np.full(len(timestamps), 100.0, dtype=np.float64)
    balances[0] = 100.0 - old_close_pnl
    unrealized = np.zeros(len(timestamps), dtype=np.float64)
    unrealized[0] = np.nan
    unrealized[14:] = -1.0
    history = {
        "hsl_coin_compact_replay": {
            "timestamps": timestamps,
            "balances": balances,
            "realized_pnl": realized,
            "pair_values": {
                ("long", symbol): {
                    "realized_pnl": realized,
                    "unrealized_pnl": unrealized,
                }
            },
        },
        "panic_flatten_events": [],
        "fill_events": [
            {
                "timestamp": old_entry_ts,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
                "pnl": 0.0,
                "fee_paid": old_entry_fee,
            },
            {
                "timestamp": old_flatten_ts,
                "symbol": symbol,
                "pside": "long",
                "action": "decrease",
                "qty": 1.0,
                "pnl": old_close_pnl,
            },
            {
                "timestamp": current_entry_ts,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
                "pnl": 0.0,
            },
        ],
    }
    bot = make_coin_bot()
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.hsl["long"]["restart_after_red_policy"] = restart_policy
    bot.hsl["long"]["cooldown_minutes_after_red"] = cooldown_minutes
    bot.get_exchange_time = lambda: int(timestamps[-1])

    async def current_upnl(pside=None, symbol=None):
        return -1.0 if pside == "long" and symbol == "UNI/USDT:USDT" else 0.0

    bot._calc_upnl_sum_strict = current_upnl

    async def fake_history(current_balance=None, **kwargs):
        return history

    bot.get_balance_equity_history = fake_history

    if should_raise:
        with pytest.raises(
            ValueError,
            match="missing required unrealized_pnl_by_coin_pside value",
        ):
            await bot._equity_hard_stop_initialize_coin_from_history()
        return

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["last_metrics"]["timestamp_ms"] == int(timestamps[-1])
    assert state["last_metrics"]["realized_pnl"] == pytest.approx(0.0)
    assert state["last_metrics"]["unrealized_pnl"] == pytest.approx(-1.0)
    assert state["last_metrics"]["tier"] == "green"
    assert state["pnl_reset_timestamp_ms"] is None


@pytest.mark.asyncio
async def test_compact_replay_keeps_held_and_ambiguous_pairs_dense():
    import numpy as np

    symbol = "A"
    row_count = 20

    async def run(position_size, fill_events):
        bot = _flat_position_bot(symbol)
        bot.positions[symbol]["long"] = {"size": position_size, "price": 100.0}
        captured = []

        def record_event(event_type, tags, data, **kwargs):
            captured.append((event_type, data))

        bot._monitor_record_event = record_event
        payload = {
            "hsl_coin_compact_replay": {
                "timestamps": np.arange(1, row_count + 1, dtype=np.int64) * 60_000,
                "balances": np.full(row_count, 100.0, dtype=np.float64),
                "realized_pnl": np.zeros(row_count, dtype=np.float64),
                "pair_values": {
                    ("long", symbol): {
                        "realized_pnl": np.zeros(row_count, dtype=np.float64),
                        "unrealized_pnl": np.zeros(row_count, dtype=np.float64),
                    }
                },
            },
            "panic_flatten_events": [],
            "fill_events": fill_events,
        }

        async def fake_history(current_balance=None, **kwargs):
            return payload

        bot.get_balance_equity_history = fake_history
        bot.get_exchange_time = lambda: row_count * 60_000
        await bot._equity_hard_stop_initialize_coin_from_history()
        return next(data for event_type, data in captured if event_type == "hsl.replay.completed")

    held = await run(1.0, [])
    assert held["candidate_rows"] == row_count
    assert held["dense_replay_pairs"] == 1
    assert held["dense_fallback_pairs"] == 0
    assert held["sparse_replay_pairs"] == 0

    ambiguous = await run(
        0.0,
        [
            {
                "timestamp": 60_001,
                "symbol": symbol,
                "pside": "long",
                "action": "unknown",
                "qty": 1.0,
            }
        ],
    )
    assert ambiguous["replay_strategy"] == "dense_compact"
    assert ambiguous["candidate_rows"] == row_count
    assert ambiguous["dense_equivalent_rows"] == row_count
    assert ambiguous["dense_replay_pairs"] == 1
    assert ambiguous["dense_fallback_pairs"] == 1
    assert ambiguous["sparse_replay_pairs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_sparse_and_dense_replay_reject_nan_after_pair_coverage(compact):
    symbol = "A"
    history = {
        "timeline": [
            {
                "timestamp": minute * 60_000,
                "balance": 100.0,
                "realized_pnl": 0.0,
                "realized_pnl_by_coin_pside": {
                    symbol: {"long": value, "short": 0.0}
                },
                "unrealized_pnl_by_coin_pside": {
                    symbol: {"long": value, "short": 0.0}
                },
            }
            for minute, value in ((1, 0.0), (2, 0.0), (3, float("nan")))
        ],
        "panic_flatten_events": [],
        "fill_events": [
            {
                "timestamp": 60_001,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
            },
            {
                "timestamp": 120_001,
                "symbol": symbol,
                "pside": "long",
                "action": "decrease",
                "qty": 1.0,
            },
        ],
    }
    bot = _flat_position_bot(symbol)

    async def fake_history(current_balance=None, **kwargs):
        return _coin_history_as_compact(history, symbol) if compact else history

    bot.get_balance_equity_history = fake_history
    bot.get_exchange_time = lambda: 180_000

    with pytest.raises(ValueError, match="realized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_coin_hsl_replay_latches_cooldown_from_red_episode_ordinary_flatten(
    compact,
):
    # Episode crosses RED (drawdown 0.6 >= 0.5) and is flattened at 150_000 by an
    # ordinary close with no panic marker: cooldown must anchor at the flatten fill.
    symbol = "A"
    bot = _flat_position_bot(symbol)
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        history = _red_episode_history(
            symbol, flatten_pnl=-30.0, rows_upnl=[0.0, 0.0]
        )
        return _coin_history_as_compact(history, symbol) if compact else history

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is False
    assert state["cooldown_until_ms"] == 150_000 + 300_000
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 150_000
    assert state["pnl_reset_timestamp_ms"] == 150_001
    assert bot._runtime_forced_modes["long"][symbol] == "graceful_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_coin_hsl_replay_latches_no_restart_from_red_episode_ordinary_flatten(
    compact,
):
    # Same shape but the episode drawdown (1.6) breaches the no-restart threshold
    # (0.9) under policy=threshold: terminal halt, no cooldown.
    symbol = "A"
    bot = _flat_position_bot(symbol)
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        history = _red_episode_history(
            symbol, flatten_pnl=-80.0, rows_upnl=[0.0, 0.0]
        )
        return _coin_history_as_compact(history, symbol) if compact else history

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is True
    assert state["cooldown_until_ms"] is None
    assert bot._runtime_forced_modes["long"][symbol] == "graceful_stop"


@pytest.mark.asyncio
async def test_coin_hsl_replay_never_policy_latches_no_restart_on_red_episode():
    symbol = "A"
    bot = _flat_position_bot(symbol)
    bot.hsl["long"]["restart_after_red_policy"] = "never"
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return _red_episode_history(symbol, flatten_pnl=-30.0, rows_upnl=[0.0, 0.0])

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is True
    assert state["cooldown_until_ms"] is None


@pytest.mark.asyncio
async def test_coin_hsl_replay_red_recovered_before_ordinary_flatten_still_cools_down():
    # RED is seen mid-episode (row 120_000, drawdown 0.6) but the current sample
    # has recovered by the flatten row: red_seen_in_episode still activates
    # cooldown, and no-restart is evaluated on at-flatten drawdown (small), so
    # the halt is a cooldown, not terminal.
    symbol = "A"
    bot = _flat_position_bot(symbol)
    bot.get_exchange_time = lambda: 300_000

    async def fake_history(current_balance=None, **kwargs):
        return _red_episode_history(
            symbol,
            flatten_pnl=-1.0,
            rows_upnl=[0.0, -30.0, -1.0, 0.0],
            flatten_ts=210_000,
        )

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is False
    assert state["cooldown_until_ms"] == 210_000 + 300_000
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 210_000
    assert bot._runtime_forced_modes["long"][symbol] == "graceful_stop"


@pytest.mark.asyncio
async def test_coin_hsl_replay_red_free_ordinary_flatten_resets_without_stop():
    # Control: an episode that never saw RED and flattens ordinarily keeps the
    # existing plain episode reset with no cooldown/no-restart accounting.
    symbol = "A"
    bot = _flat_position_bot(symbol)
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return _red_episode_history(symbol, flatten_pnl=-1.0, rows_upnl=[0.0, 0.0])

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["no_restart_latched"] is False
    assert state["cooldown_until_ms"] is None
    assert state["last_stop_event"] is None
    assert symbol not in bot._runtime_forced_modes["long"]


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_coin_hsl_replay_splits_same_minute_flatten_and_reentry(compact):
    """A zero crossing is an episode boundary even when the minute ends held."""
    symbol = "A"
    flatten_ts = 150_000
    reentry_ts = 160_000
    history = _red_episode_history(
        symbol,
        flatten_pnl=-1.0,
        # The large row-end UPnL belongs to the re-entry, not the episode that
        # ended ten seconds earlier. It must not turn that old episode RED.
        rows_upnl=[0.0, -80.0, 0.0],
        flatten_ts=flatten_ts,
    )
    history["fill_events"].append(
        {
            "timestamp": reentry_ts,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "pnl": 0.0,
        }
    )
    bot = _flat_position_bot(symbol)
    bot.positions[symbol]["long"] = {"size": 1.0, "price": 100.0}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return _coin_history_as_compact(history, symbol) if compact else history

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["last_stop_event"] is None
    assert state["pnl_reset_timestamp_ms"] == flatten_ts + 1
    assert state["runtime"].red_seen_in_episode() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("later_pnl", [-30.0, 60.0])
async def test_coin_hsl_replay_uses_first_of_multiple_same_minute_flattens(
    compact, later_pnl
):
    """Later-episode PnL and balance must not alter the first RED episode."""
    symbol = "A"
    first_flatten_ts = 150_000
    history = _red_episode_history(
        symbol,
        flatten_pnl=-30.0,
        rows_upnl=[0.0, 0.0],
        flatten_ts=first_flatten_ts,
    )
    history["fill_events"].extend(
        [
            {
                "timestamp": 160_000,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
                "pnl": 0.0,
            },
            {
                "timestamp": 170_000,
                "symbol": symbol,
                "pside": "long",
                "action": "decrease",
                "qty": 1.0,
                "pnl": later_pnl,
            },
        ]
    )
    # The row closes at account balance 100 - 30 + later_pnl. The first RED
    # episode nevertheless ended with balance 70 at 150_000. Reusing the row's
    # ending balance either hides RED after the later profit or falsely makes
    # the earlier episode terminal after the later loss.
    for row in history["timeline"]:
        if row["timestamp"] == 120_000:
            row["balance"] = 70.0 + later_pnl
            row["realized_pnl"] = -30.0 + later_pnl
            row["realized_pnl_by_coin_pside"][symbol]["long"] = (
                -30.0 + later_pnl
            )
    bot = _flat_position_bot(symbol)
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return _coin_history_as_compact(history, symbol) if compact else history

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is False
    assert state["last_stop_event"]["stop_event_timestamp_ms"] == first_flatten_ts
    assert state["pnl_reset_timestamp_ms"] == first_flatten_ts + 1
    assert state["cooldown_until_ms"] == first_flatten_ts + 300_000


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_does_not_latch_recovered_red_without_panic_marker():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["runtime"].red_latched() is False
    assert state["runtime"].tier() == "green"
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"
    assert state["pending_red_since_ms"] is None
    assert state["pending_stop_event"] is None
    assert symbol not in bot._runtime_forced_modes["long"]
    assert bot._equity_hard_stop_coin_red_active() is False


@pytest.mark.asyncio
async def test_coin_hsl_check_defers_stop_event_until_flat_confirmation():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def calc_upnl(pside=None, symbol=None):
        return -80.0

    async def fail_compute(*_args, **_kwargs):
        raise AssertionError("coin HSL must not snapshot stop event at RED trigger time")

    bot._calc_upnl_sum_strict = calc_upnl
    bot._equity_hard_stop_compute_coin_stop_event = fail_compute
    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        60_000,
        100.0,
        0.0,
        0.0,
        0.0,
    )

    out = await bot._equity_hard_stop_check_coin()

    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["tier"] == "red"
    assert state["pending_red_since_ms"] == 180_000
    assert state["pending_stop_event"] is None
    assert bot._runtime_forced_modes["long"][symbol] == "panic"


@pytest.mark.asyncio
async def test_coin_stop_event_does_not_read_unscoped_account_pnl():
    bot = make_coin_bot()
    symbol = "A"
    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        180_000,
        100.0,
        0.0,
        0.0,
        0.0,
    )

    def fail_unscoped_read(*_args, **_kwargs):
        raise AssertionError("coin stop finalization must not read account-wide PnL")

    bot._equity_hard_stop_realized_pnl_now = fail_unscoped_read

    stop_event = await bot._equity_hard_stop_compute_coin_stop_event(
        "long", symbol, 180_000
    )

    assert stop_event["realized_pnl_total"] is None


@pytest.mark.asyncio
async def test_coin_hsl_finalize_uses_latest_flatten_fill_for_reset_boundary():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_finalize"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    bot._pnls_manager = make_fake_pnls_manager(
        [
            {
                "timestamp": 170_000,
                "symbol": symbol,
                "pside": "long",
                "pb_order_type": "close_panic_long",
                "pnl": -12.0,
                "fee_paid": -0.1,
            }
        ]
    )

    stop_ts_ms = bot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
        "long", symbol=symbol, since_ms=state["pending_red_since_ms"]
    )
    assert stop_ts_ms == 170_000
    stop_event = await bot._equity_hard_stop_compute_coin_stop_event(
        "long", symbol, stop_ts_ms
    )
    await bot._equity_hard_stop_finalize_coin_red_stop("long", symbol, stop_event)

    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 170_000
    assert state["pnl_reset_timestamp_ms"] == 170_001
    assert state["cooldown_until_ms"] == 470_000
    assert state["red_trigger_event_emitted"] is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_RED_TRIGGERED]
    assert len(events) == 1
    assert events[0].cycle_id == "cy_coin_finalize"
    assert events[0].pside == "long"
    assert events[0].symbol == symbol
    assert events[0].reason_code == "coin_red_stop_finalized"
    assert events[0].data["stop_event_timestamp_ms"] == 170_000
    assert events[0].data["cooldown_until_ms"] == 470_000
    assert [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER
    ] == []
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_finalize_emits_flat_without_order_event():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_flat_finalize"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000

    stop_event = await bot._equity_hard_stop_compute_coin_stop_event(
        "long", symbol, 180_000
    )
    await bot._equity_hard_stop_finalize_coin_red_stop(
        "long",
        symbol,
        stop_event,
        finalized_without_order=True,
        flat_confirmations=2,
        entry_orders=0,
        nonpanic_close_orders=0,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER
    ]
    assert len(events) == 1
    event = events[0]
    assert event.cycle_id == "cy_coin_flat_finalize"
    assert event.pside == "long"
    assert event.symbol == symbol
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.HSL_RED_FINALIZED_WITHOUT_EXCHANGE_ORDER
    assert event.data["no_exchange_close_needed"] is True
    assert event.data["exchange_close_order_submitted"] is False
    assert event.data["panic_order_submitted_count"] == 0
    assert event.data["symbol_position_open"] is False
    assert event.data["position_count"] == 0
    assert event.data["entry_orders"] == 0
    assert event.data["nonpanic_close_orders"] == 0
    assert event.data["flat_confirmations"] == 2
    assert event.data["stop_event_timestamp_ms"] == 180_000
    assert event.data["stop_event_anchor_source"] == "provided_stop_event"
    assert event.data["stop_event_anchor_timestamp_ms"] == 180_000
    assert event.data["stop_event_anchor_fallback_used"] is False
    assert event.data["cooldown_until_ms"] == 480_000
    assert event.data["drawdown_raw"] == 0.0
    red_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_RED_TRIGGERED
    ]
    assert len(red_events) == 1
    red_event = red_events[0]
    assert red_event.level == "info"
    assert red_event.status == "succeeded"
    assert red_event.reason_code == "coin_red_stop_finalized"
    assert red_event.data["no_exchange_close_needed"] is True
    assert red_event.data["exchange_close_order_submitted"] is False
    assert red_event.data["panic_order_submitted_count"] == 0
    assert red_event.data["symbol_position_open"] is False
    assert red_event.data["entry_orders"] == 0
    assert red_event.data["nonpanic_close_orders"] == 0
    assert red_event.data["flat_confirmations"] == 2
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_finalize_flat_without_order_event_records_flatten_fill_anchor():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_flat_fill_anchor"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    bot._pnls_manager = make_fake_pnls_manager(
        [
            {
                "timestamp": 170_000,
                "symbol": symbol,
                "pside": "long",
                "pb_order_type": "close_panic_long",
                "pnl": -12.0,
                "fee_paid": -0.1,
            }
        ]
    )

    stop_ts_ms = bot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
        "long", symbol=symbol, since_ms=state["pending_red_since_ms"]
    )
    assert stop_ts_ms == 170_000
    stop_event = await bot._equity_hard_stop_compute_coin_stop_event(
        "long", symbol, stop_ts_ms
    )
    await bot._equity_hard_stop_finalize_coin_red_stop(
        "long",
        symbol,
        stop_event,
        finalized_without_order=True,
        flat_confirmations=2,
        entry_orders=0,
        nonpanic_close_orders=0,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER
    ]
    assert len(events) == 1
    event = events[0]
    assert event.cycle_id == "cy_coin_flat_fill_anchor"
    assert event.data["stop_event_timestamp_ms"] == 170_000
    assert event.data["stop_event_anchor_source"] == "provided_stop_event"
    assert event.data["stop_event_anchor_timestamp_ms"] == 170_000
    assert event.data["stop_event_anchor_fallback_used"] is False
    assert event.data["cooldown_until_ms"] == 470_000
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_finalize_does_not_duplicate_prior_red_trigger_event():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_finalize_duplicate"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    state["red_trigger_event_emitted"] = True

    stop_event = await bot._equity_hard_stop_compute_coin_stop_event(
        "long", symbol, 170_000
    )
    await bot._equity_hard_stop_finalize_coin_red_stop("long", symbol, stop_event)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    red_events = [event for event in sink.events if event.event_type == EventTypes.HSL_RED_TRIGGERED]
    cooldown_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_COOLDOWN_STARTED
    ]
    assert red_events == []
    assert len(cooldown_events) == 1
    assert cooldown_events[0].reason_code == "coin_red_stop_finalized"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_rebases_lookback_window_realized_points():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["red_threshold"] = 0.9
    bot.config["live"]["pnls_max_lookback_days"] = 1.0 / 1440.0
    bot.get_exchange_time = lambda: 240_000
    fill_events = [
        {"timestamp": 120_000, "symbol": symbol, "pside": "long", "pnl": -20.0},
        {"timestamp": 240_000, "symbol": symbol, "pside": "long", "pnl": -35.0},
    ]
    bot._pnls_manager = make_fake_pnls_manager(fill_events)

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 240_000,
                    "balance": 100.0,
                    "realized_pnl": -55.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -55.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": fill_events,
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["runtime"].red_latched() is False
    assert state["last_metrics"]["realized_pnl"] == pytest.approx(-35.0)
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(0.70)


@pytest.mark.parametrize(
    ("restart_after_red_policy", "expected_latched", "expected_cooldown_until_ms"),
    [
        ("threshold", True, None),
        ("always", False, 420_500),
        ("never", True, None),
    ],
)
@pytest.mark.asyncio
async def test_coin_hsl_history_replay_honors_restart_after_red_policy(
    restart_after_red_policy, expected_latched, expected_cooldown_until_ms
):
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["no_restart_drawdown_threshold"] = 0.7
    bot.hsl["long"]["restart_after_red_policy"] = restart_after_red_policy

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is expected_latched
    assert state["cooldown_until_ms"] == expected_cooldown_until_ms
    assert state["last_stop_event"]["drawdown_raw"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_ignores_panic_marker_without_reconstructed_red():
    bot = make_coin_bot()
    symbol = "A"
    writes = []
    bot._equity_hard_stop_write_latch = (
        lambda pside, payload, symbol=None: writes.append((pside, symbol, payload))
        or "/tmp/hsl_coin_ignored.json"
    )

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [
                {
                    "timestamp": 120_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history
    bot.get_exchange_time = lambda: 180_000

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert state["last_stop_event"] is None
    assert state["runtime"].red_latched() is False
    assert writes == []


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_ignores_raw_red_pending_panic_marker(caplog):
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["ema_span_minutes"] = 100.0
    writes = []
    bot._equity_hard_stop_write_latch = (
        lambda pside, payload, symbol=None: writes.append((pside, symbol, payload))
        or "/tmp/hsl_coin_raw_pending_ignored.json"
    )

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [
                {
                    "timestamp": 120_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history
    bot.get_exchange_time = lambda: 180_000

    with caplog.at_level(logging.WARNING):
        await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert state["last_stop_event"] is None
    assert state["runtime"].red_latched() is False
    assert writes == []
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ignored historical coin panic marker without reconstructed RED" in message
        and "drawdown_raw=1.000000" in message
        and "drawdown_score=0.019802" in message
        for message in messages
    )


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_coin_timeline_fields():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="realized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_compact_replay_requires_nonflat_upnl():
    import numpy as np

    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0},
        }
    }

    async def fake_history(current_balance=None, **kwargs):
        return {
            "hsl_coin_compact_replay": {
                "timestamps": np.asarray([60_000], dtype=np.int64),
                "balances": np.asarray([100.0], dtype=np.float64),
                "realized_pnl": np.asarray([0.0], dtype=np.float64),
                "pair_values": {
                    ("long", symbol): {
                        "realized_pnl": np.asarray([0.0], dtype=np.float64),
                        "unrealized_pnl": np.asarray([np.nan], dtype=np.float64),
                    }
                },
            },
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="unrealized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_open_position_missing_history_uses_current_sample():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_open_position_empty_coin_history_uses_current_sample():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_allows_leading_rows_before_first_fill():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_relevant_symbol_fields():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="missing required coin HSL symbol"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_startup_skips_flat_nonpanic_history_missing_upnl():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                },
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot._equity_hard_stop_coin_initialized is True
    assert bot._runtime_forced_modes == {"long": {}, "short": {}}


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_allows_flat_realized_only_rows():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                },
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_resets_current_episode_after_nonpanic_flatten():
    # The episode stays below RED (drawdown 0.6 < 0.7): an ordinary flatten is a
    # plain episode reset. RED-crossing episodes now latch cooldown/no-restart
    # instead (see test_coin_hsl_replay_latches_cooldown_from_red_episode_*).
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["red_threshold"] = 0.7
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }
    bot.get_exchange_time = lambda: 300_000
    fill_events = [
        {
            "timestamp": 60_500,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": 180_500,
            "symbol": symbol,
            "pside": "long",
            "action": "decrease",
            "qty": 1.0,
            "price": 80.0,
            "pnl": -20.0,
            "pb_order_type": "close_grid_long",
        },
        {
            "timestamp": 240_500,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
    ]

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -30.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 80.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 240_000,
                    "balance": 80.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": fill_events,
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["pnl_reset_timestamp_ms"] == 180_501
    assert state["halted"] is False
    assert state["last_stop_event"] is None
    assert state["last_metrics"]["timestamp_ms"] == 300_000
    assert state["last_metrics"]["tier"] == "green"
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(0.0)
    assert bot._runtime_forced_modes == {"long": {}, "short": {}}


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_upnl_for_carry_in_decrease():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="unrealized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_upnl_for_flat_ambiguous_decrease():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_000,
                    "minute_timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                }
            ],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                    "pb_order_type": "close_panic_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="unrealized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_unresolved_panic_residue_on_restart():
    bot = make_coin_bot(policy="normal")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 200_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 121_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 421_500
    assert state["cooldown_unresolved_residue"] is True
    assert state["cooldown_intervention_active"] is False
    assert bot._runtime_forced_modes["long"][symbol] == "panic"

    changed = await bot._equity_hard_stop_handle_coin_position_during_cooldown(
        "long", symbol, 200_000
    )
    assert changed is False
    assert state["halted"] is True


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_manual_cooldown_intervention_on_restart():
    bot = make_coin_bot(policy="manual")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 100_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                },
                {
                    "timestamp": 130_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "pb_order_type": "entry_initial_normal_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 400_000
    assert state["cooldown_unresolved_residue"] is False
    assert state["cooldown_intervention_active"] is True
    assert bot._runtime_forced_modes["long"][symbol] == "manual"


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_normal_cooldown_intervention_as_override():
    bot = make_coin_bot(policy="normal")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 100_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                },
                {
                    "timestamp": 130_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "pb_order_type": "entry_initial_normal_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert symbol not in bot._runtime_forced_modes["long"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,expected_mode",
    [
        ("manual", "manual"),
        ("tp_only", "tp_only_with_active_entry_cancellation"),
    ],
)
async def test_coin_hsl_check_preserves_cooldown_policy_forced_mode(policy, expected_mode):
    bot = make_coin_bot(policy=policy)
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    state = bot._hsl_coin_state("long", symbol)
    state["halted"] = True
    state["cooldown_until_ms"] = 300_000

    await bot._equity_hard_stop_check_coin()

    assert bot._runtime_forced_modes["long"][symbol] == expected_mode


@pytest.mark.asyncio
async def test_coin_hsl_check_tp_only_orange_blocks_flat_initial_entries():
    bot = make_coin_bot()
    open_symbol = "A"
    flat_symbol = "B"
    bot.positions = {
        open_symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}},
        flat_symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0}},
    }

    async def calc_upnl(pside=None, symbol=None):
        return -20.0

    bot._calc_upnl_sum_strict = calc_upnl
    bot._equity_hard_stop_prime_coin_runtime_for_replay("long", open_symbol, 180_000)
    bot._equity_hard_stop_prime_coin_runtime_for_replay("long", flat_symbol, 180_000)

    out = await bot._equity_hard_stop_check_coin()

    assert out[f"long:{open_symbol}"]["tier"] == "orange"
    assert out[f"long:{flat_symbol}"]["tier"] == "orange"
    assert (
        bot._runtime_forced_modes["long"][open_symbol]
        == "tp_only_with_active_entry_cancellation"
    )
    assert (
        bot._runtime_forced_modes["long"][flat_symbol]
        == "tp_only_with_active_entry_cancellation"
    )
