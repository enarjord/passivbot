import logging

import pytest


def _forager_score_weights(volume=0.0, ema_readiness=0.0, volatility=0.0):
    return {
        "volume": float(volume),
        "ema_readiness": float(ema_readiness),
        "volatility": float(volatility),
    }


@pytest.mark.asyncio
async def test_active_candle_refresh_only_fetches_urgent_symbols(monkeypatch):
    import passivbot as pb_mod

    now_ms = 1_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []
            self.last_refresh = {
                "A/USDT:USDT": 1,
                "B/USDT:USDT": now_ms - 10_000,
                "C/USDT:USDT": now_ms - 20_000,
            }

        def is_rate_limited(self):
            return False

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        def get_completed_candle_health(self, symbol, windows=None, now_ms=None):
            return {
                "ok": True,
                "timeframes": {
                    "1m": {
                        "coverage_ok": True,
                        "latest_expected_ts": 60_000,
                        "last_cached_ts": 60_000,
                        "runtime_synthetic_count": 0,
                        "missing_candles": 0,
                    }
                },
            }

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            return []

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 4}}
        active_symbols = [
            "POS/USDT:USDT",
            "OO/USDT:USDT",
            "A/USDT:USDT",
            "B/USDT:USDT",
            "C/USDT:USDT",
        ]
        PB_modes = {
            "long": {
                "A/USDT:USDT": "normal",
                "B/USDT:USDT": "graceful_stop",
                "C/USDT:USDT": "graceful_stop",
            },
            "short": {},
        }
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {"OO/USDT:USDT": [{"id": "1"}]}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False

        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def _maybe_log_candle_refresh(self, *args, **kwargs):
            return None

        def _schedule_forager_candidate_candle_refresh(self):
            return None

        def _ensure_freshness_ledger(self):
            class Ledger:
                def stamp(self, *args, **kwargs):
                    return None

            return Ledger()

        has_position = pb_mod.Passivbot.has_position
        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    await pb_mod.Passivbot.update_ohlcvs_1m_for_actives(bot)

    ttl_by_symbol = {symbol: kwargs["max_age_ms"] for symbol, kwargs in bot.cm.calls}
    assert ttl_by_symbol["POS/USDT:USDT"] == 60_000
    assert ttl_by_symbol["OO/USDT:USDT"] == 60_000
    assert ttl_by_symbol["A/USDT:USDT"] == 60_000
    assert "B/USDT:USDT" not in ttl_by_symbol
    assert "C/USDT:USDT" not in ttl_by_symbol


@pytest.mark.asyncio
async def test_active_candle_refresh_ignores_broad_graceful_stop_universe(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []
            self.last_final = {
                "POS/USDT:USDT": now_ms - 60_000,
                "FRESH/USDT:USDT": now_ms - 60_000,
                "STALE/USDT:USDT": now_ms - 30 * 60_000,
                "OLDER/USDT:USDT": now_ms - 20 * 60_000,
            }
            self.last_refresh = {s: ts for s, ts in self.last_final.items()}

        def is_rate_limited(self):
            return False

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        def get_last_final_ts(self, symbol):
            return int(self.last_final.get(symbol, 0))

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            return []

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 4}}
        active_symbols = [
            "POS/USDT:USDT",
            "FRESH/USDT:USDT",
            "STALE/USDT:USDT",
            "OLDER/USDT:USDT",
        ]
        PB_modes = {
            "long": {
                "FRESH/USDT:USDT": "graceful_stop",
                "STALE/USDT:USDT": "graceful_stop",
                "OLDER/USDT:USDT": "graceful_stop",
            },
            "short": {},
        }
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def _maybe_log_candle_refresh(self, *args, **kwargs):
            return None

        def _schedule_forager_candidate_candle_refresh(self):
            return None

        def _ensure_freshness_ledger(self):
            class Ledger:
                def stamp(self, *args, **kwargs):
                    return None

            return Ledger()

        has_position = pb_mod.Passivbot.has_position
        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    await pb_mod.Passivbot.update_ohlcvs_1m_for_actives(bot)

    called = [symbol for symbol, _kwargs in bot.cm.calls]
    assert called == ["POS/USDT:USDT"]


@pytest.mark.asyncio
async def test_active_candle_refresh_does_not_stamp_when_rate_limited(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)
    stamps = []

    class FakeCM:
        default_window_candles = 120

        def is_rate_limited(self):
            return True

        async def get_candles(self, symbol, **kwargs):
            raise AssertionError(
                "rate-limited active refresh should break before fetch"
            )

        def get_completed_candle_health(self, symbol, windows=None, now_ms=None):
            return {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "coverage_ok": False,
                        "latest_expected_ts": 120_000,
                        "last_cached_ts": 60_000,
                        "missing_candles": 1,
                    }
                },
            }

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 4}}
        active_symbols = ["POS/USDT:USDT"]
        PB_modes = {"long": {}, "short": {}}
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        stop_signal_received = False
        cm = FakeCM()

        def _get_fetch_delay_seconds(self):
            return 0.0

        def _maybe_log_candle_refresh(self, *args, **kwargs):
            return None

        def _ensure_freshness_ledger(self):
            class Ledger:
                def stamp(self, *args, **kwargs):
                    stamps.append((args, kwargs))

            return Ledger()

        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _completed_candle_freshness_signature = (
            pb_mod.Passivbot._completed_candle_freshness_signature
        )

    bot = FakeBot()

    assert await pb_mod.Passivbot.update_ohlcvs_1m_for_actives(bot) is False
    assert stamps == []


@pytest.mark.asyncio
async def test_execute_to_exchange_schedules_forager_candidates_after_planning(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 10_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 4}}
        debug_mode = True
        stop_signal_received = False

        def __init__(self):
            self.execution_scheduled = False
            self.events = []

        async def execution_cycle(self):
            self.events.append("execution_cycle")

        async def calc_orders_to_cancel_and_create(self):
            self.events.append("plan")
            return [], []

        def _schedule_forager_candidate_candle_refresh(self):
            self.events.append("forager_refresh_scheduled")

        async def _refresh_forager_candidate_candles(self):
            raise AssertionError(
                "forager candidate refresh must not block execute_to_exchange"
            )

    bot = FakeBot()
    assert await pb_mod.Passivbot.execute_to_exchange(bot) == ([], [])
    assert bot.events == ["execution_cycle", "plan", "forager_refresh_scheduled"]


@pytest.mark.asyncio
async def test_forager_candidate_refresh_scheduler_coalesces_running_task():
    import asyncio
    import passivbot as pb_mod

    class FakeBot:
        stop_signal_received = False
        config = {"live": {}}

        def __init__(self):
            self.maintainers = {}
            self.started_count = 0
            self.release = asyncio.Event()

        def is_forager_mode(self, pside=None):
            return True

        async def _refresh_forager_candidate_candles(self):
            self.started_count += 1
            await self.release.wait()

        _forager_candidate_candle_refresh_task = (
            pb_mod.Passivbot._forager_candidate_candle_refresh_task
        )
        _schedule_forager_candidate_candle_refresh = (
            pb_mod.Passivbot._schedule_forager_candidate_candle_refresh
        )

    bot = FakeBot()
    pb_mod.Passivbot._schedule_forager_candidate_candle_refresh(bot)
    pb_mod.Passivbot._schedule_forager_candidate_candle_refresh(bot)
    await asyncio.sleep(0)

    task = bot.maintainers.get("forager_candidate_candle_refresh")
    assert task is not None
    assert bot.started_count == 1

    bot.release.set()
    await task


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_uses_cache_only_for_secondary_forager_symbols(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def __init__(self):
            self.calls = []
            self.last_refresh = {
                "A/USDT:USDT": 1,
                "B/USDT:USDT": now_ms - 10_000,
            }

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("qv", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_log_range(
            self, symbol, *, span, tf=None, max_age_ms=None, **kwargs
        ):
            self.calls.append(("lr", symbol, tf, int(max_age_ms)))
            return 1.0

    class FakeBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 4,
                "max_forager_candle_staleness_minutes": 10,
            }
        }
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "ema_span_0":
                return 10.0
            if key == "ema_span_1":
                return 20.0
            if key == "entry_volatility_ema_span_hours":
                return 2.0
            return 0.0

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 5.0
            if key == "forager_volatility_ema_span_1m":
                return 7.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights()
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    symbols = ["POS/USDT:USDT", "A/USDT:USDT", "B/USDT:USDT"]
    (
        _m1_close_emas,
        _m1_volume_emas,
        _m1_log_range_emas,
        _h1_log_range_emas,
        _volumes_long,
        _log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, symbols, modes={})

    by_symbol = {}
    for kind, symbol, tf, max_age_ms in bot.cm.calls:
        by_symbol.setdefault(symbol, []).append((kind, tf, max_age_ms))

    assert all(max_age <= 600_000 for _kind, _tf, max_age in by_symbol["POS/USDT:USDT"])
    assert "A/USDT:USDT" not in by_symbol
    assert "B/USDT:USDT" in by_symbol
    assert all(
        max_age >= 365 * 24 * 3600 * 1000
        for _kind, _tf, max_age in by_symbol["B/USDT:USDT"]
    )
    assert bot._orchestrator_ema_unavailable_symbols == {"A/USDT:USDT"}


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_fetches_flat_default_normal_planning_symbols(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)
    symbol = "ENTRY/USDT:USDT"

    class FakeCM:
        def __init__(self):
            self.calls = []
            self.last_refresh = {symbol: 0}

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            self.calls.append(("close", symbol, int(max_age_ms), allow_remote_fetch))
            return 1.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            self.calls.append(("qv", symbol, int(max_age_ms), allow_remote_fetch))
            return 1.0

        async def get_latest_ema_log_range(
            self,
            symbol,
            *,
            span,
            tf=None,
            max_age_ms=None,
            allow_remote_fetch=True,
            **kwargs,
        ):
            self.calls.append(("lr", symbol, int(max_age_ms), allow_remote_fetch))
            return 1.0

    class FakeBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 4,
                "max_forager_candle_staleness_minutes": 10,
            }
        }
        positions = {symbol: {"long": {"size": 0.0}, "short": {"size": 0.0}}}
        open_orders = {}
        PB_modes = {"long": {}, "short": {}}
        active_symbols = [symbol]
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "ema_span_0":
                return 10.0
            if key == "ema_span_1":
                return 20.0
            if key == "entry_volatility_ema_span_hours":
                return 2.0
            return 0.0

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 5.0
            if key == "forager_volatility_ema_span_1m":
                return 7.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights()
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], modes={})

    assert {call[1] for call in bot.cm.calls} == {symbol}
    assert all(call[3] is True for call in bot.cm.calls)
    assert all(call[2] < 365 * 24 * 3600 * 1000 for call in bot.cm.calls)
    assert bot._orchestrator_ema_unavailable_symbols == set()


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_marks_missing_required_forager_ema_unavailable(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)
    symbol = "HYPE/USDT:USDT"

    class FakeCM:
        def get_last_refresh_ms(self, symbol):
            return now_ms

        def get_completed_candle_health(self, symbol, windows=None, now_ms=None):
            return {"ok": True, "timeframes": {"1m": {"coverage_ok": True}}}

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            return 100.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            return float("nan")

        async def get_latest_ema_log_range(
            self,
            symbol,
            *,
            span,
            tf=None,
            max_age_ms=None,
            allow_remote_fetch=True,
            **kwargs,
        ):
            return 0.001

    class FakeBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 4,
                "max_forager_candle_staleness_minutes": 10,
            }
        }
        positions = {}
        open_orders = {}
        PB_modes = {"long": {}, "short": {}}
        active_symbols = []
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def _get_fetch_delay_seconds(self):
            return 0.0

        def _strategy_params_to_rust_dict(self, pside, symbol):
            return {
                "ema_span_0": 10.0,
                "ema_span_1": 20.0,
                "volatility_ema_span_1m": 5.0,
                "volatility_ema_span_1h": 0.0,
                "entry": {
                    "threshold_volatility_1m_weight": 0.0,
                    "retracement_volatility_1m_weight": 0.0,
                },
                "close": {
                    "threshold_volatility_1m_weight": 0.0,
                    "retracement_volatility_1m_weight": 0.0,
                },
            }

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 760.0 if pside == "long" else 0.0
            if key == "forager_volatility_ema_span_1m":
                return 0.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return {
                    "volume": 1.0 if pside == "long" else 0.0,
                    "ema_readiness": 0.0,
                    "volatility": 0.0,
                }
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _mode_override_to_orchestrator_mode = (
            pb_mod.Passivbot._mode_override_to_orchestrator_mode
        )
        _pb_mode_to_orchestrator_mode = pb_mod.Passivbot._pb_mode_to_orchestrator_mode
        _pside_blocks_new_entries = pb_mod.Passivbot._pside_blocks_new_entries
        _completed_candle_health_now_ms = pb_mod.Passivbot._completed_candle_health_now_ms
        _completed_candle_tail_gap_fallback_signature = (
            pb_mod.Passivbot._completed_candle_tail_gap_fallback_signature
        )
        _active_candle_tail_gap_max_ms = pb_mod.Passivbot._active_candle_tail_gap_max_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    (
        m1_close_emas,
        m1_volume_emas,
        _m1_log_range_emas,
        _h1_log_range_emas,
        _volumes_long,
        _log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(
        bot, [symbol], modes=bot.PB_modes
    )

    assert m1_close_emas[symbol]
    assert m1_volume_emas[symbol] == {}
    assert bot._orchestrator_ema_unavailable_symbols == {symbol}


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_marks_flat_forager_candidate_required_m1_lr_unavailable(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)
    symbol = "NEAR/USDT:USDT"

    class FakeCM:
        def get_last_refresh_ms(self, symbol):
            return now_ms

        def get_completed_candle_health(self, symbol, windows=None, now_ms=None):
            return {"ok": True, "timeframes": {"1m": {"coverage_ok": True}}}

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            return 100.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            return 1.0

        async def get_latest_ema_log_range(
            self,
            symbol,
            *,
            span,
            tf=None,
            max_age_ms=None,
            allow_remote_fetch=True,
            **kwargs,
        ):
            return float("nan")

    class FakeBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 4,
                "max_forager_candle_staleness_minutes": 10,
                "strategy_kind": "trailing_martingale",
            }
        }
        positions = {symbol: {"long": {"size": 0.0}, "short": {"size": 0.0}}}
        open_orders = {}
        PB_modes = {"long": {}, "short": {}}
        active_symbols = []
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def has_position(self, pside=None, symbol=None):
            if pside is None:
                return any(
                    abs(float(pos.get("size", 0.0) or 0.0)) > 0.0
                    for pos in self.positions.get(symbol, {}).values()
                )
            return (
                abs(float(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0)))
                > 0.0
            )

        def _get_fetch_delay_seconds(self):
            return 0.0

        def _strategy_params_to_rust_dict(self, pside, symbol):
            return {
                "ema_span_0": 10.0,
                "ema_span_1": 20.0,
                "volatility_ema_span_1m": 5.0,
                "volatility_ema_span_1h": 0.0,
                "entry": {
                    "threshold_volatility_1m_weight": 1.0,
                    "retracement_volatility_1m_weight": 0.0,
                    "threshold_volatility_1h_weight": 0.0,
                    "retracement_volatility_1h_weight": 0.0,
                },
                "close": {
                    "threshold_volatility_1m_weight": 0.0,
                    "retracement_volatility_1m_weight": 0.0,
                    "threshold_volatility_1h_weight": 0.0,
                    "retracement_volatility_1h_weight": 0.0,
                },
            }

        def bot_value(self, pside, key):
            if key in {"forager_volume_ema_span_1m", "forager_volatility_ema_span_1m"}:
                return 0.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights()
            return 0.0

        _mode_override_to_orchestrator_mode = pb_mod.Passivbot._mode_override_to_orchestrator_mode
        _pb_mode_to_orchestrator_mode = pb_mod.Passivbot._pb_mode_to_orchestrator_mode
        _pside_blocks_new_entries = pb_mod.Passivbot._pside_blocks_new_entries
        _completed_candle_health_now_ms = pb_mod.Passivbot._completed_candle_health_now_ms
        _completed_candle_tail_gap_fallback_signature = (
            pb_mod.Passivbot._completed_candle_tail_gap_fallback_signature
        )
        _active_candle_tail_gap_max_ms = pb_mod.Passivbot._active_candle_tail_gap_max_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    (
        m1_close_emas,
        _m1_volume_emas,
        m1_log_range_emas,
        _h1_log_range_emas,
        _volumes_long,
        _log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(
        bot, [symbol], modes=bot.PB_modes
    )

    assert m1_close_emas[symbol] == {}
    assert m1_log_range_emas[symbol] == {}
    assert bot._orchestrator_ema_unavailable_symbols == {symbol}

    bot_active = FakeBot()
    bot_active.PB_modes = {"long": {symbol: "normal"}, "short": {}}
    bot_active.active_symbols = [symbol]
    with pytest.raises(RuntimeError, match="missing required m1_log_range EMA"):
        await pb_mod.Passivbot._load_orchestrator_ema_bundle(
            bot_active, [symbol], modes=bot_active.PB_modes
        )

    bot_with_position = FakeBot()
    bot_with_position.positions = {
        symbol: {"long": {"size": 1.0}, "short": {"size": 0.0}}
    }
    with pytest.raises(RuntimeError, match="missing required m1_log_range EMA"):
        await pb_mod.Passivbot._load_orchestrator_ema_bundle(
            bot_with_position, [symbol], modes=bot_with_position.PB_modes
        )


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_skips_cache_only_never_fetched_secondaries(
    monkeypatch,
    caplog,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def __init__(self):
            self.calls = []
            self.last_refresh = {
                "FETCH/USDT:USDT": 0,
                "SKIP/USDT:USDT": 0,
            }

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("qv", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_log_range(
            self, symbol, *, span, tf=None, max_age_ms=None, **kwargs
        ):
            self.calls.append(("lr", symbol, tf, int(max_age_ms)))
            return 1.0

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 4}}
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "ema_span_0":
                return 10.0
            if key == "ema_span_1":
                return 20.0
            if key == "entry_volatility_ema_span_hours":
                return 2.0
            return 0.0

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 5.0
            if key == "forager_volatility_ema_span_1m":
                return 7.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights()
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    with caplog.at_level(logging.DEBUG):
        (
            m1_close_emas,
            m1_volume_emas,
            m1_log_range_emas,
            h1_log_range_emas,
            volumes_long,
            log_ranges_long,
        ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(
            bot, ["POS/USDT:USDT", "FETCH/USDT:USDT", "SKIP/USDT:USDT"], modes={}
        )

    called_symbols = {symbol for _kind, symbol, _tf, _max_age_ms in bot.cm.calls}
    assert "POS/USDT:USDT" in called_symbols
    assert "FETCH/USDT:USDT" not in called_symbols
    assert "SKIP/USDT:USDT" not in called_symbols
    assert m1_close_emas["FETCH/USDT:USDT"] == {}
    assert m1_volume_emas["FETCH/USDT:USDT"] == {}
    assert m1_log_range_emas["FETCH/USDT:USDT"] == {}
    assert h1_log_range_emas["FETCH/USDT:USDT"] == {}
    assert volumes_long["FETCH/USDT:USDT"] == 0.0
    assert log_ranges_long["FETCH/USDT:USDT"] == 0.0
    assert m1_close_emas["SKIP/USDT:USDT"] == {}
    assert m1_volume_emas["SKIP/USDT:USDT"] == {}
    assert m1_log_range_emas["SKIP/USDT:USDT"] == {}
    assert h1_log_range_emas["SKIP/USDT:USDT"] == {}
    assert volumes_long["SKIP/USDT:USDT"] == 0.0
    assert log_ranges_long["SKIP/USDT:USDT"] == 0.0
    assert bot._orchestrator_ema_unavailable_symbols == {
        "FETCH/USDT:USDT",
        "SKIP/USDT:USDT",
    }
    assert any(
        "cache-only EMA skipped" in record.message
        and "unavailable=2" in record.message
        for record in caplog.records
    )
    assert not any(
        "skipping orchestrator EMA fetch for never-fetched cache-only symbol"
        in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_uses_cache_only_for_secondaries_without_open_slots(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def __init__(self):
            self.calls = []
            self.last_refresh = {
                "POS/USDT:USDT": now_ms - 10_000,
                "SECONDARY/USDT:USDT": now_ms - 10_000,
            }

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("qv", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_log_range(
            self, symbol, *, span, tf=None, max_age_ms=None, **kwargs
        ):
            self.calls.append(("lr", symbol, tf, int(max_age_ms)))
            return 1.0

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 30}}
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "ema_span_0":
                return 10.0
            if key == "ema_span_1":
                return 20.0
            if key == "entry_volatility_ema_span_hours":
                return 2.0
            return 0.0

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 5.0
            if key == "forager_volatility_ema_span_1m":
                return 7.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights()
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        get_symbols_with_pos = pb_mod.Passivbot.get_symbols_with_pos
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    await pb_mod.Passivbot._load_orchestrator_ema_bundle(
        bot, ["POS/USDT:USDT", "SECONDARY/USDT:USDT"], modes={}
    )

    cache_only_ttl = 365 * 24 * 3600 * 1000
    by_symbol = {}
    for kind, symbol, tf, max_age_ms in bot.cm.calls:
        by_symbol.setdefault(symbol, []).append((kind, tf, max_age_ms))

    assert all(
        max_age < cache_only_ttl for _kind, _tf, max_age in by_symbol["POS/USDT:USDT"]
    )
    assert all(
        max_age >= cache_only_ttl
        for _kind, _tf, max_age in by_symbol["SECONDARY/USDT:USDT"]
    )


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_disables_remote_fetch_for_cache_only_secondaries(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def __init__(self):
            self.calls = []
            self.last_refresh = {
                "POS/USDT:USDT": now_ms - 10_000,
                "SECONDARY/USDT:USDT": now_ms - 10_000,
            }

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            self.calls.append(("close", symbol, bool(allow_remote_fetch)))
            return 1.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, allow_remote_fetch=True, **kwargs
        ):
            self.calls.append(("qv", symbol, bool(allow_remote_fetch)))
            return 1.0

        async def get_latest_ema_log_range(
            self,
            symbol,
            *,
            span,
            tf=None,
            max_age_ms=None,
            allow_remote_fetch=True,
            **kwargs,
        ):
            self.calls.append(("lr", symbol, bool(allow_remote_fetch)))
            return 1.0

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 30}}
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "ema_span_0":
                return 10.0
            if key == "ema_span_1":
                return 20.0
            if key == "entry_volatility_ema_span_hours":
                return 2.0
            return 0.0

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 5.0
            if key == "forager_volatility_ema_span_1m":
                return 7.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights()
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        get_symbols_with_pos = pb_mod.Passivbot.get_symbols_with_pos
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    await pb_mod.Passivbot._load_orchestrator_ema_bundle(
        bot, ["POS/USDT:USDT", "SECONDARY/USDT:USDT"], modes={}
    )

    by_symbol = {}
    for kind, symbol, allow_remote_fetch in bot.cm.calls:
        by_symbol.setdefault(symbol, []).append((kind, allow_remote_fetch))

    assert by_symbol["POS/USDT:USDT"]
    assert by_symbol["SECONDARY/USDT:USDT"]
    assert all(allowed for _kind, allowed in by_symbol["POS/USDT:USDT"])
    assert not any(allowed for _kind, allowed in by_symbol["SECONDARY/USDT:USDT"])


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_marks_incomplete_cache_only_symbol_unavailable(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 2_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def __init__(self):
            self.calls = []
            self.last_refresh = {
                "FETCH/USDT:USDT": 1,
                "CACHE/USDT:USDT": now_ms - 10_000,
            }

        def get_last_refresh_ms(self, symbol):
            return int(self.last_refresh.get(symbol, 0))

        async def get_latest_ema_close(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(
            self, symbol, *, span, max_age_ms=None, **kwargs
        ):
            self.calls.append(("qv", symbol, None, int(max_age_ms)))
            if symbol == "CACHE/USDT:USDT":
                raise RuntimeError("no cached volume")
            return 1.0

        async def get_latest_ema_log_range(
            self, symbol, *, span, tf=None, max_age_ms=None, **kwargs
        ):
            self.calls.append(("lr", symbol, tf, int(max_age_ms)))
            return 1.0

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 4}}
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return True

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "ema_span_0":
                return 10.0
            if key == "ema_span_1":
                return 20.0
            if key == "entry_volatility_ema_span_hours":
                return 2.0
            return 0.0

        def bot_value(self, pside, key):
            if key == "forager_volume_ema_span_1m":
                return 5.0
            if key == "forager_volatility_ema_span_1m":
                return 7.0
            if key == "forager_volume_drop_pct":
                return 0.0
            if key == "forager_score_weights":
                return _forager_score_weights(volume=1.0 if pside == "long" else 0.0)
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = (
            pb_mod.Passivbot._rank_symbols_by_candle_staleness
        )
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    (
        m1_close_emas,
        m1_volume_emas,
        m1_log_range_emas,
        h1_log_range_emas,
        volumes_long,
        log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(
        bot, ["POS/USDT:USDT", "FETCH/USDT:USDT", "CACHE/USDT:USDT"], modes={}
    )

    called_symbols = {symbol for _kind, symbol, _tf, _max_age_ms in bot.cm.calls}
    assert "CACHE/USDT:USDT" in called_symbols
    assert m1_close_emas["CACHE/USDT:USDT"] == {}
    assert m1_volume_emas["CACHE/USDT:USDT"] == {}
    assert m1_log_range_emas["CACHE/USDT:USDT"] == {}
    assert h1_log_range_emas["CACHE/USDT:USDT"] == {}
    assert volumes_long["CACHE/USDT:USDT"] == 0.0
    assert log_ranges_long["CACHE/USDT:USDT"] == 0.0
    assert bot._orchestrator_ema_unavailable_symbols == {
        "CACHE/USDT:USDT",
        "FETCH/USDT:USDT",
    }


def test_required_candle_health_windows_include_indicator_and_diagnostic_timeframes():
    import passivbot as pb_mod

    class FakeCM:
        pass

    class FakeBot:
        config = {"live": {"warmup_ratio": 0.0}}
        cm = FakeCM()
        approved_coins_minus_ignored_coins = {
            "long": {"BTC/USDT:USDT"},
            "short": set(),
        }

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def is_forager_mode(self, pside=None):
            return pside == "long"

        def get_symbols_approved_or_has_pos(self, pside):
            return {"BTC/USDT:USDT"} if pside == "long" else set()

        def get_symbols_with_pos(self, pside):
            return {"BTC/USDT:USDT"} if pside == "long" else set()

        def bp(self, pside, key, symbol):
            values = {
                "ema_span_0": 10.0,
                "ema_span_1": 25.0,
                "forager_volume_ema_span_1m": 7.0,
                "forager_volatility_ema_span_1m": 11.0,
                "entry_volatility_ema_span_hours": 3.0,
            }
            return values.get(key, 0.0)

    windows = pb_mod.Passivbot._required_candle_windows_by_symbol(FakeBot())

    assert windows["BTC/USDT:USDT"]["1m"] == {"candles": 25, "required": True}
    assert windows["BTC/USDT:USDT"]["15m"] == {"candles": 1, "required": False}
    assert windows["BTC/USDT:USDT"]["1h"] == {"candles": 3, "required": True}


def test_forager_target_staleness_obeys_configured_cap():
    import passivbot as pb_mod

    class FakeBot:
        config = {"live": {"max_forager_candle_staleness_minutes": 10}}
        inactive_coin_candle_ttl_ms = 600_000

    bot = FakeBot()
    target_ms = pb_mod.Passivbot._forager_target_staleness_ms(bot, 100, 2)

    assert target_ms == 10 * 60_000


def test_startup_active_candle_symbols_include_positions_orders_and_active():
    import passivbot as pb_mod

    class FakeBot:
        active_symbols = ["ACTIVE/USDT:USDT"]
        positions = {
            "POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}},
            "FLAT/USDT:USDT": {"long": {"size": 0.0}, "short": {"size": 0.0}},
        }
        open_orders = {"OO/USDT:USDT": [{"id": "1"}], "EMPTY/USDT:USDT": []}

    assert pb_mod.Passivbot._startup_active_candle_symbols(FakeBot()) == [
        "ACTIVE/USDT:USDT",
        "OO/USDT:USDT",
        "POS/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_trading_ready_warmup_only_uses_startup_active_symbols(monkeypatch):
    import passivbot as pb_mod

    now_ms = 20_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeBot:
        active_symbols = ["ACTIVE/USDT:USDT"]
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {"OO/USDT:USDT": [{"id": "1"}]}

        def __init__(self):
            self.calls = []

        _startup_active_candle_symbols = pb_mod.Passivbot._startup_active_candle_symbols

        async def warmup_candles_staggered(self, **kwargs):
            self.calls.append(kwargs)

    bot = FakeBot()
    await pb_mod.Passivbot.warmup_trading_ready_candles(bot)

    assert bot.calls == [
        {
            "symbols_override": ["ACTIVE/USDT:USDT", "OO/USDT:USDT", "POS/USDT:USDT"],
            "skip_jitter": True,
            "context": "trading-ready warmup",
        }
    ]


@pytest.mark.asyncio
async def test_background_candle_warmup_is_scheduled_not_awaited():
    import passivbot as pb_mod

    class FakeBot:
        config = {"live": {}}

        def __init__(self):
            self.maintainers = {}
            self.started = False
            self.release = asyncio.Event()

        async def _background_candle_warmup_task(self):
            self.started = True
            await self.release.wait()

    import asyncio

    bot = FakeBot()
    await pb_mod.Passivbot.start_background_candle_warmup(bot)

    task = bot.maintainers.get("background_candle_warmup")
    assert task is not None
    await asyncio.sleep(0)
    assert bot.started is True
    assert not task.done()
    bot.release.set()
    await task


@pytest.mark.asyncio
async def test_forager_candidate_refresh_rotates_by_completed_candle_staleness(
    monkeypatch,
):
    import passivbot as pb_mod

    now_holder = {"now": 10_000_000}
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_holder["now"])

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []
            self.last_final = {
                "FRESH/USDT:USDT": now_holder["now"] - 60_000,
                "STALE/USDT:USDT": now_holder["now"] - 30 * 60_000,
                "OLDER/USDT:USDT": now_holder["now"] - 20 * 60_000,
            }

        def get_last_final_ts(self, symbol):
            return int(self.last_final.get(symbol, 0))

        def get_last_refresh_ms(self, symbol):
            return int(self.last_final.get(symbol, 0))

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            self.last_final[symbol] = int(kwargs["end_ts"])
            return []

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 1}}
        approved_coins_minus_ignored_coins = {
            "long": {"FRESH/USDT:USDT", "STALE/USDT:USDT", "OLDER/USDT:USDT"},
            "short": set(),
        }
        active_symbols = []
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        start_time_ms = 0
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "forager_volume_ema_span_1m":
                return 10.0
            if key == "forager_volatility_ema_span_1m":
                return 10.0
            return 0.0

        _forager_refresh_budget = pb_mod.Passivbot._forager_refresh_budget
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)
    now_holder["now"] += 60_000
    await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)

    assert [symbol for symbol, _kwargs in bot.cm.calls] == [
        "STALE/USDT:USDT",
        "OLDER/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_forager_candidate_refresh_skips_only_urgent_symbols(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []
            self.last_final = {
                "POS/USDT:USDT": now_ms - 30 * 60_000,
                "NORMAL/USDT:USDT": now_ms - 30 * 60_000,
                "GRACE/USDT:USDT": now_ms - 30 * 60_000,
                "STALE/USDT:USDT": now_ms - 20 * 60_000,
            }

        def get_last_final_ts(self, symbol):
            return int(self.last_final.get(symbol, 0))

        def get_last_refresh_ms(self, symbol):
            return int(self.last_final.get(symbol, 0))

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            return []

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 8}}
        approved_coins_minus_ignored_coins = {
            "long": {
                "POS/USDT:USDT",
                "NORMAL/USDT:USDT",
                "GRACE/USDT:USDT",
                "STALE/USDT:USDT",
            },
            "short": set(),
        }
        active_symbols = [
            "POS/USDT:USDT",
            "NORMAL/USDT:USDT",
            "GRACE/USDT:USDT",
            "STALE/USDT:USDT",
        ]
        PB_modes = {
            "long": {
                "NORMAL/USDT:USDT": "normal",
                "GRACE/USDT:USDT": "graceful_stop",
                "STALE/USDT:USDT": "graceful_stop",
            },
            "short": {},
        }
        positions = {"POS/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        start_time_ms = 0
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 2 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "forager_volume_ema_span_1m":
                return 10.0
            if key == "forager_volatility_ema_span_1m":
                return 10.0
            return 0.0

        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _forager_refresh_budget = pb_mod.Passivbot._forager_refresh_budget
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)

    assert [symbol for symbol, _kwargs in bot.cm.calls] == [
        "GRACE/USDT:USDT",
        "STALE/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_forager_candidate_refresh_caps_accumulated_budget(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10_000_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)
    symbols = [f"S{i}/USDT:USDT" for i in range(21)]

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []

        def get_last_final_ts(self, symbol):
            return now_ms - 60 * 60_000

        def get_last_refresh_ms(self, symbol):
            return now_ms - 60 * 60_000

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            return []

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 30}}
        approved_coins_minus_ignored_coins = {"long": set(symbols), "short": set()}
        active_symbols = []
        positions = {}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        start_time_ms = 0
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 4 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 4 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "forager_volume_ema_span_1m":
                return 10.0
            if key == "forager_volatility_ema_span_1m":
                return 10.0
            return 0.0

        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _forager_refresh_budget = pb_mod.Passivbot._forager_refresh_budget
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)

    assert len(bot.cm.calls) == 8


@pytest.mark.asyncio
async def test_forager_candidate_refresh_yields_after_wall_time_cap(
    monkeypatch, caplog
):
    import passivbot as pb_mod

    now_holder = {"now": 10_000_000}
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_holder["now"])
    symbols = [f"S{i}/USDT:USDT" for i in range(21)]

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []

        def get_last_final_ts(self, symbol):
            return now_holder["now"] - 60 * 60_000

        def get_last_refresh_ms(self, symbol):
            return now_holder["now"] - 60 * 60_000

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            now_holder["now"] += 31_000
            return []

    class FakeBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 30,
                "max_forager_candle_refresh_seconds": 45,
            }
        }
        approved_coins_minus_ignored_coins = {"long": set(symbols), "short": set()}
        active_symbols = []
        positions = {}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        start_time_ms = 0
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 4 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 4 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "forager_volume_ema_span_1m":
                return 10.0
            if key == "forager_volatility_ema_span_1m":
                return 10.0
            return 0.0

        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _forager_refresh_budget = pb_mod.Passivbot._forager_refresh_budget
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    with caplog.at_level("INFO"):
        await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)

    assert len(bot.cm.calls) == 2
    assert "forager refresh paused by wall-time cap" in caplog.text


@pytest.mark.asyncio
async def test_forager_candidate_refresh_sleep_respects_wall_time_cap(monkeypatch):
    import passivbot as pb_mod

    now_holder = {"now": 10_000_000}
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_holder["now"])
    symbols = [f"S{i}/USDT:USDT" for i in range(21)]
    sleep_calls = []

    async def fake_sleep(_bot, sleep_s, *, stage=""):
        sleep_calls.append((float(sleep_s), stage))
        now_holder["now"] += int(float(sleep_s) * 1000)

    monkeypatch.setattr(pb_mod.Passivbot, "_sleep_unless_shutdown", fake_sleep)

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []

        def get_last_final_ts(self, symbol):
            return now_holder["now"] - 60 * 60_000

        def get_last_refresh_ms(self, symbol):
            return now_holder["now"] - 60 * 60_000

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            now_holder["now"] += 10_000
            return []

    class FakeBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 30,
                "max_forager_candle_refresh_seconds": 15,
            }
        }
        approved_coins_minus_ignored_coins = {"long": set(symbols), "short": set()}
        active_symbols = []
        positions = {}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        start_time_ms = 0
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 4 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 4 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 10.0

        def bp(self, pside, key, symbol):
            if key == "forager_volume_ema_span_1m":
                return 10.0
            if key == "forager_volatility_ema_span_1m":
                return 10.0
            return 0.0

        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _forager_refresh_budget = pb_mod.Passivbot._forager_refresh_budget
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)

    assert len(bot.cm.calls) == 1
    assert sleep_calls == [(5.0, "forager_candidate_candle_refresh")]


@pytest.mark.asyncio
async def test_forager_candidate_refresh_skips_latest_final_candles(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10_000_000
    latest_final = (now_ms // 60_000) * 60_000 - 60_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        default_window_candles = 120

        def __init__(self):
            self.calls = []

        def get_last_final_ts(self, symbol):
            return latest_final

        def get_last_refresh_ms(self, symbol):
            return now_ms - 120_000

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            return []

    class FakeBot:
        config = {"live": {"max_ohlcv_fetches_per_minute": 30}}
        approved_coins_minus_ignored_coins = {
            "long": {"FRESH/USDT:USDT"},
            "short": set(),
        }
        active_symbols = []
        positions = {}
        open_orders = {}
        inactive_coin_candle_ttl_ms = 600_000
        stop_signal_received = False
        start_time_ms = 0
        cm = FakeCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def get_current_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def bp(self, pside, key, symbol):
            if key == "forager_volume_ema_span_1m":
                return 10.0
            if key == "forager_volatility_ema_span_1m":
                return 10.0
            return 0.0

        _urgent_active_candle_symbols = pb_mod.Passivbot._urgent_active_candle_symbols
        _forager_refresh_budget = pb_mod.Passivbot._forager_refresh_budget
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms

    bot = FakeBot()
    await pb_mod.Passivbot._refresh_forager_candidate_candles(bot)

    assert bot.cm.calls == []


def test_completed_candle_freshness_allows_bounded_active_tail_gap(monkeypatch, caplog):
    import logging
    import passivbot as pb_mod

    now_ms = 10 * 60_000
    latest_expected = 9 * 60_000
    last_cached = 6 * 60_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def get_completed_candle_health(self, symbol, windows, *, now_ms=None):
            assert windows == {"1m": 1}
            return {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "timeframe": "1m",
                        "coverage_ok": False,
                        "latest_expected_ts": latest_expected,
                        "last_cached_ts": last_cached,
                        "missing_candles": 3,
                        "missing_spans": [(last_cached + 60_000, latest_expected)],
                        "open_tail_gap": True,
                        "tail_gap_candles": 3,
                        "tail_gap_age_ms": latest_expected - last_cached,
                    }
                },
            }

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.config = {"live": {"max_active_candle_tail_gap_minutes": 10}}
    bot.cm = FakeCM()

    with caplog.at_level(logging.INFO):
        signature, missing = pb_mod.Passivbot._completed_candle_freshness_signature(
            bot, ["TAIL/USDT:USDT"], now_ms=now_ms
        )

    assert missing == []
    assert signature == (
        (
            "TAIL/USDT:USDT",
            latest_expected,
            "tail_gap_fallback",
            last_cached,
            latest_expected - last_cached,
        ),
    )
    assert any(
        "active tail gap EMA projection" in record.message
        and record.levelno == logging.INFO
        for record in caplog.records
    )


def test_completed_candle_tail_gap_fallback_repeats_at_debug(monkeypatch, caplog):
    import passivbot as pb_mod

    now_ms = 10 * 60_000
    current_now = {"ms": now_ms}
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: current_now["ms"])

    class FakeCM:
        def get_completed_candle_health(self, symbol, windows, *, now_ms=None):
            latest_expected = int(now_ms) - 60_000
            last_cached = latest_expected - 60_000
            return {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "timeframe": "1m",
                        "coverage_ok": False,
                        "latest_expected_ts": latest_expected,
                        "last_cached_ts": last_cached,
                        "missing_candles": 1,
                        "missing_spans": [(last_cached + 60_000, latest_expected)],
                        "open_tail_gap": True,
                        "tail_gap_candles": 1,
                        "tail_gap_age_ms": latest_expected - last_cached,
                    }
                },
            }

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.config = {"live": {"max_active_candle_tail_gap_minutes": 10}}
    bot.cm = FakeCM()

    with caplog.at_level(logging.DEBUG):
        pb_mod.Passivbot._completed_candle_freshness_signature(
            bot, ["TAIL/USDT:USDT"], now_ms=current_now["ms"]
        )
        current_now["ms"] += 60_000
        pb_mod.Passivbot._completed_candle_freshness_signature(
            bot, ["TAIL/USDT:USDT"], now_ms=current_now["ms"]
        )

    records = [r for r in caplog.records if "active tail gap EMA projection" in r.message]
    assert [r.levelno for r in records] == [logging.DEBUG, logging.DEBUG]


def test_completed_candle_tail_gap_fallback_warns_near_cap(monkeypatch, caplog):
    import logging
    import passivbot as pb_mod

    now_ms = 10 * 60_000
    latest_expected = 9 * 60_000
    last_cached = 1 * 60_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def get_completed_candle_health(self, symbol, windows, *, now_ms=None):
            return {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "timeframe": "1m",
                        "coverage_ok": False,
                        "latest_expected_ts": latest_expected,
                        "last_cached_ts": last_cached,
                        "missing_candles": 8,
                        "missing_spans": [(last_cached + 60_000, latest_expected)],
                        "open_tail_gap": True,
                        "tail_gap_candles": 8,
                        "tail_gap_age_ms": latest_expected - last_cached,
                    }
                },
            }

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.config = {"live": {"max_active_candle_tail_gap_minutes": 10}}
    bot.cm = FakeCM()

    with caplog.at_level(logging.WARNING):
        signature, missing = pb_mod.Passivbot._completed_candle_freshness_signature(
            bot, ["TAIL/USDT:USDT"], now_ms=now_ms
        )

    assert missing == []
    assert signature[0][2] == "tail_gap_fallback"
    assert any(
        "active tail gap EMA projection" in record.message
        and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_completed_candle_freshness_allows_real_cm_bounded_active_tail_gap(
    monkeypatch, tmp_path
):
    import numpy as np
    import passivbot as pb_mod
    from candlestick_manager import CANDLE_DTYPE, CandlestickManager

    now_ms = 10 * 60_000
    latest_expected = 9 * 60_000
    last_cached = 6 * 60_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    cm = CandlestickManager(
        exchange=None,
        exchange_name="testex",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._persist_batch(
        "TAIL/USDT:USDT",
        np.array([(last_cached, 100.0, 100.0, 100.0, 100.0, 1.0)], dtype=CANDLE_DTYPE),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.config = {"live": {"max_active_candle_tail_gap_minutes": 10}}
    bot.cm = cm

    signature, missing = pb_mod.Passivbot._completed_candle_freshness_signature(
        bot, ["TAIL/USDT:USDT"], now_ms=now_ms
    )

    assert missing == []
    assert signature == (
        (
            "TAIL/USDT:USDT",
            latest_expected,
            "tail_gap_fallback",
            last_cached,
            latest_expected - last_cached,
        ),
    )


def test_completed_candle_freshness_blocks_excessive_active_tail_gap(monkeypatch):
    import passivbot as pb_mod

    now_ms = 20 * 60_000
    latest_expected = 19 * 60_000
    last_cached = 8 * 60_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def get_completed_candle_health(self, symbol, windows, *, now_ms=None):
            assert windows == {"1m": 1}
            return {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "timeframe": "1m",
                        "coverage_ok": False,
                        "latest_expected_ts": latest_expected,
                        "last_cached_ts": last_cached,
                        "missing_candles": 11,
                        "missing_spans": [(last_cached + 60_000, latest_expected)],
                        "open_tail_gap": True,
                        "tail_gap_candles": 11,
                        "tail_gap_age_ms": latest_expected - last_cached,
                    }
                },
            }

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.config = {"live": {"max_active_candle_tail_gap_minutes": 10}}
    bot.cm = FakeCM()

    signature, missing = pb_mod.Passivbot._completed_candle_freshness_signature(
        bot, ["TAIL/USDT:USDT"], now_ms=now_ms
    )

    assert signature == ()
    assert missing[0]["reason"] == "active_candle_tail_gap_exceeded"
    assert missing[0]["tail_gap_age_ms"] == latest_expected - last_cached


def test_completed_candle_freshness_blocks_bounded_gap_plus_tail(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10 * 60_000
    latest_expected = 9 * 60_000
    last_cached = 6 * 60_000
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    class FakeCM:
        def get_completed_candle_health(self, symbol, windows, *, now_ms=None):
            assert windows == {"1m": 1}
            return {
                "ok": False,
                "timeframes": {
                    "1m": {
                        "timeframe": "1m",
                        "coverage_ok": False,
                        "latest_expected_ts": latest_expected,
                        "last_cached_ts": last_cached,
                        "missing_candles": 4,
                        "missing_spans": [
                            (2 * 60_000, 2 * 60_000),
                            (last_cached + 60_000, latest_expected),
                        ],
                        "open_tail_gap": True,
                        "tail_gap_candles": 3,
                        "tail_gap_age_ms": latest_expected - last_cached,
                    }
                },
            }

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.config = {"live": {"max_active_candle_tail_gap_minutes": 10}}
    bot.cm = FakeCM()

    signature, missing = pb_mod.Passivbot._completed_candle_freshness_signature(
        bot, ["TAIL/USDT:USDT"], now_ms=now_ms
    )

    assert signature == ()
    assert missing[0]["reason"] == "missing_latest_completed_1m"


@pytest.mark.asyncio
async def test_projected_open_tail_ema_metrics_are_read_only(tmp_path, monkeypatch):
    import math
    import numpy as np
    from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS

    cm = CandlestickManager(exchange=None, exchange_name="testex", cache_dir=str(tmp_path / "caches"))
    symbol = "TAIL/USDT:USDT"
    last_cached = 60 * ONE_MIN_MS
    latest_expected = last_cached + 3 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: (latest_expected + ONE_MIN_MS) / 1000.0)
    seed = np.array(
        [(last_cached, 100.0, 101.0, 99.0, 100.0, 2.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._cache[symbol] = seed.copy()
    cm._ema_cache[symbol] = {("sentinel", 1.0, str(ONE_MIN_MS)): (123.0, last_cached, last_cached)}

    projected = await cm.get_projected_open_tail_ema_metrics(
        symbol,
        {"close": [4.0], "qv": [4.0], "log_range": [4.0]},
        latest_expected_ts=latest_expected,
        last_cached_ts=last_cached,
        max_tail_gap_ms=10 * ONE_MIN_MS,
    )

    projected_rows = np.array(
        [
            (last_cached, 100.0, 101.0, 99.0, 100.0, 2.0),
            (last_cached + ONE_MIN_MS, 100.0, 100.0, 100.0, 100.0, 0.0),
            (last_cached + 2 * ONE_MIN_MS, 100.0, 100.0, 100.0, 100.0, 0.0),
            (latest_expected, 100.0, 100.0, 100.0, 100.0, 0.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    assert projected["close"][4.0] == pytest.approx(100.0)
    assert projected["qv"][4.0] == pytest.approx(
        cm._ema(cm._ema_metric_series("qv", projected_rows), 4.0)
    )
    assert projected["log_range"][4.0] == pytest.approx(
        cm._ema(cm._ema_metric_series("log_range", projected_rows), 4.0)
    )
    assert np.array_equal(cm._cache[symbol], seed)
    assert cm._synthetic_timestamps.get(symbol, set()) == set()
    assert cm._ema_cache[symbol] == {
        ("sentinel", 1.0, str(ONE_MIN_MS)): (123.0, last_cached, last_cached)
    }
    assert math.isfinite(projected["qv"][4.0])


@pytest.mark.asyncio
async def test_projected_open_tail_ema_recomputes_when_late_real_candles_arrive(
    tmp_path, monkeypatch
):
    import numpy as np
    from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS

    cm = CandlestickManager(exchange=None, exchange_name="testex", cache_dir=str(tmp_path / "caches"))
    symbol = "LATE/USDT:USDT"
    t58 = 58 * ONE_MIN_MS
    t59 = 59 * ONE_MIN_MS
    t60 = 60 * ONE_MIN_MS
    t61 = 61 * ONE_MIN_MS

    monkeypatch.setattr("time.time", lambda: (t60 + ONE_MIN_MS) / 1000.0)
    cm._cache[symbol] = np.array([(t58, 100.0, 100.0, 100.0, 100.0, 1.0)], dtype=CANDLE_DTYPE)
    first_projection = await cm.get_projected_open_tail_ema_metrics(
        symbol,
        {"close": [4.0]},
        latest_expected_ts=t60,
        last_cached_ts=t58,
        max_tail_gap_ms=10 * ONE_MIN_MS,
    )
    assert first_projection["close"][4.0] == pytest.approx(100.0)
    assert cm._ema_cache.get(symbol, {}) == {}

    real_late = np.array(
        [
            (t59, 110.0, 110.0, 110.0, 110.0, 3.0),
            (t61, 120.0, 120.0, 120.0, 120.0, 4.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, real_late, timeframe="1m", merge_cache=True, last_refresh_ms=t61)
    monkeypatch.setattr("time.time", lambda: (t61 + ONE_MIN_MS) / 1000.0)

    ema = await cm.get_latest_ema_close(symbol, 4.0, allow_remote_fetch=False)
    expected_rows = np.array(
        [
            (t58, 100.0, 100.0, 100.0, 100.0, 1.0),
            (t59, 110.0, 110.0, 110.0, 110.0, 3.0),
            (t60, 110.0, 110.0, 110.0, 110.0, 0.0),
            (t61, 120.0, 120.0, 120.0, 120.0, 4.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    assert ema == pytest.approx(cm._ema(expected_rows["c"].astype(float), 4.0))
    assert ema != pytest.approx(first_projection["close"][4.0])
    assert t60 in cm._synthetic_timestamps.get(symbol, set())
    assert t59 not in cm._synthetic_timestamps.get(symbol, set())
    assert t61 not in cm._synthetic_timestamps.get(symbol, set())


@pytest.mark.asyncio
async def test_projected_open_tail_ema_seeds_when_tail_exceeds_span(tmp_path, monkeypatch):
    import numpy as np
    from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS

    cm = CandlestickManager(exchange=None, exchange_name="testex", cache_dir=str(tmp_path / "caches"))
    symbol = "SHORTSPAN/USDT:USDT"
    last_cached = 60 * ONE_MIN_MS
    latest_expected = last_cached + 3 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: (latest_expected + ONE_MIN_MS) / 1000.0)
    seed = np.array(
        [(last_cached, 100.0, 101.0, 99.0, 100.0, 2.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._cache[symbol] = seed.copy()

    projected = await cm.get_projected_open_tail_ema_metrics(
        symbol,
        {"close": [2.0], "qv": [2.0], "log_range": [2.0]},
        latest_expected_ts=latest_expected,
        last_cached_ts=last_cached,
        max_tail_gap_ms=10 * ONE_MIN_MS,
    )

    assert projected["close"][2.0] == pytest.approx(100.0)
    assert projected["qv"][2.0] == pytest.approx(0.0)
    assert projected["log_range"][2.0] == pytest.approx(0.0)
    assert np.array_equal(cm._cache[symbol], seed)
    assert cm._ema_cache.get(symbol, {}) == {}
    assert cm._synthetic_timestamps.get(symbol, set()) == set()


@pytest.mark.asyncio
async def test_projected_open_tail_ema_refuses_tail_beyond_threshold(tmp_path, monkeypatch):
    import numpy as np
    from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS

    cm = CandlestickManager(exchange=None, exchange_name="testex", cache_dir=str(tmp_path / "caches"))
    symbol = "STALE/USDT:USDT"
    last_cached = 10 * ONE_MIN_MS
    latest_expected = last_cached + 11 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: (latest_expected + ONE_MIN_MS) / 1000.0)
    cm._cache[symbol] = np.array(
        [(last_cached, 100.0, 100.0, 100.0, 100.0, 1.0)],
        dtype=CANDLE_DTYPE,
    )

    with pytest.raises(ValueError, match="exceeds max_tail_gap_ms"):
        await cm.get_projected_open_tail_ema_metrics(
            symbol,
            {"close": [4.0]},
            latest_expected_ts=latest_expected,
            last_cached_ts=last_cached,
            max_tail_gap_ms=10 * ONE_MIN_MS,
        )
