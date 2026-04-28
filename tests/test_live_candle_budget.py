import pytest


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
        _rank_symbols_by_candle_staleness = pb_mod.Passivbot._rank_symbols_by_candle_staleness
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
        _rank_symbols_by_candle_staleness = pb_mod.Passivbot._rank_symbols_by_candle_staleness
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    await pb_mod.Passivbot.update_ohlcvs_1m_for_actives(bot)

    called = [symbol for symbol, _kwargs in bot.cm.calls]
    assert called == ["POS/USDT:USDT"]


@pytest.mark.asyncio
async def test_execute_to_exchange_refreshes_forager_candidates_after_planning(monkeypatch):
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

        async def _refresh_forager_candidate_candles(self):
            self.events.append("forager_refresh")

    bot = FakeBot()
    assert await pb_mod.Passivbot.execute_to_exchange(bot) == ([], [])
    assert bot.events == ["execution_cycle", "plan", "forager_refresh"]


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
async def test_orchestrator_ema_bundle_budgets_forager_only_symbols(monkeypatch):
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

        async def get_latest_ema_close(self, symbol, *, span, max_age_ms=None, **kwargs):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(self, symbol, *, span, max_age_ms=None, **kwargs):
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
            if key == "forager_volume_ema_span":
                return 5.0
            if key == "forager_volatility_ema_span":
                return 7.0
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = pb_mod.Passivbot._rank_symbols_by_candle_staleness
        _forager_target_staleness_ms = pb_mod.Passivbot._forager_target_staleness_ms
        _token_bucket_budget = pb_mod.Passivbot._token_bucket_budget

    bot = FakeBot()
    symbols = ["POS/USDT:USDT", "A/USDT:USDT", "B/USDT:USDT"]
    await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, symbols, modes={})

    by_symbol = {}
    for kind, symbol, tf, max_age_ms in bot.cm.calls:
        by_symbol.setdefault(symbol, []).append((kind, tf, max_age_ms))

    assert all(max_age <= 600_000 for _kind, _tf, max_age in by_symbol["POS/USDT:USDT"])
    assert all(max_age <= 600_000 for _kind, _tf, max_age in by_symbol["A/USDT:USDT"])
    assert "B/USDT:USDT" in by_symbol
    assert all(
        max_age >= 365 * 24 * 3600 * 1000
        for _kind, _tf, max_age in by_symbol["B/USDT:USDT"]
    )


@pytest.mark.asyncio
async def test_orchestrator_ema_bundle_skips_cache_only_never_fetched_secondaries(monkeypatch):
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

        async def get_latest_ema_close(self, symbol, *, span, max_age_ms=None, **kwargs):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(self, symbol, *, span, max_age_ms=None, **kwargs):
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
            if key == "forager_volume_ema_span":
                return 5.0
            if key == "forager_volatility_ema_span":
                return 7.0
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = pb_mod.Passivbot._rank_symbols_by_candle_staleness
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
        bot, ["POS/USDT:USDT", "FETCH/USDT:USDT", "SKIP/USDT:USDT"], modes={}
    )

    called_symbols = {symbol for _kind, symbol, _tf, _max_age_ms in bot.cm.calls}
    assert "POS/USDT:USDT" in called_symbols
    assert "FETCH/USDT:USDT" in called_symbols
    assert "SKIP/USDT:USDT" not in called_symbols
    assert m1_close_emas["SKIP/USDT:USDT"] == {}
    assert m1_volume_emas["SKIP/USDT:USDT"] == {}
    assert m1_log_range_emas["SKIP/USDT:USDT"] == {}
    assert h1_log_range_emas["SKIP/USDT:USDT"] == {}
    assert volumes_long["SKIP/USDT:USDT"] == 0.0
    assert log_ranges_long["SKIP/USDT:USDT"] == 0.0
    assert bot._orchestrator_ema_unavailable_symbols == {"SKIP/USDT:USDT"}


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

        async def get_latest_ema_close(self, symbol, *, span, max_age_ms=None, **kwargs):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(self, symbol, *, span, max_age_ms=None, **kwargs):
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
            if key == "forager_volume_ema_span":
                return 5.0
            if key == "forager_volatility_ema_span":
                return 7.0
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        get_symbols_with_pos = pb_mod.Passivbot.get_symbols_with_pos
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = pb_mod.Passivbot._rank_symbols_by_candle_staleness
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

    assert all(max_age < cache_only_ttl for _kind, _tf, max_age in by_symbol["POS/USDT:USDT"])
    assert all(
        max_age >= cache_only_ttl
        for _kind, _tf, max_age in by_symbol["SECONDARY/USDT:USDT"]
    )


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

        async def get_latest_ema_close(self, symbol, *, span, max_age_ms=None, **kwargs):
            self.calls.append(("close", symbol, None, int(max_age_ms)))
            return 1.0

        async def get_latest_ema_quote_volume(self, symbol, *, span, max_age_ms=None, **kwargs):
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
            if key == "forager_volume_ema_span":
                return 5.0
            if key == "forager_volatility_ema_span":
                return 7.0
            return 0.0

        has_position = pb_mod.Passivbot.has_position
        _compute_fetch_budget_ttls = pb_mod.Passivbot._compute_fetch_budget_ttls
        _candle_staleness_ms = pb_mod.Passivbot._candle_staleness_ms
        _rank_symbols_by_candle_staleness = pb_mod.Passivbot._rank_symbols_by_candle_staleness
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
    assert bot._orchestrator_ema_unavailable_symbols == {"CACHE/USDT:USDT"}


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
                "forager_volume_ema_span": 7.0,
                "forager_volatility_ema_span": 11.0,
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
async def test_forager_candidate_refresh_rotates_by_completed_candle_staleness(monkeypatch):
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
            if key == "forager_volume_ema_span":
                return 10.0
            if key == "forager_volatility_ema_span":
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
            if key == "forager_volume_ema_span":
                return 10.0
            if key == "forager_volatility_ema_span":
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
            if key == "forager_volume_ema_span":
                return 10.0
            if key == "forager_volatility_ema_span":
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
        approved_coins_minus_ignored_coins = {"long": {"FRESH/USDT:USDT"}, "short": set()}
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
            if key == "forager_volume_ema_span":
                return 10.0
            if key == "forager_volatility_ema_span":
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
