import pytest

from passivbot import Passivbot


class CoinFilterHarness(Passivbot):
    def __init__(
        self,
        *,
        approved_by_side,
        volumes,
        log_ranges,
        ema_readiness=None,
        max_positions,
        clip_pct=0.0,
        score_weights=None,
        forager=True,
        min_cost_ok=None,
        age_ok=None,
        filter_min_cost_flag=True,
    ):
        self.PB_modes = {"long": {}, "short": {}}
        self.approved_coins_minus_ignored_coins = approved_by_side
        self._volumes = volumes
        self._log_ranges = log_ranges
        self._ema_readiness = ema_readiness or {}
        self._max_positions = max_positions
        self._clip_pct = clip_pct
        self._score_weights = score_weights or {
            "volume": 0.0,
            "ema_readiness": 0.0,
            "volatility": 1.0,
        }
        self._forager = forager
        self._min_cost_ok = min_cost_ok or {}
        self._age_ok = age_ok or {}
        self._live_flags = {"filter_by_min_effective_cost": filter_min_cost_flag}
        self.warn_triggered = False
        self.positions = {}
        self.config = {"live": {}}

    # ---- overrides for get_filtered_coins dependencies ----

    def get_forced_PB_mode(self, pside, symbol=None):
        return None

    def is_old_enough(self, pside, symbol):
        return self._age_ok.get(symbol, True)

    def effective_min_cost_is_low_enough(self, pside, symbol):
        return self._min_cost_ok.get(symbol, True)

    def warn_on_high_effective_min_cost(self, _pside):
        self.warn_triggered = True

    def is_forager_mode(self, _pside):
        return self._forager

    def bot_value(self, pside, key):
        if key == "forager_volume_drop_pct":
            return self._clip_pct
        if key == "forager_score_weights":
            return self._score_weights
        if key == "n_positions":
            return float(self._max_positions[pside])
        return 0.0

    def bp(self, pside, key, symbol=None):
        if key == "filter_volume_ema_span":
            return 12.0
        if key == "filter_volatility_ema_span":
            return 12.0
        if key == "ema_span_0":
            return 10.0
        if key == "ema_span_1":
            return 20.0
        if key == "entry_initial_ema_dist":
            return 0.1
        return self.bot_value(pside, key)

    def get_max_n_positions(self, pside):
        return self._max_positions[pside]

    def live_value(self, key):
        return self._live_flags.get(key, False)

    async def calc_volumes(self, _pside, symbols):
        return {sym: self._volumes[sym] for sym in symbols}

    async def calc_log_range(
        self, _pside, eligible_symbols, max_age_ms=None, max_network_fetches=None
    ):
        return {sym: self._log_ranges[sym] for sym in eligible_symbols}

    async def calc_volumes_and_log_ranges(
        self, _pside, symbols=None, *, max_age_ms=None, max_network_fetches=None
    ):
        return (
            {sym: self._volumes[sym] for sym in symbols},
            {sym: self._log_ranges[sym] for sym in symbols},
        )

    async def build_forager_candidate_payload(
        self,
        pside,
        symbols,
        min_cost_flags,
        *,
        max_age_ms=None,
        max_network_fetches=None,
    ):
        payload = []
        entry_initial_ema_dist = float(self.bp(pside, "entry_initial_ema_dist"))
        for idx, sym in enumerate(symbols):
            readiness = float(self._ema_readiness.get(sym, 0.0))
            if pside == "long":
                ema_lower = 100.0
                bid = ema_lower * (1.0 - entry_initial_ema_dist) * (1.0 + readiness)
                ask = bid
                ema_upper = 110.0
            else:
                ema_upper = 100.0
                ask = ema_upper * (1.0 + entry_initial_ema_dist) * (1.0 - readiness)
                bid = ask
                ema_lower = 90.0
            payload.append(
                {
                    "index": idx,
                    "enabled": bool(min_cost_flags[sym]),
                    "volume_score": self._volumes[sym],
                    "volatility_score": self._log_ranges[sym],
                    "bid": bid,
                    "ask": ask,
                    "ema_lower": ema_lower,
                    "ema_upper": ema_upper,
                    "entry_initial_ema_dist": entry_initial_ema_dist,
                }
            )
        return payload

    def is_pside_enabled(self, _pside):
        return True


@pytest.mark.asyncio
async def test_forager_prefers_highest_volatility():
    approved = {"long": ["AAA", "BBB", "CCC"], "short": []}
    volumes = {"AAA": 10.0, "BBB": 8.0, "CCC": 3.0}
    log_ranges = {"AAA": 0.2, "BBB": 0.6, "CCC": 0.4}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 2, "short": 0},
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == ["BBB", "CCC"]


@pytest.mark.asyncio
async def test_volume_clip_applied_before_volatility_sort():
    approved = {"long": ["AAA", "BBB", "CCC", "DDD"], "short": []}
    volumes = {"AAA": 100.0, "BBB": 80.0, "CCC": 10.0, "DDD": 5.0}
    log_ranges = {"AAA": 0.3, "BBB": 0.2, "CCC": 0.9, "DDD": 0.8}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 2, "short": 0},
        clip_pct=0.5,
    )
    # clip removes bottom 50% by volume -> keep AAA,BBB
    coins = await bot.get_filtered_coins("long")
    assert coins == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_ema_readiness_can_override_volatility_when_weighted():
    approved = {"long": ["AAA", "BBB", "CCC", "DDD"], "short": []}
    volumes = {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0, "DDD": 10.0}
    log_ranges = {"AAA": 0.9, "BBB": 0.8, "CCC": 0.7, "DDD": 0.6}
    ema_readiness = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": -0.01}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        ema_readiness=ema_readiness,
        max_positions={"long": 2, "short": 0},
        score_weights={"volume": 0.0, "ema_readiness": 1.0, "volatility": 0.0},
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == ["DDD", "CCC"]


@pytest.mark.asyncio
async def test_min_cost_filter_warns_when_all_removed():
    approved = {"long": ["AAA", "BBB"], "short": []}
    volumes = {"AAA": 1.0, "BBB": 1.0}
    log_ranges = {"AAA": 0.1, "BBB": 0.2}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 1, "short": 0},
        min_cost_ok={"AAA": False, "BBB": False},
        filter_min_cost_flag=True,
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == []
    assert bot.warn_triggered is True


@pytest.mark.asyncio
async def test_non_forager_returns_sorted_candidates():
    approved = {"long": ["CCC", "AAA", "BBB"], "short": []}
    volumes = {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}
    log_ranges = {"AAA": 0.1, "BBB": 0.2, "CCC": 0.3}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 3, "short": 0},
        forager=False,
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == ["AAA", "BBB", "CCC"]


def test_split_forager_budget_by_side_round_robins_remainder():
    bot = Passivbot.__new__(Passivbot)
    first = bot._split_forager_budget_by_side(1, ["long", "short"])
    second = bot._split_forager_budget_by_side(1, ["long", "short"])
    assert first in ({"long": 1, "short": 0}, {"long": 0, "short": 1})
    assert second in ({"long": 1, "short": 0}, {"long": 0, "short": 1})
    assert first != second


class _CMColdCacheOnlyStub:
    def __init__(self):
        self.calls = 0

    def get_last_refresh_ms(self, _symbol):
        return 0

    async def get_latest_ema_metrics(self, _symbol, spans_by_metric, **_kwargs):
        self.calls += 1
        return {k: 1.0 for k in spans_by_metric}


class _ForagerCMStub:
    def __init__(self, *, last_refresh_ms=None, metrics_by_symbol=None, bounds_by_symbol=None):
        self.last_refresh_ms = last_refresh_ms or {}
        self.metrics_by_symbol = metrics_by_symbol or {}
        self.bounds_by_symbol = bounds_by_symbol or {}
        self.metric_calls = []
        self.bounds_calls = []
        self.close_calls = []

    def get_last_refresh_ms(self, symbol):
        return self.last_refresh_ms.get(symbol, 0)

    async def get_latest_ema_metrics(self, symbol, spans_by_metric, **_kwargs):
        self.metric_calls.append((symbol, dict(spans_by_metric)))
        return self.metrics_by_symbol[symbol]

    async def get_ema_bounds(self, symbol, span_0, span_1, **_kwargs):
        self.bounds_calls.append((symbol, span_0, span_1))
        return self.bounds_by_symbol[symbol]

    async def get_current_close(self, symbol, max_age_ms=None):
        self.close_calls.append((symbol, max_age_ms))
        return 100.0


@pytest.mark.asyncio
async def test_calc_log_range_respects_cache_only_budget_for_cold_symbols():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _CMColdCacheOnlyStub()
    bot.open_orders = {}
    bot.positions = {}
    bot.bot_value = lambda _pside, key: (
        12.0 if key in ("filter_volatility_ema_span", "filter_volume_ema_span") else 0.0
    )
    bot.has_position = lambda *_args, **_kwargs: False
    out = await bot.calc_log_range(
        "long",
        eligible_symbols=["AAA", "BBB", "CCC"],
        max_age_ms=60_000,
        max_network_fetches=0,
    )
    assert out == {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}
    assert bot.cm.calls == 0


@pytest.mark.asyncio
async def test_calc_volumes_and_log_ranges_respects_cache_only_budget_for_cold_symbols():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _CMColdCacheOnlyStub()
    bot.open_orders = {}
    bot.positions = {}
    bot.bot_value = lambda _pside, key: (
        12.0 if key in ("filter_volatility_ema_span", "filter_volume_ema_span") else 0.0
    )
    bot.has_position = lambda *_args, **_kwargs: False
    volumes, log_ranges = await bot.calc_volumes_and_log_ranges(
        "long",
        symbols=["AAA", "BBB", "CCC"],
        max_age_ms=60_000,
        max_network_fetches=0,
    )
    assert volumes == {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}
    assert log_ranges == {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}
    assert bot.cm.calls == 0


@pytest.mark.asyncio
async def test_build_forager_candidate_payload_fails_for_cold_cache_only_symbol():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _ForagerCMStub(last_refresh_ms={"AAA": 0})
    bot.open_orders = {}
    bot.positions = {}
    bot.inactive_coin_candle_ttl_ms = 600_000
    bot.bot_value = lambda _pside, key: (
        {"volume": 1.0, "ema_readiness": 0.0, "volatility": 0.0}
        if key == "forager_score_weights"
        else 0.0
    )
    bot.bp = lambda _pside, key, symbol=None: 12.0
    bot.has_position = lambda pside=None, symbol=None: False
    with pytest.raises(RuntimeError, match="cannot refresh cold symbol AAA"):
        await bot.build_forager_candidate_payload(
            "long",
            ["AAA"],
            {"AAA": True},
            max_age_ms=60_000,
            max_network_fetches=0,
        )


@pytest.mark.asyncio
async def test_build_forager_candidate_payload_skips_ema_inputs_when_weight_zero():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _ForagerCMStub(
        last_refresh_ms={"AAA": 123},
        metrics_by_symbol={"AAA": {"qv": 7.0}},
        bounds_by_symbol={"AAA": (90.0, 110.0)},
    )
    bot.open_orders = {}
    bot.positions = {}
    bot.inactive_coin_candle_ttl_ms = 600_000
    bot.bot_value = lambda _pside, key: (
        {"volume": 1.0, "ema_readiness": 0.0, "volatility": 0.0}
        if key == "forager_score_weights"
        else 0.0
    )
    bot.bp = lambda _pside, key, symbol=None: 12.0
    bot.has_position = lambda pside=None, symbol=None: False
    payload = await bot.build_forager_candidate_payload(
        "long",
        ["AAA"],
        {"AAA": True},
        max_age_ms=60_000,
        max_network_fetches=1,
    )
    assert payload == [
        {
            "index": 0,
            "enabled": True,
            "volume_score": 7.0,
            "volatility_score": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "ema_lower": 0.0,
            "ema_upper": 0.0,
            "entry_initial_ema_dist": 0.0,
        }
    ]
    assert bot.cm.bounds_calls == []
    assert bot.cm.close_calls == []
