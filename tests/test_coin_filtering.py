import pytest

from passivbot import Passivbot


class CoinFilterHarness(Passivbot):
    def __init__(
        self,
        *,
        approved_by_side,
        volumes,
        log_ranges,
        max_positions,
        clip_pct=0.0,
        volatility_drop_pct=0.0,
        forager=True,
        min_cost_ok=None,
        age_ok=None,
        filter_min_cost_flag=True,
    ):
        self.PB_modes = {"long": {}, "short": {}}
        self.approved_coins_minus_ignored_coins = approved_by_side
        self._volumes = volumes
        self._log_ranges = log_ranges
        self._max_positions = max_positions
        self._clip_pct = clip_pct
        self._volatility_drop_pct = volatility_drop_pct
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
        if key == "filter_volume_drop_pct":
            return self._clip_pct
        if key == "filter_volatility_drop_pct":
            return self._volatility_drop_pct
        if key == "n_positions":
            return float(self._max_positions[pside])
        return 0.0

    def get_max_n_positions(self, pside):
        return self._max_positions[pside]

    def live_value(self, key):
        return self._live_flags.get(key, False)

    async def calc_volumes(self, _pside, symbols):
        return {sym: self._volumes[sym] for sym in symbols}

    async def calc_log_range(self, _pside, eligible_symbols, max_age_ms=None, max_network_fetches=None):
        return {sym: self._log_ranges[sym] for sym in eligible_symbols}

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
async def test_volatility_drop_pct_discards_top_tail():
    approved = {"long": ["AAA", "BBB", "CCC", "DDD"], "short": []}
    volumes = {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0, "DDD": 10.0}
    log_ranges = {"AAA": 0.9, "BBB": 0.8, "CCC": 0.7, "DDD": 0.6}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 2, "short": 0},
        volatility_drop_pct=0.5,
    )
    coins = await bot.get_filtered_coins("long")
    # drops top 50% => keep CCC,DDD and pick highest among them => [CCC, DDD]
    assert coins == ["CCC", "DDD"]


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


@pytest.mark.asyncio
async def test_calc_log_range_respects_cache_only_budget_for_cold_symbols():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _CMColdCacheOnlyStub()
    bot.open_orders = {}
    bot.positions = {}
    bot.bot_value = (
        lambda _pside, key: 12.0
        if key in ("filter_volatility_ema_span", "filter_volume_ema_span")
        else 0.0
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
    bot.bot_value = (
        lambda _pside, key: 12.0
        if key in ("filter_volatility_ema_span", "filter_volume_ema_span")
        else 0.0
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
