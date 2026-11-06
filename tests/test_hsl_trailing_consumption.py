"""Offline regressions for the final fill-derived trailing consumption cohort."""

from copy import deepcopy

import pytest

from candlestick_manager import CandlestickManager
from live import hsl_live
from test_hsl_inputs import event as fill_event
from test_unstucking_safeguards import (
    _DummyPnlsManager,
    _dummy_config,
    _make_candles,
    _make_dummy_bot,
    _set_basic_state,
    _trailing_default,
)


NOW = 361_000
ROWS = [
    (180_000, 100.0, 105.0, 99.0, 102.0, 1.0),
    (240_000, 102.0, 109.0, 101.0, 108.0, 1.0),
    (300_000, 108.0, 112.0, 107.0, 110.0, 1.0),
]


class NoNetworkExchange:
    id = "fake"
    timeframes = {"1m": "1m", "5m": "5m", "1h": "1h", "1d": "1d"}

    def __init__(self):
        self.calls = []

    async def fetch_ohlcv(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("final trailing derivation must not fetch remotely")


def candle_manager(path, symbol, rows):
    exchange = NoNetworkExchange()
    cm = CandlestickManager(exchange=exchange, cache_dir=str(path), archive_enabled=False)
    cm._now_ms = lambda: NOW
    cm._persist_batch(symbol, _make_candles(rows), merge_cache=True)
    return cm, exchange


def trailing_bot(path, monkeypatch):
    import passivbot
    import utils

    monkeypatch.setattr(passivbot, "utc_ms", lambda: NOW)
    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    cfg = _dummy_config()
    cfg["bot"]["portfolio"]["hsl"]["scope"] = "coin"
    cfg["bot"]["long"]["hsl"]["enabled"] = True
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)
    bot.is_trailing = lambda sym, pside=None: pside == "long"
    bot.get_exchange_time = lambda: NOW
    bot._completed_candle_health_now_ms = lambda: NOW
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._staged_planner_required_surfaces = lambda **_: {"fills"}
    bot._pnls_manager = _DummyPnlsManager([
        fill_event(symbol=symbol, position_side="long", timestamp=120_000,
                   id="old-fill", psize=1.0, pprice=100.0, c_mult=1.0)
    ])
    bot.cm, exchange = candle_manager(path, symbol, ROWS)
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch(now_ms=NOW)
    for surface in ("balance", "positions", "open_orders", "fills"):
        ledger.stamp(surface, now_ms=NOW)
    return bot, symbol, exchange


async def pending_position_bot(path, monkeypatch):
    bot, symbol, exchange = trailing_bot(path, monkeypatch)
    baseline = [{"symbol": symbol, "position_side": "long", "size": 1.0, "price": 100.0}]
    changed = [{"symbol": symbol, "position_side": "long", "size": 1.5, "price": 101.0}]
    bot._apply_positions_snapshot(baseline)
    bot._trailing_fill_fetch_generation = 1
    await bot.update_trailing_data(allow_remote_fetch=False)
    assert bot._trailing_pending_fill_confirmations == {}
    bot._pnls_manager._events.append(
        fill_event(symbol=symbol, position_side="long", timestamp=240_000,
                   id="new-fill", psize=1.5, pprice=101.0, c_mult=1.0)
    )
    bot._apply_positions_snapshot(changed)
    bot._trailing_fill_fetch_generation = 2
    return bot, symbol, exchange


@pytest.mark.asyncio
async def test_preload_does_not_clear_confirmation_before_corrected_after_state(tmp_path, monkeypatch):
    # Control: performing the former early derivation really would confirm the
    # old matching after-state, not merely invoke a mocked confirmation hook.
    control, symbol, control_exchange = await pending_position_bot(tmp_path / "control", monkeypatch)
    assert control._trailing_pending_fill_confirmations == {(symbol, "long"): "fill:120000:old-fill"}
    await control.update_trailing_data(allow_remote_fetch=False)
    assert control._trailing_pending_fill_confirmations == {}
    assert control.trailing_prices[symbol]["long"]["max_since_open"] == pytest.approx(112.0)

    bot, symbol, exchange = await pending_position_bot(tmp_path / "subject", monkeypatch)
    pending = deepcopy(bot._trailing_pending_fill_confirmations)
    minimums = deepcopy(bot._trailing_pending_fill_min_generations)
    expected_states = deepcopy(bot._trailing_pending_position_states)
    old_receipt = hsl_live.runtime.observe_fills(bot, (NOW - 200, NOW - 100))
    assert old_receipt is not None
    bot._hsl_fill_observation = old_receipt
    old_anchor = bot._get_last_position_change_anchors()[symbol]["long"]
    signature = bot._fill_events_signature(bot._pnls_manager.get_events())
    bot._ensure_freshness_ledger().stamp("fills", signature, now_ms=NOW)

    await bot._preload_trailing_candles()
    assert bot._trailing_pending_fill_confirmations == pending
    assert bot._trailing_pending_fill_min_generations == minimums
    assert bot._trailing_pending_position_states == expected_states

    # A successful history renewal during slow preparation corrects derived
    # after-state while preserving the latest fill identity and ledger payload.
    # Those coarse equalities cannot validate the old trailing confirmation.
    bot._pnls_manager._events[-1] = fill_event(
        symbol=symbol, position_side="long", timestamp=240_000,
        id="new-fill", psize=2.5, pprice=130.0, c_mult=1.0
    )
    latest = hsl_live.runtime.observe_fills(bot, (NOW - 50, NOW))
    assert latest is not old_receipt and latest.tape == old_receipt.tape
    bot._hsl_fill_observation = latest
    bot._trailing_fill_fetch_generation = 3
    new_anchor = bot._get_last_position_change_anchors()[symbol]["long"]
    assert new_anchor["epoch"] == old_anchor["epoch"]
    assert new_anchor["timestamp"] == old_anchor["timestamp"]
    assert new_anchor["psize"] != old_anchor["psize"]
    assert bot._fill_events_signature(bot._pnls_manager.get_events()) == signature

    owner = hsl_live.owner(bot)
    receipts = await owner.prepare_ordinary_consumption(owner.account_facts())
    assert receipts.fills is latest
    assert bot._trailing_pending_fill_confirmations == pending
    assert bot._trailing_pending_fill_min_generations == minimums
    assert bot._trailing_pending_position_states == expected_states
    assert bot._trailing_fill_confirmation_diagnostics[(symbol, "long")]["failed_predicates"] == ["fill_after_state_mismatch"]
    assert bot._orchestrator_trailing_unavailable_reasons == {symbol: ["position_fill_confirmation_pending"]}
    assert bot._orchestrator_trailing_unavailable_psides == {symbol: ["long"]}
    assert bot.trailing_prices[symbol]["long"] == _trailing_default()
    assert exchange.calls == control_exchange.calls == []


@pytest.mark.asyncio
async def test_final_cache_only_missing_reset_candle_rejects_old_bundle(tmp_path, monkeypatch):
    import passivbot
    import passivbot_rust as pbr
    from test_orchestrator_json_api import adaptive_strategy_params, compute, make_input, make_symbol

    bot, symbol, exchange = trailing_bot(tmp_path / "complete", monkeypatch)
    await bot.update_trailing_data(allow_remote_fetch=False)
    assert bot._orchestrator_trailing_unavailable_symbols == set()
    assert bot.trailing_prices[symbol]["long"]["max_since_open"] == pytest.approx(112.0)
    old_bundle = deepcopy(bot.trailing_prices[symbol]["long"])
    epochs = deepcopy(bot._trailing_position_change_epochs)
    # Cache loss removes the first completed minute after the unchanged fill.
    # Later rows cannot establish that reset boundary or inherit the old bundle.
    bot.cm, missing_exchange = candle_manager(tmp_path / "missing-reset", symbol, ROWS[1:])
    reads, archive_calls = [], []
    local_read = bot.cm.get_candles

    async def observed_read(*args, **kwargs):
        reads.append(kwargs)
        return await local_read(*args, **kwargs)

    async def forbidden_archive(*args, **kwargs):
        archive_calls.append((args, kwargs))
        raise AssertionError("final trailing derivation must not fetch archives")

    bot.cm.get_candles = observed_read
    bot.cm.archive_enabled = True
    bot.cm._prefetch_archives_for_range = forbidden_archive
    owner = hsl_live.owner(bot)
    await owner.prepare_ordinary_consumption(owner.account_facts())
    assert bot._trailing_position_change_epochs == epochs
    assert bot._orchestrator_trailing_unavailable_reasons == {symbol: ["incomplete_trailing_candle_coverage"]}
    assert bot._orchestrator_trailing_unavailable_psides == {symbol: ["long"]}
    trailing, available = passivbot._orchestrator_trailing_input(bot, symbol, "long")
    assert available is False
    assert reads and all(call["allow_remote_fetch"] is False for call in reads)
    assert exchange.calls == missing_exchange.calls == archive_calls == []
    assert bot.cm._cache[symbol]["ts"].tolist() == [240_000, 300_000]

    # Exercise the actual native strategy consumer. The old known extrema
    # would trigger a close; unavailable current coverage must prevent that
    # consuming branch regardless of retained numbers in Python's container.
    assert not getattr(pbr, "__is_stub__", False)
    strategy = adaptive_strategy_params(
        entry={"retracement_base_fraction": .01},
        close={"retracement_base_fraction": .01},
    )
    native_symbol = make_symbol(
        0, bid=110.0, ask=110.0, long_pos_size=1.0, long_pos_price=100.0,
        long_strategy=strategy,
    )
    native_symbol["long"].update(trailing=old_bundle, trailing_available=True)
    control = compute(pbr, make_input(balance=1000.0, symbols=[native_symbol]))
    assert any(order["order_type"] == "close_trailing_long" for order in control["orders"])
    native_symbol["long"].update(trailing=trailing, trailing_available=available)
    actual = compute(pbr, make_input(balance=1000.0, symbols=[native_symbol]))
    assert actual["orders"] == []
    assert {"strategy_input_unavailable": {
        "symbol_idx": 0, "pside": "long", "scope": "strategy_orders"
    }} in actual["diagnostics"]["warnings"]
