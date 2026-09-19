"""The minimal exit contract rejects partial intent without requiring risk history."""
import copy
import json

import pytest

from live.reconciler import parse_and_validate_protective_closes
from passivbot_exceptions import FatalBotException


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    assert not getattr(pbr, "__is_stub__", False)
    return pbr


def close_inputs():
    return [dict(symbol_idx=0, pside="short", position_size=-0.000003,
                 order_book={"bid": 99.97, "ask": 100.03}, price_step=0.1,
                 execution_type="market")]


def test_real_rust_closes_whole_dust_position(require_real_passivbot_rust_module):
    pbr = require_real_passivbot_rust_module
    inputs = close_inputs()
    orders = parse_and_validate_protective_closes(pbr.compute_protective_closes_json(json.dumps(inputs)), inputs)
    assert orders == [dict(symbol_idx=0, pside="short", qty=0.000003, price=100.1,
                          order_type="close_panic_short", execution_type="market",
                          execution_priority="risk_critical")]


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "qty", "sign", "price", "scope", "type", "priority", "family", "nan", "boolean_index"])
def test_protective_producer_errors_are_fatal(require_real_passivbot_rust_module, mutation):
    inputs = close_inputs()
    orders = json.loads(require_real_passivbot_rust_module.compute_protective_closes_json(json.dumps(inputs)))
    if mutation == "omit":
        orders.clear()
    elif mutation == "duplicate":
        orders.append(copy.deepcopy(orders[0]))
    else:
        field, value = {
            "qty": ("qty", 0.000001), "sign": ("qty", -0.000003),
            "price": ("price", 99.9), "scope": ("pside", "long"),
            "type": ("execution_type", "limit"), "priority": ("execution_priority", "ordinary"),
            "family": ("order_type", "entry_initial_normal_short"),
            "nan": ("price", float("nan")), "boolean_index": ("symbol_idx", False),
        }[mutation]
        orders[0][field] = value
    with pytest.raises(FatalBotException):
        parse_and_validate_protective_closes(json.dumps(orders), inputs)


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_sign", "bad_step", "crossed", "missing", "strategy_field"])
def test_real_rust_rejects_invalid_exit_inputs(require_real_passivbot_rust_module, mutation):
    inputs = close_inputs()
    if mutation == "duplicate":
        inputs.append(copy.deepcopy(inputs[0]))
    elif mutation == "wrong_sign":
        inputs[0]["position_size"] = 1.0
    elif mutation == "bad_step":
        inputs[0]["price_step"] = 0.0
    elif mutation == "crossed":
        inputs[0]["order_book"]["ask"] = 99.0
    elif mutation == "missing":
        del inputs[0]["position_size"]
    else:
        inputs[0]["balance"] = 100.0
    with pytest.raises(ValueError):
        require_real_passivbot_rust_module.compute_protective_closes_json(json.dumps(inputs))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['unavailable', 'malformed'])
async def test_ready_exit_isolated_from_another_symbols_quote_outage(monkeypatch, require_real_passivbot_rust_module, failure):
    from types import SimpleNamespace
    from passivbot import Passivbot
    from live.market_snapshot import MarketSnapshotUnavailable
    from live import planning_gates
    quote = SimpleNamespace(bid=99.0, ask=101.0)
    async def quotes(symbols):
        if 'A' in symbols:
            if failure == 'malformed':
                raise ValueError('malformed quote')
            raise MarketSnapshotUnavailable('quote temporarily unavailable')
        return {'B': quote}
    monkeypatch.setattr(Passivbot, '_equity_hard_stop_enabled', lambda *a, **k: False)
    monkeypatch.setattr(Passivbot, '_monitor_record_price_ticks', lambda *a, **k: None)
    monkeypatch.setattr(planning_gates, 'build_protective_planning_snapshot',
                        lambda bot, symbols, snapshots: SimpleNamespace(last_prices=lambda: {'B': 100.0}))
    bot = SimpleNamespace(
        positions={symbol: {'long': {'size': 2.0}} for symbol in ('A', 'B')},
        price_steps={'A': 0.1, 'B': 0.1},
        _get_orchestrator_market_snapshots=quotes,
        _record_market_snapshot_surface=lambda *a: None,
        _to_executable_orders=lambda orders, prices: (orders, []),
        _finalize_reduce_only_orders=lambda orders, prices: orders,
    )
    if failure == 'malformed':
        with pytest.raises(ValueError, match='malformed quote'):
            await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot, target_psides_by_symbol={'A': {'long'}, 'B': {'long'}})
        return
    result = await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot, target_psides_by_symbol={'A': {'long'}, 'B': {'long'}})
    assert set(result) == {'B'}
    assert result['B'][0][0] == -2.0
    assert bot._hsl_protective_unavailable_symbols == {'A'}
    assert bot._protective_panic_reconcile_psides_by_symbol == {'B': {'long'}}
    assert bot._protective_panic_reconcile_symbols == ['B']


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['missing', 'nan', 'crossed', 'payload_shape'])
async def test_provider_quote_partition_keeps_combined_freshness_for_two_ready_exits(monkeypatch, require_real_passivbot_rust_module, fault):
    from types import SimpleNamespace
    from live.freshness import FreshnessLedger
    from live.market_snapshot import MarketSnapshotProvider
    from live import market_data
    from passivbot import Passivbot
    from utils import utc_ms
    tickers = {symbol: {'bid': 99.0, 'ask': 101.0, 'last': 100.0} for symbol in ('A', 'B', 'C')}
    if fault == 'missing':
        del tickers['A']
    elif fault == 'nan':
        tickers['A']['bid'] = float('nan')
    elif fault == 'crossed':
        tickers['A']['ask'] = 98.0
    async def fetch():
        return [] if fault == 'payload_shape' else tickers
    provider = MarketSnapshotProvider(exchange_name='fake', fetch_tickers=fetch)
    ledger = FreshnessLedger()
    ledger.begin_epoch()
    for surface in ('positions', 'open_orders'):
        ledger.stamp(surface, (), now_ms=utc_ms())
    bot = SimpleNamespace(
        positions={symbol: {'long': {'size': 2.0}} for symbol in tickers.keys() | {'A'}},
        price_steps={symbol: 0.1 for symbol in ('A', 'B', 'C')},
        _ensure_freshness_ledger=lambda: ledger,
        _live_market_snapshot_max_age_ms=lambda: 10_000,
        config_get=lambda *a: 'fake_protective_quote_partition',
        _to_executable_orders=lambda orders, prices: (orders, []),
        _finalize_reduce_only_orders=lambda orders, prices: orders,
    )
    bot._market_snapshot_signature = lambda symbols, snapshots: market_data.market_snapshot_signature(bot, symbols, snapshots)
    bot._record_market_snapshot_surface = lambda symbols, snapshots: market_data.record_market_snapshot_surface(bot, symbols, snapshots)
    bot._market_snapshot_signature_invalid = lambda symbols: market_data.market_snapshot_signature_invalid(bot, symbols)
    async def quotes(symbols):
        snapshots = await provider.get_snapshots(symbols)
        bot._record_market_snapshot_surface(symbols, snapshots)
        return snapshots
    bot._get_orchestrator_market_snapshots = quotes
    monkeypatch.setattr(Passivbot, '_equity_hard_stop_enabled', lambda *a, **k: False)
    monkeypatch.setattr(Passivbot, '_monitor_record_price_ticks', lambda *a, **k: None)
    targets = {symbol: {'long'} for symbol in ('A', 'B', 'C')}
    if fault == 'payload_shape':
        with pytest.raises(RuntimeError, match='non-dict'):
            await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot, target_psides_by_symbol=targets)
        return
    result = await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot, target_psides_by_symbol=targets)
    assert set(result) == {'B', 'C'}
    assert all(orders[0][0] == -2.0 for orders in result.values())
    assert bot._current_planning_snapshot.symbols == ('B', 'C')
    assert bot._market_snapshot_signature_invalid(['B', 'C']) == []
    assert bot._hsl_protective_unavailable_symbols == {'A'}
