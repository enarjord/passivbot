import numpy as np
import pytest

from optimization.gpu.model import _strict_fill_tick_boundaries


@pytest.mark.parametrize('buffer', [0.0, 0.0001, 0.0015, 0.1, 0.999])
@pytest.mark.parametrize('step', [0.000001, 0.0001, 0.01, 0.1, 0.5])
def test_buffered_ticks_match_strict_float64_order_predicate(buffer, step):
    # Decimal-preserving exchange prices, including equality and adjacent floats.
    ticks = np.array([1, 7, 999, 371177], dtype=np.int64)
    prices = np.round(ticks * step, 10)
    sell = prices * (1.0 + buffer)
    buy = prices * (1.0 - buffer)
    high = np.concatenate([np.nextafter(sell, -np.inf), sell, np.nextafter(sell, np.inf)])
    low = np.concatenate([np.nextafter(buy, -np.inf), buy, np.nextafter(buy, np.inf)])
    order_ticks = np.tile(ticks, 3)
    order_prices = np.tile(prices, 3)
    high_ticks, low_ticks = _strict_fill_tick_boundaries(high, low, step, buffer)
    np.testing.assert_array_equal(order_ticks <= high_ticks, high > order_prices * (1 + buffer))
    np.testing.assert_array_equal(order_ticks > low_ticks, low < order_prices * (1 - buffer))


@pytest.mark.parametrize('buffer', [-1, 1, float('nan'), float('inf'), True, None])
def test_buffered_ticks_reject_invalid_buffer(buffer):
    with pytest.raises(ValueError, match='limit_order_fill_buffer_pct'):
        _strict_fill_tick_boundaries([100.0], [99.0], 0.01, buffer)


def test_buffered_ticks_reject_overflow_before_integer_conversion():
    with pytest.raises(ValueError, match='32-bit'):
        _strict_fill_tick_boundaries([100.0], [99.0], 0.0001, np.nextafter(1.0, 0.0))


def test_buffered_suite_cache_separates_fill_assumptions(monkeypatch):
    from optimization.gpu import service
    from test_gpu_prepared_data import _inputs
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs['limit_order_fill_buffer_pct'])
        return dict(n_coins=2, n=10, invariant_bytes=1)
    monkeypatch.setattr(service, 'build_mps_multicoin_data', build)
    cache = {}
    original = service._prepared_multicoin_data(**_inputs(), cache=cache)
    buffered = service._prepared_multicoin_data(**_inputs(), cache=cache, limit_order_fill_buffer_pct=0.0015)
    assert buffered is not original
    assert service._prepared_multicoin_data(**_inputs(), cache=cache, limit_order_fill_buffer_pct=0.0015) is buffered
    assert calls == [0.0, 0.0015]


def test_buffer_changes_gpu_checkpoint_identity():
    from optimization.gpu.service import _gpu_proxy_execution_checkpoint_contract
    args = dict(strategy_kind='trailing_martingale', exchange='bybit', enabled_sides=['long'],
                hlcvs=np.ones((3, 1, 4)), timestamps=np.arange(3),
                backtest_params={'coins': ['BTC'], 'limit_order_fill_buffer_pct': 0.0},
                exchange_params=[], base_params={})
    zero = _gpu_proxy_execution_checkpoint_contract(**args)
    args['backtest_params']['limit_order_fill_buffer_pct'] = 0.0015
    buffered = _gpu_proxy_execution_checkpoint_contract(**args)
    assert zero != buffered
    assert buffered['backtest']['limit_order_fill_buffer_pct'] == 0.0015


def _fixture(side, strategy_kind, buffer, coin_count=1, market_orders=False):
    from config.schema import get_template_config
    cfg = get_template_config()
    cfg['live'].update(strategy_kind=strategy_kind, max_warmup_minutes=1,
                       market_orders_allowed=market_orders, market_order_near_touch_threshold=0.01)
    coins = ['BTC', 'ETH'][:coin_count]
    cfg['live']['approved_coins'] = {s: coins if s == side else [] for s in ['long', 'short']}
    cfg['backtest'].update(coins={'bybit': coins}, exchanges=['bybit'],
                          starting_balance=1000.0, limit_order_fill_buffer_pct=buffer)
    for s in ['long', 'short']:
        cfg['bot'][s]['risk'].update(total_wallet_exposure_limit=1.0 if s == side else 0.0,
                                     n_positions=coin_count if s == side else 0,
                                     entry_cooldown_minutes=0.0, we_excess_allowance_pct=0.0,
                                     position_exposure_enforcer_enabled=False,
                                     total_exposure_enforcer_enabled=False)
        cfg['bot'][s]['hsl']['enabled'] = False
        cfg['bot'][s]['unstuck']['enabled'] = False
        if strategy_kind == 'trailing_martingale':
            st = cfg['bot'][s]['strategy'][strategy_kind]
            st['entry'].update(ema_span_0=2., ema_span_1=3., initial_ema_dist=0.,
                               initial_qty_pct=0.1, ema_gate_mode='disabled',
                               threshold_base_pct=0.01, threshold_we_weight=0.,
                               threshold_volatility_1m_weight=0., threshold_volatility_1h_weight=0.,
                               retracement_base_pct=0., retracement_we_weight=0.,
                               retracement_volatility_1m_weight=0., retracement_volatility_1h_weight=0.)
            st['close'].update(qty_pct=1., threshold_base_pct=0.01, threshold_we_weight=0.,
                               threshold_volatility_1m_weight=0., threshold_volatility_1h_weight=0.,
                               retracement_base_pct=0., retracement_volatility_1m_weight=0.,
                               retracement_volatility_1h_weight=0.)
        else:
            cfg['bot'][s]['strategy'][strategy_kind].update(base_qty_pct=0.1, ema_span_0=2.,
                                                            ema_span_1=3., offset=0., offset_psize_weight=0.,
                                                            offset_volatility_1m_weight=0., offset_volatility_1h_weight=0.)
    n = 120
    candles = np.tile([[[100.005, 99.995, 100., 1000.]]], (n, coin_count, 1))
    timestamps = 1704067200000 + np.arange(n, dtype=np.int64) * 60000
    mss = {c: dict(qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=1., c_mult=1.,
                   maker=0.0002, taker=0.0005, exchange='bybit', first_valid_index=0,
                   last_valid_index=n-1, warmup_minutes=1) for c in coins}
    mss['__meta__'] = dict(requested_start_ts=int(timestamps[0]), warmup_minutes_requested=1)
    return cfg, candles, mss, np.full(n, 50000.), timestamps


@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('strategy_kind', ['trailing_martingale', 'ema_anchor'])
@pytest.mark.parametrize('coin_count', [1, 2])
@pytest.mark.parametrize('market_orders', [False, True])
def test_buffered_gpu_matches_rust_marginal_entries(side, strategy_kind, coin_count, market_orders):
    torch = pytest.importorskip('torch')
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip('GPU unavailable')
    from backtest import run_backtest
    from optimization.gpu.service import MpsSingleCoinProxy, MpsMulticoinProxy
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    cfg, candles, mss, btc, ts = _fixture(side, strategy_kind, 0.0015, coin_count, market_orders)
    proxy_cls = MpsSingleCoinProxy if coin_count == 1 else MpsMulticoinProxy
    for buffer in [0.0, 0.0015]:
        cfg['backtest']['limit_order_fill_buffer_pct'] = buffer
        proxy = proxy_cls(config=cfg, hlcvs=candles, mss=mss, btc=btc, timestamps=ts,
                          exchange='bybit', batch_size=1, needed_metrics={'fills_per_day'})
        result = proxy.evaluate([{}])[0]
        fills, _, exact = run_backtest(candles, mss, cfg, 'bybit', btc, ts)
        assert result['fills_per_day'] == pytest.approx(exact['fills_per_day'], rel=1e-5)
        if buffer and not market_orders:
            assert len(fills) == 0
        else:
            assert len(fills) > 0
        # Preparation must not alter candle inputs used by indicators or touches.
        if coin_count == 1:
            np.testing.assert_array_equal(proxy.data['high_f'].cpu().numpy(), candles[:, 0, 0].astype(np.float32))
            np.testing.assert_array_equal(proxy.data['low_f'].cpu().numpy(), candles[:, 0, 1].astype(np.float32))


@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('coin_count', [1, 2])
def test_buffer_delays_limit_close_until_strict_crossing(side, coin_count):
    torch = pytest.importorskip('torch')
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip('GPU unavailable')
    from backtest import run_backtest
    from optimization.gpu.service import MpsSingleCoinProxy, MpsMulticoinProxy
    cfg, candles, mss, btc, ts = _fixture(side, 'trailing_martingale', 0.0015, coin_count)
    candles[:, :, :3] = 100.0
    if side == 'long':
        candles[10, :, 1] = 99.8
        candles[11:21, :, 0] = 101.05  # Touch through close, but short of buffer.
        candles[21, :, 0] = 101.3
    else:
        candles[10, :, 0] = 100.2
        candles[11:21, :, 1] = 98.95
        candles[21, :, 1] = 98.7
    cls = MpsSingleCoinProxy if coin_count == 1 else MpsMulticoinProxy
    for buffer in [0.0, 0.0015]:
        cfg['backtest']['limit_order_fill_buffer_pct'] = buffer
        proxy = cls(config=cfg, hlcvs=candles, mss=mss, btc=btc, timestamps=ts,
                    exchange='bybit', batch_size=1, needed_metrics={'fills_per_day', 'adg_usd'})
        result = proxy.evaluate([{}])[0]
        fills, _, exact = run_backtest(candles, mss, cfg, 'bybit', btc, ts)
        assert len(fills) == 2 * coin_count
        close_fills = [f for f in fills if 'close' in str(f[13])]
        assert len(close_fills) == coin_count
        assert all(int(f[0]) == (21 if buffer else 11) for f in close_fills)
        assert result['fills_per_day'] == pytest.approx(exact['fills_per_day'], rel=1e-5)
        assert result['adg_usd'] == pytest.approx(exact['adg_usd'], abs=1e-5, rel=1e-3)
