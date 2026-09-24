from copy import deepcopy

import pytest

from backtest import get_backtest_execution_settings, prep_backtest_args
from config import prepare_config
from config_utils import get_template_config


@pytest.mark.parametrize('buffer', [0.0, 0.0001, 0.001, 0.999])
def test_buffer_survives_config_and_backtest_payload(buffer):
    raw = get_template_config()
    raw['backtest']['limit_order_fill_buffer_pct'] = buffer
    cfg = prepare_config(raw, verbose=False)
    assert cfg['backtest']['limit_order_fill_buffer_pct'] == buffer
    cfg['backtest']['coins'] = {'binance': ['BTC']}
    mss = {'BTC': dict(maker=0.0002, taker=0.00055, qty_step=0.001,
                       price_step=0.1, min_qty=0.001, min_cost=10.0, c_mult=1.0)}
    *_, params = prep_backtest_args(cfg, mss, 'binance')
    assert params['limit_order_fill_buffer_pct'] == buffer


def test_older_configs_receive_zero_default():
    cfg = get_template_config()
    del cfg['backtest']['limit_order_fill_buffer_pct']
    assert prepare_config(cfg, verbose=False)['backtest']['limit_order_fill_buffer_pct'] == 0.0


@pytest.mark.parametrize('buffer', [-0.0001, 1.0, 2.0, float('nan'), float('inf'), -float('inf'), None, True, 'invalid'])
def test_invalid_buffer_rejected_by_config_and_runtime(buffer):
    cfg = get_template_config()
    cfg['backtest']['limit_order_fill_buffer_pct'] = buffer
    with pytest.raises((ValueError, TypeError), match='limit_order_fill_buffer_pct'):
        prepare_config(cfg, verbose=False)
    with pytest.raises((ValueError, TypeError), match='limit_order_fill_buffer_pct'):
        get_backtest_execution_settings(cfg)


def test_resume_requires_same_fill_assumption():
    from optimize import _resume_config_mismatches
    cfg = get_template_config()
    prior = deepcopy(cfg)
    from optimization.evaluation_contract import build_evaluation_contract
    prior["optimizer_evaluation_contract"] = build_evaluation_contract(cfg)
    assert _resume_config_mismatches(prior, cfg) == []
    cfg['backtest']['limit_order_fill_buffer_pct'] = 0.0001
    assert any('limit_order_fill_buffer_pct' in x for x in _resume_config_mismatches(prior, cfg))
    del prior['backtest']['limit_order_fill_buffer_pct']
    assert any('limit_order_fill_buffer_pct' in x for x in _resume_config_mismatches(prior, cfg))
    cfg['backtest']['limit_order_fill_buffer_pct'] = 0.0
    assert _resume_config_mismatches(prior, cfg) == []


def test_gpu_rejects_nonzero_buffer_before_preparing_data():
    from optimization.backends.gpu_backend import _validate_gpu_static_scope
    cfg = get_template_config()
    cfg['backtest']['limit_order_fill_buffer_pct'] = 0.0001
    with pytest.raises(ValueError, match='limit_order_fill_buffer_pct.*CPU'):
        _validate_gpu_static_scope(cfg)


def _native_args():
    import numpy as np

    cfg = get_template_config()
    cfg['backtest']['coins'] = {'binance': ['BTC']}
    cfg['backtest']['btc_collateral_cap'] = 0.0
    cfg['backtest']['starting_balance'] = 1000.0
    cfg['live']['approved_coins'] = {'long': ['BTC'], 'short': []}
    cfg['live']['market_orders_allowed'] = False
    for side in ['long', 'short']:
        cfg['bot'][side]['hsl']['enabled'] = False
        cfg['bot'][side]['unstuck']['enabled'] = False
        cfg['bot'][side]['risk']['n_positions'] = 1
        cfg['bot'][side]['risk']['total_wallet_exposure_limit'] = 1.0 if side == 'long' else 0.0
        cfg['bot'][side]['strategy']['trailing_martingale']['entry']['ema_gate_mode'] = 'disabled'
    mss = {'BTC': dict(maker=0.0002, taker=0.00055, qty_step=0.001,
                       price_step=0.01, min_qty=0.001, min_cost=1.0, c_mult=1.0)}
    bot, strategy, exchange, params = prep_backtest_args(cfg, mss, 'binance')
    n = 60
    params.update(first_timestamp_ms=1704067200000, requested_start_timestamp_ms=1704067200000,
                  first_valid_indices=[0], last_valid_indices=[n - 1], warmup_minutes=[1],
                  trade_start_indices=[1], global_warmup_bars=1, candle_interval_minutes=1)
    candles = np.tile([[[100.005, 99.995, 100.0, 1000.0]]], (n, 1, 1))
    return candles, np.full(n, 50000.0), bot, strategy, exchange, params


def test_real_rust_buffer_rejects_marginal_entry_and_keeps_fill_price():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(pbr, '__is_stub__', False)
    verify_loaded_runtime_extension()
    args = _native_args()
    baseline = pbr.run_backtest(*args)
    assert len(baseline[0]) > 0
    # The native fill schema has fill_price at index 10.
    assert all(float(fill[10]) == 100.0 for fill in baseline[0])
    args[-1]['limit_order_fill_buffer_pct'] = 0.0001
    stressed = pbr.run_backtest(*args)
    assert len(stressed[0]) == 0


@pytest.mark.parametrize('value', [-0.1, 1.0, float('nan'), float('inf')])
def test_real_rust_rejects_invalid_buffer_at_python_boundary(value):
    import passivbot_rust as pbr

    assert not getattr(pbr, '__is_stub__', False)
    args = _native_args()
    args[-1]['limit_order_fill_buffer_pct'] = value
    with pytest.raises(ValueError, match='limit_order_fill_buffer_pct'):
        pbr.run_backtest(*args)


def test_suite_can_vary_buffer_without_mutating_base_config():
    from suite_runner import SuiteScenario, apply_scenario

    cfg = get_template_config()
    cfg['live']['approved_coins'] = {'long': ['BTC'], 'short': ['BTC']}
    cfg['live']['ignored_coins'] = {'long': [], 'short': []}
    cfg['backtest']['exchanges'] = ['binance']
    scenario = SuiteScenario(
        label='strict_fills', start_date=None, end_date=None,
        coins=['BTC'], ignored_coins=[],
        overrides={'backtest.limit_order_fill_buffer_pct': 0.0001},
    )
    changed, _ = apply_scenario(
        cfg, scenario, master_coins=['BTC'], master_ignored=[],
        available_exchanges=['binance'], available_coins={'BTC'},
        base_coin_sources={'BTC': 'binance'},
    )
    assert get_backtest_execution_settings(changed).limit_order_fill_buffer_pct == 0.0001
    assert cfg['backtest']['limit_order_fill_buffer_pct'] == 0.0
