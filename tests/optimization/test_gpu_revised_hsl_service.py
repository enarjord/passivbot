"""Canonical configuration through the internal revised GPU screening service."""
from copy import deepcopy

import numpy as np
import pytest
from config import prepare_config
from config.hsl_revised import generated_template
from config.schema import get_template_config
from optimization.gpu.service import MpsSingleCoinProxy
from tools.gpu_proxy_benchmark import _synthetic_hlcvs

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason='GPU unavailable',
)


def make_proxy(mode, strategy='trailing_martingale', override=None, lookback=1, enabled=True):
    config = generated_template(get_template_config(), mode)
    config['live'].update(strategy_kind=strategy,
        approved_coins={'long':['AAA'], 'short':['AAA']},
        ignored_coins={'long':[], 'short':[]}, pnls_max_lookback_days=lookback,
        max_warmup_minutes=60)
    config['backtest'].update(coins={'binance':['AAA']}, starting_balance=1000.,
                             btc_collateral_cap=0.)
    for side in ('long','short'):
        config['bot'][side]['risk'].update(n_positions=1, total_wallet_exposure_limit=2,
                                          position_exposure_enforcer_enabled=False)
        config['bot'][side]['hsl'].update(enabled=enabled and mode != 'unified', red_threshold=.02,
            ema_span_minutes=2.5, cooldown_minutes_after_red=5,
            restart_after_red_policy='always')
    if mode == 'unified':
        config['bot']['hsl'].update(enabled=enabled, red_threshold=.015, ema_span_minutes=2.75,
            cooldown_minutes_after_red=7, restart_after_red_policy='never')
    if override:
        config['coin_overrides']={'AAA':{'bot':{'long':{'hsl':override}}}}
    config = prepare_config(config, verbose=False, target='canonical', runtime=None)
    config['backtest']['coins']={'binance':['AAA']}
    candles, timestamps = _synthetic_hlcvs(3000, 1, 43)
    markets={'AAA':dict(qty_step=.001,price_step=.01,min_qty=.001,min_cost=1.,c_mult=1.,
                        maker=.0002,taker=.0005,exchange='binance')}
    proxy=MpsSingleCoinProxy(config=config,hlcvs=candles,mss=markets,
        btc=np.full(3000,50000.),timestamps=timestamps,exchange='binance',batch_size=3,
        needed_metrics={'adg_usd'},max_dispatch_candidate_bars=100000)
    return proxy, config


@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
def test_canonical_revised_policy_reaches_shader(mode,strategy):
    proxy,config=make_proxy(mode,strategy)
    assert proxy.runner.hsl_engine == 'revised'
    assert proxy.runner.pnl_lookback_bars == 1440
    assert proxy.checkpoint_contract['backtest']['equity_hard_stop_loss']['engine']=='revised'
    matrix=proxy._parameter_matrix([{}])
    for side in ('long','short'):
        expected=config['bot']['hsl'] if mode=='unified' else config['bot'][side]['hsl']
        assert proxy.base_params[side]['hsl_ema_span_minutes']==expected['ema_span_minutes']
        assert proxy.base_params[side]['hsl_red_threshold']==expected['red_threshold']
    result=proxy.runner.run(matrix)
    assert torch.isfinite(result['balance']).all()
    assert float(result['fill_count'].sum())>0


def test_shared_portfolio_candidate_and_static_coin_override_precedence():
    proxy,_=make_proxy('unified')
    candidates=[{'hsl_red_threshold':.03},{'hsl_red_threshold':.06}]
    before=deepcopy(candidates)
    matrix=proxy._parameter_matrix(candidates)
    i=proxy.param_keys.index('hsl_red_threshold')
    np.testing.assert_array_equal(matrix[:,i], [.03,.06])
    np.testing.assert_array_equal(matrix[:,len(proxy.param_keys)+i], [.03,.06])
    assert candidates==before
    coin,_=make_proxy('coin',override={'red_threshold':.07,'ema_span_minutes':4.5})
    matrix=coin._parameter_matrix([{'long_hsl_red_threshold':.03,'short_hsl_red_threshold':.04}])
    i=coin.param_keys.index('hsl_red_threshold')
    assert matrix[0,i]==pytest.approx(.07)
    assert matrix[0,len(coin.param_keys)+i]==pytest.approx(.04)


@pytest.mark.parametrize('lookback',[0,'all'])
@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
def test_inactive_revised_zero_and_unbounded_history_execute(lookback,strategy):
    proxy,_=make_proxy('coin',strategy,lookback=lookback,enabled=False)
    assert proxy.runner.pnl_lookback_bars==0
    assert proxy.runner.revised_capacity==2
    rows=proxy.evaluate([{}])
    assert np.isfinite(rows[0]['adg_usd'])
