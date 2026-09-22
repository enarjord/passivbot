"""Torch-free revised GPU policy ownership and optimizer gene transport."""
from copy import deepcopy

import pytest
from config.hsl_revised import generated_template
from config.schema import get_template_config
from optimization.backends.gpu_backend import (
    EMA_BOUND_MAP, _gpu_hsl_bound_map, _gpu_hsl_checkpoint_contract,
    _gpu_hsl_parameter_active, _gpu_hsl_search_sides, _gpu_hsl_side_enabled,
    _validate_hsl_bound_contracts, _validate_gpu_suite_override_paths,
)
from optimization.bounds import Bound


def config(mode):
    c = generated_template(get_template_config(), mode)
    c['live']['approved_coins'] = {'long': ['AAA'], 'short': ['AAA']}
    for side in ('long', 'short'):
        c['bot'][side]['risk'].update(n_positions=1, total_wallet_exposure_limit=1.)
        c['bot'][side]['hsl']['enabled'] = mode != 'unified'
    if mode == 'unified':
        c['bot']['hsl']['enabled'] = True
    return c


def test_unified_has_portfolio_genes_and_policy_even_with_side_hsl_disabled():
    c=config('unified')
    assert all(_gpu_hsl_side_enabled(c, side) for side in ('long','short'))
    m=_gpu_hsl_bound_map(c, EMA_BOUND_MAP)
    assert m['hsl_red_threshold'] == 'hsl_red_threshold'
    assert not any(k.startswith(('long_hsl_','short_hsl_')) for k in m)
    active=_gpu_hsl_search_sides(c, None)
    assert active == {'portfolio'}
    assert _gpu_hsl_parameter_active('hsl_red_threshold', active)
    assert not _gpu_hsl_parameter_active('long_hsl_red_threshold', active)
    _validate_gpu_suite_override_paths(c,label='policy',overrides={'bot.hsl.red_threshold':.2})


@pytest.mark.parametrize('mode',['coin','pside'])
def test_directional_genes_keep_their_scope(mode):
    c=config(mode)
    assert _gpu_hsl_bound_map(c,EMA_BOUND_MAP) == EMA_BOUND_MAP
    assert _gpu_hsl_search_sides(c,None) == {'long','short'}
    assert not _gpu_hsl_parameter_active('hsl_red_threshold', {'long','short'})


def test_engine_and_portfolio_policy_invalidate_resume_identity():
    c=config('unified')
    original=_gpu_hsl_checkpoint_contract(c)
    changed=deepcopy(c)
    changed['bot']['hsl']['red_threshold'] += .01
    assert _gpu_hsl_checkpoint_contract(changed) != original
    changed=deepcopy(c)
    changed['live']['hsl_engine']='legacy'
    assert _gpu_hsl_checkpoint_contract(changed) != original


def test_portfolio_bounds_are_validated_using_the_portfolio_policy():
    c=config('unified')
    with pytest.raises(ValueError,match='greater than zero'):
        _validate_hsl_bound_contracts({'hsl_red_threshold':Bound(0.,.2)},c)
    with pytest.raises(ValueError,match='EMA span'):
        _validate_hsl_bound_contracts({'hsl_ema_span_minutes':Bound(.5,2.)},c)
    _validate_hsl_bound_contracts({'hsl_red_threshold':Bound(.01,.2),
                                 'hsl_ema_span_minutes':Bound(1.,2.5)},c)


def test_multicoin_revised_scope_is_rejected_explicitly():
    from optimization.backends.gpu_backend import _validate_scope_config
    from optimization.gpu.service import MpsMulticoinProxy
    c = config('unified')
    c['optimize']['scoring'] = [{'metric': 'adg_usd', 'goal': 'max'}]
    c['optimize']['limits'] = []
    with pytest.raises(ValueError, match='exactly one prepared coin'):
        _validate_scope_config(c, exchanges=['binance'], coin_count=2)
    with pytest.raises(ValueError, match='multicoin integration'):
        MpsMulticoinProxy(config=c, hlcvs=None, mss=None, btc=None,
                         timestamps=None, exchange='binance', batch_size=1,
                         needed_metrics={'adg_usd'})


@pytest.mark.parametrize('base_enabled,override_enabled', [(True,False),(False,True)])
def test_coin_activity_uses_only_prepared_effective_policy(base_enabled,override_enabled):
    c=config('coin')
    c['backtest']['coins']={'binance':['AAA']}
    c['bot']['long']['hsl']['enabled']=base_enabled
    c['bot']['short']['hsl']['enabled']=False
    c['coin_overrides']={
        'AAA':{'bot':{'long':{'hsl':{'enabled':override_enabled}}}},
        'OUTSIDE':{'bot':{'short':{'hsl':{'enabled':True}}}},
    }
    assert _gpu_hsl_search_sides(c,None)==({'long'} if override_enabled else set())
    # A suite's prepared membership wins over its broader source config.
    c['backtest']['coins']={'binance':['AAA','OUTSIDE']}
    scenario={'config':c,'exchange':'binance','coins':['AAA'],'mss':{'AAA':{}}}
    assert _gpu_hsl_search_sides(c,[scenario])==({'long'} if override_enabled else set())


@pytest.mark.parametrize('minutes,expected',[(1440.25,1440),(1440.999,1440),
                                           (1441.-.0004/60000,1441)])
def test_revised_lookback_matches_native_millisecond_boundary(minutes,expected):
    from optimization.gpu.service import _revised_hsl_lookback_bars
    assert _revised_hsl_lookback_bars({'pnls_max_lookback_days':minutes/1440},
                                    hsl_enabled=True)==expected


@pytest.mark.parametrize('lookback',[0,-1])
def test_disabled_revised_lookback_does_not_require_history(lookback):
    from optimization.gpu.service import _revised_hsl_lookback_bars
    assert _revised_hsl_lookback_bars({'pnls_max_lookback_days':lookback},
                                    hsl_enabled=False)==0
