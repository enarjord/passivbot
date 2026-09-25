"""Revised scoped HSL through real multi-coin screening and replay buffers."""
from copy import deepcopy
import numpy as np
import pytest
from config import prepare_config
from config.hsl_revised import generated_template
from config.schema import get_template_config
from optimization.gpu.service import MpsMulticoinProxy
from tools.gpu_proxy_benchmark import _synthetic_hlcvs

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason='GPU unavailable',
)


def make_proxy(mode, strategy='trailing_martingale', sides=('long','short'), minutes=3000,
               engine='revised', enabled=True, override=None, chunk=False, lookback=1, coin_count=2):
    config=(generated_template(get_template_config(),mode) if engine=='revised'
            else get_template_config())
    config['live'].update(hsl_engine=engine,hsl_signal_mode=mode)
    coins=['AAA','BBB']+[f'COIN{i}' for i in range(2,coin_count)]
    config['live'].update(strategy_kind=strategy,
        approved_coins={s:coins if s in sides else [] for s in ('long','short')},
        ignored_coins={'long':[],'short':[]},pnls_max_lookback_days=lookback,max_warmup_minutes=60)
    config['backtest'].update(coins={'binance':coins},starting_balance=1000.,btc_collateral_cap=0.)
    for side in ('long','short'):
        config['bot'][side]['risk'].update(n_positions=1 if side in sides else 0,
            total_wallet_exposure_limit=2 if side in sides else 0,
            position_exposure_enforcer_enabled=False)
        config['bot'][side]['hsl'].update(enabled=enabled and (mode!='unified' or engine=='legacy'),red_threshold=.02,
            ema_span_minutes=2.5,cooldown_minutes_after_red=5,restart_after_red_policy='always')
    if mode=='unified' and engine=='revised':
        config['bot']['hsl'].update(enabled=enabled,red_threshold=.02,ema_span_minutes=2.5,
            cooldown_minutes_after_red=5,restart_after_red_policy='always')
    if override:
        config['coin_overrides']={'AAA':{'bot':{'long':{'hsl':override}}}}
    config=prepare_config(config,verbose=False,target='canonical',runtime=None)
    config['backtest']['coins']={'binance':coins}
    candles,timestamps=_synthetic_hlcvs(minutes,len(coins),43)
    markets={c:dict(qty_step=.001,price_step=.01,min_qty=.001,min_cost=1.,c_mult=1.,
                    maker=.0002,taker=1e-6,exchange='binance') for c in coins}
    proxy=MpsMulticoinProxy(config=config,hlcvs=candles,mss=markets,btc=np.full(minutes,50000.),
        timestamps=timestamps,exchange='binance',batch_size=3,needed_metrics={'adg_usd'},
        max_dispatch_candidate_bars=minutes * len(coins) * len(sides) if chunk else 1000000)
    if chunk:
        assert proxy.fused_runner is None  # temporal multi-coin replay is single-side
        for runner in proxy.runners.values():
            runner.max_dispatch_candidate_bars=500
    return proxy


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('sides',[('long',),('short',),('long','short')])
def test_multicoin_revised_service_dispatch(strategy,mode,sides):
    proxy=make_proxy(mode,strategy,sides)
    rows=proxy.evaluate([{}])
    assert len(rows)==1
    assert np.isfinite(rows[0]['adg_usd'])
    runners=[proxy.fused_runner] if proxy.fused_runner else list(proxy.runners.values())
    assert all(r.hsl_engine=='revised' for r in runners)


def raw(proxy,candidates,**kwargs):
    if proxy.fused_runner:
        runner=proxy.fused_runner
        matrix=np.concatenate([proxy._parameter_matrix(candidates,s) for s in ('long','short')],axis=1)
    else:
        side=proxy.sides[0]
        runner=proxy.runners[side]
        matrix=proxy._parameter_matrix(candidates,side)
    return runner,{k:v.clone() if isinstance(v,torch.Tensor) else v
                   for k,v in runner.run(matrix,**kwargs).items()}


def compare(a,b):
    assert set(a)==set(b)
    for key in a:
        if isinstance(a[key],torch.Tensor):
            np.testing.assert_allclose(a[key].cpu(),b[key].cpu(),rtol=3e-5,atol=2e-5,
                                       equal_nan=True,err_msg=key)
        else:
            assert a[key]==b[key]


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('sides',[('long',),('short',),('long','short')])
def test_disabled_revised_multicoin_preserves_legacy_results(strategy,mode,sides):
    _,legacy=raw(make_proxy(mode,strategy,sides,enabled=False,engine='legacy'),[{}])
    runner,revised=raw(make_proxy(mode,strategy,sides,enabled=False),[{}])
    assert runner.dispatch_hsl_disabled is False
    compare(legacy,revised)


@pytest.mark.parametrize('mode',['coin','pside','unified'])
def test_multicoin_temporal_replay_and_scratch_batches(mode):
    candidates=[{f'long_hsl_red_threshold':1e-6,'hsl_red_threshold':1e-6},
                {'long_hsl_red_threshold':.002,'hsl_red_threshold':.002},
                {'long_hsl_red_threshold':.006,'hsl_red_threshold':.006}]
    full=make_proxy(mode,sides=('long',))
    _,expected=raw(full,candidates)
    chunks=make_proxy(mode,sides=('long',),chunk=True)
    runner,actual=raw(chunks,candidates)
    compare(expected,actual)
    assert float(actual['fill_count'].sum())>0
    runner.revised_scratch_budget_bytes=runner._history_bytes_per_candidate()*2
    _,split=raw(chunks,candidates,profile=True)
    compare(actual,split)
    assert runner.last_profile['candidate_batch_count']==2
    _,repeated=raw(chunks,candidates)
    compare(split,repeated)


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
@pytest.mark.parametrize('mode',['coin','pside','unified'])
def test_multicoin_candidates_do_not_share_history(strategy,mode):
    p=make_proxy(mode,strategy)
    candidates=[{}, {'long_hsl_red_threshold':1e-6,'short_hsl_red_threshold':1e-6,
                     'hsl_red_threshold':1e-6}]
    runner,first=raw(p,candidates)
    _,reordered=raw(p,candidates[::-1])
    compare({k:v.flip(0) if isinstance(v,torch.Tensor) else v for k,v in first.items()},reordered)
    runner.revised_scratch_budget_bytes=runner._history_bytes_per_candidate()
    _,split=raw(p,candidates)
    compare(first,split)


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
@pytest.mark.parametrize('mode',['coin','pside','unified'])
def test_multicoin_hsl_threshold_controls_actual_panics(strategy,mode):
    p=make_proxy(mode,strategy)
    _,r=raw(p,[{'long_hsl_red_threshold':1e-6,'short_hsl_red_threshold':1e-6,
               'hsl_red_threshold':1e-6},{}])
    triggers=r['hsl_triggers_long']+r['hsl_triggers_short']
    assert triggers[0]>0
    assert triggers[1]==0
    assert r['alive'].all()
    if mode=='unified':
        assert (r['hsl_triggers_short']==0).all()  # one portfolio owner


@pytest.mark.parametrize('lookback',[0,'all'])
@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
def test_multicoin_inactive_history_is_inert(strategy,lookback):
    p=make_proxy('coin',strategy,enabled=False,lookback=lookback)
    _,r=raw(p,[{}])
    assert r['alive'].all()
    assert (r['hsl_triggers_long']+r['hsl_triggers_short']==0).all()


@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
def test_multicoin_static_policy_does_not_replace_other_coin_base(strategy):
    p=make_proxy('coin',strategy,override={'enabled':False,'red_threshold':.7})
    assert p.base_params['long']['hsl_enabled']==1.
    assert p.base_params['long']['hsl_red_threshold']==pytest.approx(.02)
    _,r=raw(p,[{'long_hsl_red_threshold':1e-6,'short_hsl_red_threshold':1e-6}])
    assert r['alive'].all()


@pytest.mark.parametrize('strategy,chunk,sides',[
    ('ema_anchor',False,('long',)),('ema_anchor',False,('long','short')),
    ('trailing_martingale',False,('long',)),
    ('trailing_martingale',False,('long','short')),
    ('trailing_martingale',True,('long',))])
def test_scratch_profiles_count_each_candidate_once(strategy,chunk,sides):
    from optimization.gpu.service import _new_gpu_proxy_profile, _add_gpu_runner_profile
    p=make_proxy('coin',strategy,sides=sides,chunk=chunk)
    candidates=[{}, {}, {}]
    runner,_=raw(p,candidates)
    runner.revised_scratch_budget_bytes=runner._history_bytes_per_candidate()*2
    ends=np.array([3000,1700,500],dtype=np.int32)
    runner,_=raw(p,candidates,end_steps=ends,profile=True)
    steps=int((np.clip(ends,1,runner.n-1)-1).sum())
    assert runner.last_profile['kernel_candidate_steps']==steps
    assert runner.last_profile['candidate_batch_sizes']==[2,1]
    if chunk:
        assert runner.last_profile['dispatch_count']>2
    profile=_new_gpu_proxy_profile(p,candidates,[runner],coin_count=2,side_count=len(sides))
    _add_gpu_runner_profile(profile,runner,side_count=len(sides))
    assert profile['kernel_candidate_bars']==steps*2*len(sides)
    assert profile['actual_dispatch_batch_sizes']==[2,1]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal specialization")
@pytest.mark.parametrize('strategy',['ema_anchor','trailing_martingale'])
@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('sides',[('long',),('long','short')])
def test_metal_revised_capacity_matches_full_arrays(strategy,mode,sides):
    candidates=[{}, {'long_hsl_red_threshold':1e-6,'short_hsl_red_threshold':1e-6,
                     'hsl_red_threshold':1e-6}]
    small=make_proxy(mode,strategy,sides,coin_count=3)
    runner,actual=raw(small,candidates)
    assert runner.mps_coin_capacity==(4 if strategy=="trailing_martingale" else None)
    full=make_proxy(mode,strategy,sides,coin_count=3)
    full_runner=full.fused_runner if full.fused_runner else full.runners['long']
    if strategy=="trailing_martingale":
        full_runner.mps_coin_capacity=64
    _,expected=raw(full,candidates)
    compare(actual,expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal specialization")
def test_metal_capacity_replay_layout_matches_compiled_variant():
    small=make_proxy('coin',sides=('long',),coin_count=3,chunk=True)
    runner,actual=raw(small,[{}])
    full=make_proxy('coin',sides=('long',),coin_count=3,chunk=True)
    full_runner=full.runners['long']
    full_runner.mps_coin_capacity=64
    _,expected=raw(full,[{}])
    compare(actual,expected)
    assert 0 < runner._replay_state_bytes < full_runner._replay_state_bytes
