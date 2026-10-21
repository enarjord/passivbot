"""Prepared execution freezes membership while planner identity retains source policy."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from config import prepare_config
from config.universe import materialize_side_universe_sources, universe_source_view
from backtest import build_backtest_payload
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_datasets import NativeDatasetRegistry, _worker_config
from optimization.native_planning import execution_key
from optimize import Evaluator, config_to_individual
from shared_arrays import SharedArraySpec, SharedArrayManager
from tools.gpu_parity import fixture_inputs, build_parser


def prepared(source, tmp_path, strategy_kind="trailing_martingale"):
    config,candles,markets,btc,timestamps=fixture_inputs(build_parser().parse_args([
        '--fixture',strategy_kind,'--sides','both','--coins','3','--bars','128']))
    config['bot']['global']['universe']['approved_coins']=source
    config['bot']['long']['universe']['approved_coins']=None
    config['bot']['short']['universe']['approved_coins']=['COIN01']
    config['bot']['long']['universe']['ignored_coins']=['COIN01']
    config['optimize']['scoring']=[{'metric':'adg_strategy_eq','goal':'max'}]
    config['optimize']['limits']=[]
    membership=deepcopy(config['backtest']['coins'])
    from config_utils import clean_config
    config=prepare_config(clean_config(config),base_config_path=str(tmp_path/'config.json'),verbose=False)
    config['backtest']['coins']=membership
    materialize_side_universe_sources(config,'approved_coins',{'long':['COIN00','COIN02'],'short':['COIN01']})
    return config,candles,markets,btc,timestamps


def masks(config,candles,markets,btc,timestamps):
    payload=build_backtest_payload(candles,markets,config,'binance',btc,timestamps)
    return {side:[row[side]['entry_eligible'] for row in payload.bot_params_list] for side in ('long','short')}


@pytest.mark.parametrize('kind',['all','file'])
@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
def test_dataset_policy_identity_and_native_masks_survive_file_change_and_deletion(tmp_path, kind, strategy_kind):
    path=tmp_path/'coins.json';path.write_text('["COIN00","COIN02"]')
    config,candles,markets,btc,timestamps=prepared('all' if kind=='all' else 'coins.json',tmp_path,strategy_kind)
    expected={'long':[True,False,True],'short':[False,True,False]}
    assert masks(config,candles,markets,btc,timestamps)==expected
    specs=[SharedArraySpec(name,arr.shape,arr.dtype.str) for name,arr in [('candles',candles),('btc',btc),('time',timestamps)]]
    dataset=PreparedGpuDataset(config=config,markets=markets,exchange='binance',
        hlcvs=specs[0],btc=specs[1],timestamps=specs[2],
        candle_coins=config['backtest']['coins']['binance'],metrics=('adg_strategy_eq',))
    assert json.loads(dataset.config_json)['bot']['long']['universe']['approved_coins'] is None
    assert execution_key(dataset.policy_config())==execution_key(config)
    path.write_text('["COIN01"]')
    for runtime in (dataset.execution_config(),_worker_config(dataset.policy_config())):
        assert masks(runtime,candles,markets,btc,timestamps)==expected
    path.unlink()
    assert masks(dataset.execution_config(),candles,markets,btc,timestamps)==expected
    changed=deepcopy(config)
    materialize_side_universe_sources(changed,'approved_coins',{'long':['COIN01'],'short':['COIN00','COIN02']})
    assert execution_key(changed)!=execution_key(config)  # Same union, different side eligibility.
    assert universe_source_view(config)['bot']['long']['universe']['approved_coins'] is None
    if kind == 'file':
        other = tmp_path/'other'
        other.mkdir()
        (other/'coins.json').write_text('["COIN00","COIN02"]')
        other_base = deepcopy(config)
        other_base['live']['base_config_path'] = str(other/'input.json')
        assert execution_key(other_base) == execution_key(config)  # Context cannot retarget a bound file.
        other_base['bot']['global']['universe']['approved_coins'] = str(other/'coins.json')
        with pytest.raises(ValueError, match='prepared long approved_coins membership'):
            execution_key(other_base)
        materialize_side_universe_sources(other_base, 'approved_coins', {'long':['COIN00','COIN02'],'short':['COIN01']})
        assert execution_key(other_base) != execution_key(config)  # Equal masks, different selected file.


@pytest.mark.parametrize('kind',['all','file'])
@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
def test_actual_registry_and_candidate_planner_keep_prepared_universe(tmp_path, kind, strategy_kind):
    path=tmp_path/'coins.json';path.write_text('["COIN00","COIN02"]')
    config,candles,markets,btc,timestamps=prepared('all' if kind=='all' else 'coins.json',tmp_path,strategy_kind)
    manager=SharedArrayManager()
    try:
        specs=[manager.create_from(arr)[0] for arr in (candles,btc)]
        evaluator=Evaluator({'binance':specs[0]},{'binance':specs[1]}, {'binance':markets}, config,timestamps={'binance':timestamps})
        with NativeDatasetRegistry(evaluator,standalone_candle_coins={'binance':tuple(config['backtest']['coins']['binance'])}) as registry:
            vector=config_to_individual(evaluator.config,evaluator.bounds,optimization_shape=evaluator.optimization_shape)
            path.unlink()
            plan=registry.planner.prepare('candidate',vector)
            assert plan.requests
            registered=[]
            registry.register(SimpleNamespace(register_dataset=lambda name,dataset:registered.append(dataset)))
            assert registered
            assert masks(registered[0].execution_config(),candles,markets,btc,timestamps)=={'long':[True,False,True],'short':[False,True,False]}
            if kind == 'file':
                other = tmp_path/'other'
                other.mkdir()
                (other/'coins.json').write_text('["COIN00","COIN02"]')
                evaluator.config['live']['base_config_path'] = str(other/'input.json')
                assert registry.planner.prepare('same-selected-file-new-context', vector).requests
                evaluator.config['bot']['global']['universe']['approved_coins'] = str(other/'coins.json')
                with pytest.raises(ValueError, match='prepared long approved_coins membership'):
                    registry.planner.prepare('unprepared-source-edit', vector)
                materialize_side_universe_sources(evaluator.config, 'approved_coins', {'long':['COIN00','COIN02'],'short':['COIN01']})
                with pytest.raises(ValueError, match='dataset-owned execution inputs'):
                    registry.planner.prepare('different-file-base', vector)
    finally:
        manager.cleanup()


@pytest.mark.parametrize('kind', ['all', 'file'])
@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
def test_suite_planner_preserves_frozen_inherited_membership_after_source_override(tmp_path, kind, strategy_kind):
    from optimize import SuiteEvaluator
    path = tmp_path / 'coins.json'
    path.write_text('["COIN00","COIN02"]')
    config, candles, markets, btc, timestamps = prepared('all' if kind == 'all' else 'coins.json', tmp_path,strategy_kind)
    context_config = deepcopy(config)
    context_config['bot']['global']['universe']['approved_coins'] = 'all' if kind == 'all' else 'context.json'
    materialize_side_universe_sources(context_config, 'approved_coins', {'long': ['COIN00', 'COIN02'], 'short': ['COIN01']})
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(arr)[0] for arr in (candles, btc)]
        base = Evaluator({'binance': specs[0]}, {'binance': specs[1]}, {'binance': markets}, config, timestamps={'binance': timestamps})
        context = SimpleNamespace(
            label='scenario', exchanges=['binance'], config=context_config,
            msss=base.msss, timestamps={'binance': timestamps},
            overrides={'bot.global.universe.approved_coins': context_config['bot']['global']['universe']['approved_coins']},
            hlcvs_specs=base.hlcvs_specs, btc_usd_specs=base.btc_usd_specs,
            master_hlcvs_specs=None, master_btc_specs=None, time_slice=None,
            coin_slice_indices=None, coin_indices={'binance': None},
            candle_coins={'binance': tuple(config['backtest']['coins']['binance'])},
        )
        suite = SuiteEvaluator(base, [context], {'default': 'mean'})
        path.unlink()  # Prepared contexts must not re-read external policy at candidate replay.
        with NativeDatasetRegistry(suite) as registry:
            vector = config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
            assert registry.planner.prepare('candidate', vector).requests
            candidate = suite.build_scenario_candidate_config(config, context)
            assert masks(candidate, candles, markets, btc, timestamps) == {'long': [True, False, True], 'short': [False, True, False]}
            assert universe_source_view(candidate)['bot']['long']['universe']['approved_coins'] is None
            # Equal membership does not erase an explicit pin's distinct authority.
            original_context = deepcopy(context.config)
            context.config['bot']['long']['universe']['approved_coins'] = ['COIN00', 'COIN02']
            with pytest.raises(ValueError, match='dataset-owned execution inputs'):
                registry.planner.prepare('different-source-policy', vector)
            context.config = deepcopy(original_context)
            # The same union with different directional masks is another dataset.
            materialize_side_universe_sources(context.config, 'approved_coins', {
                'long': ['COIN01'], 'short': ['COIN00', 'COIN02'],
            })
            with pytest.raises(ValueError, match='dataset-owned execution inputs'):
                registry.planner.prepare('different-membership', vector)
    finally:
        manager.cleanup()


@pytest.mark.parametrize('change', ['source', 'shared_source', 'membership', 'subclass'])
def test_fast_native_projection_checks_internal_source_authority(tmp_path, change):
    from config.universe import ResolvedCoinList
    from optimization.native_planning import _ExecutionProjection
    config, *_ = prepared('all', tmp_path)
    projection = _ExecutionProjection(config, execution_key(config))
    assert projection.matches(deepcopy(config))
    changed = deepcopy(config)
    value = changed['bot']['long']['universe']['approved_coins']
    if change == 'source':
        value.source = list(value)  # Equal membership becomes an explicit pin.
    elif change == 'shared_source':
        value.shared_source = ['COIN00', 'COIN02']
    elif change == 'membership':
        value[:] = ['COIN01']
    else:
        class Unsupported(ResolvedCoinList):
            pass
        changed['bot']['long']['universe']['approved_coins'] = Unsupported(value, value.source, value.shared_source)
    assert not projection.matches(changed)
