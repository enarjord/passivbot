"""Old risk policy cannot enter v9 optimization through seed extraction."""
from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.hsl import generated_template
from config.schema import CONFIG_SCHEMA_VERSION, get_template_config
from optimization.bounds import Bound
from optimize import (
    StartingConfigSchemaError, _build_starting_seed_config,
    configs_to_individuals_streaming, extract_configs,
)
from tools.migrate_config_v9 import migrate


def legacy(mode='coin',policy='always'):
    cfg=generated_template(get_template_config(),'portfolio' if mode=='unified' else mode)
    cfg['config_version']='v8.6.0'
    cfg['live']['hsl_signal_mode']=mode
    for field in ['lookback_ema_span_minutes','realized_loss_gate_lookback_minutes','risk_bootstrap_max_lookback_minutes']:
        cfg['live'].pop(field,None)
    cfg['live']['pnls_max_lookback_days']=30.0
    block=cfg['bot']['hsl'] if mode=='unified' else cfg['bot']['long']['hsl']
    block.update(enabled=True,restart_after_red_policy=policy,red_threshold=0.234)
    return cfg


def save(tmp_path,cfg,artifact):
    path=tmp_path/('seed.json' if artifact=='json' else 'seed_pareto.txt')
    path.write_text(json.dumps({'config':cfg,'fitness':123}))
    return path


@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('policy',['threshold','always','never'])
@pytest.mark.parametrize('artifact',['json','pareto'])
@pytest.mark.parametrize('enabled',[False,True])
def test_old_schema_seed_requires_migration_even_when_disabled(mode,policy,artifact,enabled,tmp_path):
    cfg=legacy(mode,policy)
    block=cfg['bot']['hsl'] if mode=='unified' else cfg['bot']['long']['hsl']
    block['enabled']=enabled
    original=deepcopy(cfg);path=save(tmp_path,cfg,artifact);before=path.read_bytes()
    with pytest.raises(StartingConfigSchemaError,match='migrate-config-v9'):
        extract_configs(str(path))
    with pytest.raises(ValueError,match='migrate-config-v9'):
        prepare_config(cfg,verbose=False)
    assert path.read_bytes()==before and cfg==original


@pytest.mark.parametrize('artifact',['json','pareto'])
@pytest.mark.parametrize('version',[None,'v8.6.0','v9.1.0'])
def test_seed_schema_errors_are_fatal_instead_of_silently_replaced_by_random_seeds(tmp_path,artifact,version):
    cfg=get_template_config()
    if version is None:cfg.pop('config_version')
    else:cfg['config_version']=version
    path=save(tmp_path,cfg,artifact)
    with pytest.raises(StartingConfigSchemaError):
        extract_configs(str(path))


@pytest.mark.parametrize('mode',['coin','pside','unified'])
@pytest.mark.parametrize('policy',['always','never'])
def test_explicitly_migrated_seed_retains_numeric_suggestions_and_discards_old_fitness(tmp_path,mode,policy):
    cfg=legacy(mode,policy)
    if policy=='never':
        with pytest.raises(ValueError,match='acknowledge-never-removal'):
            migrate(cfg,memory_span_minutes=43200.0)
    migrated=migrate(cfg,memory_span_minutes=43200.0,acknowledge_never_removal=policy=='never').config
    path=save(tmp_path,migrated,'json')
    seed=_build_starting_seed_config(extract_configs(str(path))[0])
    assert seed['config_version']==CONFIG_SCHEMA_VERSION
    assert 'fitness' not in seed
    scope=('bot','hsl','red_threshold') if mode=='unified' else ('bot','long','hsl','red_threshold')
    values,count=configs_to_individuals_streaming([seed],[Bound.from_config('threshold',[0.1,0.4])],sig_digits=6,key_paths=[('threshold',scope)])
    assert count==1 and values==[(0.234,)]
    assert seed['live']['hsl_signal_mode']==('portfolio' if mode=='unified' else mode)


def test_bot_only_old_policy_cannot_bypass_migration():
    bot=deepcopy(get_template_config()['bot'])
    bot['long']['hsl_restart_after_red_policy']='never'
    for value in [bot,{'bot':bot}]:
        with pytest.raises(StartingConfigSchemaError,match='migrate-config-v9'):
            _build_starting_seed_config(value)
        with pytest.raises(StartingConfigSchemaError,match='migrate-config-v9'):
            configs_to_individuals_streaming([value],[],key_paths=[])


def test_current_numeric_seed_clamps_under_the_main_finite_policy(tmp_path):
    from optimize import individual_to_config
    base=prepare_config(generated_template(get_template_config()),verbose=False)
    base['bot']['long']['hsl'].update(enabled=True,cooldown_minutes_after_red=12.5)
    cfg=deepcopy(base);cfg['bot']['long']['hsl'].update(enabled=False,red_threshold=2.0)
    seeds=extract_configs(str(save(tmp_path,cfg,'json')))
    keys=[('long_hsl_red_threshold',('bot','long','hsl','red_threshold'))]
    values,count=configs_to_individuals_streaming(seeds,[Bound.from_config(keys[0][0],[0.1,0.4])],sig_digits=6,key_paths=keys)
    assert count==1 and values==[(0.4,)]
    result=individual_to_config(values[0],{},[],base,key_paths=keys)
    assert result['bot']['long']['hsl']['enabled'] is True
    assert result['bot']['long']['hsl']['cooldown_minutes_after_red']==12.5
    assert result['bot']['long']['hsl']['red_threshold']==0.4


@pytest.mark.parametrize('field',['red_threshold','ema_span_minutes','cooldown_minutes_after_red'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),'NaN'])
@pytest.mark.parametrize('artifact',['json','pareto'])
def test_nonfinite_current_numeric_seed_is_skipped(field,value,artifact,tmp_path,caplog):
    cfg=get_template_config()
    cfg['bot']['long']['hsl'][field]=str(value) if artifact=='json' else value
    assert extract_configs(str(save(tmp_path,cfg,artifact)))==[]
    assert 'non-finite HSL optimizer seed value' in caplog.text


@pytest.mark.parametrize('artifact', ['json', 'pareto'])
@pytest.mark.parametrize(('authored', 'expected'), [(1440.5, 1440.5), (0.5, 1000.0), (100000.0, 50000.0)])
def test_current_slow_span_seed_survives_extraction_and_normal_bounds_clamp(tmp_path, artifact, authored, expected):
    cfg = get_template_config()
    cfg['live']['lookback_ema_span_minutes'] = authored
    cfg['live']['realized_loss_gate_lookback_minutes'] = -1.0
    seeds = extract_configs(str(save(tmp_path, cfg, artifact)))
    keys = [('live_lookback_ema_span_minutes', ('live', 'lookback_ema_span_minutes'))]
    values, count = configs_to_individuals_streaming(seeds,
        [Bound.from_config(keys[0][0], [1000.0, 50000.0])], sig_digits=7, key_paths=keys)
    assert count == 1 and values == [(expected,)]
    assert 'realized_loss_gate_lookback_minutes' not in seeds[0]['live']


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', True, 'bad'])
def test_invalid_current_live_numeric_seed_is_not_replaced_with_default(tmp_path, value, caplog):
    cfg = get_template_config()
    cfg['live']['lookback_ema_span_minutes'] = value
    assert extract_configs(str(save(tmp_path, cfg, 'json'))) == []
    assert 'live optimizer seed value' in caplog.text


def test_anchored_fine_tune_retains_tunable_live_slow_span_suggestions(tmp_path):
    from optimize import install_anchored_fine_tune_plan, iter_anchored_fine_tune_seed_configs
    from optimization.shape import build_optimization_shape

    config = prepare_config(get_template_config(), verbose=False)
    config['optimize']['bounds'] = {
        'live_lookback_ema_span_minutes': [1000.0, 50000.0],
        'long_hsl_red_threshold': [0.1, 0.4],
    }
    for side in ('long', 'short'):
        for field in ('n_positions', 'total_wallet_exposure_limit'):
            value = config['bot'][side]['risk'][field]
            config['optimize']['bounds'][f'{side}_{field}'] = [value, value]
    config['optimize']['fixed_params'] = []
    directory = tmp_path / 'anchors'
    directory.mkdir()
    for index, span in enumerate((1440.5, 2880.25)):
        seed = deepcopy(config)
        seed['live']['lookback_ema_span_minutes'] = span
        (directory / f'anchor-{index}.json').write_text(json.dumps(seed))
    install_anchored_fine_tune_plan(config, ['live.lookback_ema_span_minutes'], str(directory))
    shape = build_optimization_shape(config)
    seeds = list(iter_anchored_fine_tune_seed_configs(config))
    assert [seed['live']['lookback_ema_span_minutes'] for seed in seeds] == [1440.5, 2880.25]
    values, count = configs_to_individuals_streaming(seeds, shape.bounds, sig_digits=7, optimization_shape=shape)
    assert count == 2
    assert sorted(tuple(row) for row in values) == [(0.0, 1440.5), (1.0, 2880.25)]
