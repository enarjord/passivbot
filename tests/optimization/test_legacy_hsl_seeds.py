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


def legacy(mode='coin', policy='always'):
    from test_v9_config_migration import legacy as released_v8
    cfg = released_v8()
    cfg['live']['hsl_signal_mode'] = mode
    if mode == 'unified':
        cfg['bot']['hsl'] = deepcopy(cfg['bot']['long']['hsl'])
        for side in ('long', 'short'):
            cfg['optimize']['bounds'][side].pop('hsl', None)
    block = cfg['bot']['hsl'] if mode == 'unified' else cfg['bot']['long']['hsl']
    block.update(enabled=True, restart_after_red_policy=policy, red_threshold=0.234)
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
@pytest.mark.parametrize('version',[None,'v8.6.0','v9.0.1','v9.1.999'])
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
    scope=('bot',"portfolio",'hsl','drawdown_ratio_threshold') if mode=='unified' else ('bot','long','hsl','drawdown_ratio_threshold')
    values,count=configs_to_individuals_streaming([seed],[Bound.from_config('threshold',[0.1,0.4])],sig_digits=6,key_paths=[('threshold',scope)])
    assert count==1 and values==[(0.234,)]
    assert seed['bot']["portfolio"]['hsl']['scope']=={'unified':'portfolio','pside':'position_side'}.get(mode,mode)


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
    base['bot']['long']['hsl'].update(enabled=True,cooldown_after_red_minutes=12.5)
    cfg=deepcopy(base);cfg['bot']['long']['hsl'].update(enabled=False,drawdown_ratio_threshold=2.0)
    seeds=extract_configs(str(save(tmp_path,cfg,'json')))
    keys=[('long_hsl_red_threshold',('bot','long','hsl','drawdown_ratio_threshold'))]
    values,count=configs_to_individuals_streaming(seeds,[Bound.from_config(keys[0][0],[0.1,0.4])],sig_digits=6,key_paths=keys)
    assert count==1 and values==[(0.4,)]
    result=individual_to_config(values[0],{},[],base,key_paths=keys)
    assert result['bot']['long']['hsl']['enabled'] is True
    assert result['bot']['long']['hsl']['cooldown_after_red_minutes']==12.5
    assert result['bot']['long']['hsl']['drawdown_ratio_threshold']==0.4


@pytest.mark.parametrize('field',['drawdown_ratio_threshold','memory_ema_span_minutes','drawdown_ratio_ema_span_minutes','cooldown_after_red_minutes'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),'NaN'])
@pytest.mark.parametrize('artifact',['json','pareto'])
def test_nonfinite_current_numeric_seed_is_skipped(field,value,artifact,tmp_path,caplog):
    cfg=get_template_config()
    cfg['bot']['long']['hsl'][field]=str(value) if artifact=='json' else value
    assert extract_configs(str(save(tmp_path,cfg,artifact)))==[]
    assert 'risk optimizer seed value' in caplog.text


@pytest.mark.parametrize('artifact', ['json', 'pareto'])
@pytest.mark.parametrize(('authored', 'expected'), [(1440.5, 1440.5), (0.5, 1000.0), (100000.0, 50000.0)])
def test_current_slow_span_seed_survives_extraction_and_normal_bounds_clamp(tmp_path, artifact, authored, expected):
    cfg = get_template_config()
    cfg['bot']["portfolio"]['unstuck']['memory_ema_span_minutes'] = authored
    seeds = extract_configs(str(save(tmp_path, cfg, artifact)))
    keys = [('unstuck_memory_ema_span_minutes', ('bot', "portfolio", 'unstuck', 'memory_ema_span_minutes'))]
    values, count = configs_to_individuals_streaming(seeds,
        [Bound.from_config(keys[0][0], [1000.0, 50000.0])], sig_digits=7, key_paths=keys)
    assert count == 1 and values == [(expected,)]
    assert 'realized_loss_gate_lookback_minutes' not in seeds[0].get('live', {})


def test_old_v9_active_gate_seed_requires_explicit_migration(tmp_path):
    from optimize import StartingConfigSchemaError

    cfg = get_template_config()
    cfg['config_version'] = 'v9.0.0'
    cfg['live']['max_realized_loss_pct'] = 0.1
    # Even a sparse old export without its rolling duration has old semantics.
    with pytest.raises(StartingConfigSchemaError, match='migrate-config-v9'):
        extract_configs(str(save(tmp_path, cfg, 'json')))


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', True, 'bad'])
def test_invalid_current_live_numeric_seed_is_not_replaced_with_default(tmp_path, value, caplog):
    cfg = get_template_config()
    cfg['bot']["portfolio"]['unstuck']['memory_ema_span_minutes'] = value
    assert extract_configs(str(save(tmp_path, cfg, 'json'))) == []
    assert 'risk optimizer seed value' in caplog.text


def test_anchored_fine_tune_retains_tunable_live_slow_span_suggestions(tmp_path):
    from optimize import install_anchored_fine_tune_plan, iter_anchored_fine_tune_seed_configs
    from optimization.shape import build_optimization_shape

    config = prepare_config(get_template_config(), verbose=False)
    config['optimize']['bounds'] = {
        "portfolio": {'unstuck': {'memory_ema_span_minutes': [1000.0, 50000.0]}},
        'long': {'hsl': {'drawdown_ratio_threshold': [0.1, 0.4]}},
    }
    for side in ('long', 'short'):
        for field in ('max_positions', 'total_wallet_exposure_limit'):
            value = config['bot'][side]['risk'][field]
            config['optimize']['bounds'].setdefault(side, {}).setdefault('risk', {})[field] = [value, value]
    config['optimize']['fixed_params'] = []
    directory = tmp_path / 'anchors'
    directory.mkdir()
    for index, span in enumerate((1440.5, 2880.25)):
        seed = deepcopy(config)
        seed['bot']["portfolio"]['unstuck']['memory_ema_span_minutes'] = span
        (directory / f'anchor-{index}.json').write_text(json.dumps(seed))
    install_anchored_fine_tune_plan(config, ['bot.portfolio.unstuck.memory_ema_span_minutes'], str(directory))
    shape = build_optimization_shape(config)
    seeds = list(iter_anchored_fine_tune_seed_configs(config))
    assert [seed['bot']["portfolio"]['unstuck']['memory_ema_span_minutes'] for seed in seeds] == [1440.5, 2880.25]
    values, count = configs_to_individuals_streaming(seeds, shape.bounds, sig_digits=7, optimization_shape=shape)
    assert count == 2
    assert sorted(tuple(row) for row in values) == [(0.0, 1440.5), (1.0, 2880.25)]


RISK_SPAN_PATHS = [
    ("bot", side, "hsl", field)
    for side in ("long", "short")
    for field in ("memory_ema_span_minutes", "drawdown_ratio_ema_span_minutes")
] + [
    ("bot", "portfolio", "hsl", field)
    for field in ("memory_ema_span_minutes", "drawdown_ratio_ema_span_minutes")
] + [
    ("bot", "portfolio", "unstuck", "memory_ema_span_minutes"),
    ("bot", "portfolio", "loss_gate", "memory_ema_span_minutes"),
]


def span_seed_config(path, value):
    mode = "portfolio" if path[1:3] == ("portfolio", "hsl") else "coin"
    config = generated_template(get_template_config(), mode)
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return config


@pytest.mark.parametrize("path", RISK_SPAN_PATHS)
@pytest.mark.parametrize("authored,expected", [(1440.5, 1440.5), (0.5, 1000.0), (100000.0, 50000.0)])
def test_independent_risk_span_seed_preserves_suggestion_until_bounds_clamp(tmp_path, path, authored, expected):
    config = span_seed_config(path, authored)
    before = deepcopy(config)
    seeds = extract_configs(str(save(tmp_path, config, "json")))
    values, count = configs_to_individuals_streaming(
        seeds, [Bound.from_config("span", [1000.0, 50000.0])],
        sig_digits=7, key_paths=[("span", path)],
    )
    assert count == 1 and values == [(expected,)]
    assert config == before


@pytest.mark.parametrize("path", RISK_SPAN_PATHS)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", True, "bad"])
def test_invalid_independent_risk_span_seed_is_skipped_without_default_substitution(tmp_path, path, value, caplog):
    config = span_seed_config(path, value)
    assert extract_configs(str(save(tmp_path, config, "json"))) == []
    assert "risk optimizer seed value" in caplog.text


def test_seed_unknown_bot_field_is_rejected_before_formatting_can_drop_it(tmp_path):
    config = get_template_config()
    config["bot"]["long"]["hsl"]["red_thresold"] = 0.2
    with pytest.raises(StartingConfigSchemaError, match="red_thresold.*unknown"):
        extract_configs(str(save(tmp_path, config, "json")))


@pytest.mark.parametrize("path", RISK_SPAN_PATHS)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", True, "bad"])
def test_direct_in_memory_seed_cannot_bypass_risk_numeric_validation(path, value):
    config = span_seed_config(path, value)
    with pytest.raises(ValueError, match="risk optimizer seed value"):
        _build_starting_seed_config(config)
    values, count = configs_to_individuals_streaming(
        [config], [Bound.from_config("span", [1000.0, 50000.0])],
        sig_digits=7, key_paths=[("span", path)],
    )
    assert values == [] and count == 1  # The count records attempted inputs, including skips.


def test_direct_in_memory_seed_cannot_bypass_unknown_field_validation():
    config = get_template_config()
    config["bot"]["long"]["hsl"]["red_thresold"] = 0.2
    with pytest.raises(StartingConfigSchemaError, match="red_thresold.*unknown"):
        _build_starting_seed_config(config)


@pytest.mark.parametrize("scope", ["portfolio", "PORTFOLIO", " portfolio "])
def test_portfolio_scope_normalization_preserves_numeric_suggestions(scope):
    config = generated_template(get_template_config(), "portfolio")
    config["bot"]["portfolio"]["hsl"].update(scope=scope, memory_ema_span_minutes=0.5)
    seed = _build_starting_seed_config(config)
    assert seed["bot"]["portfolio"]["hsl"]["memory_ema_span_minutes"] == 0.5
    assert seed["bot"]["portfolio"]["hsl"]["scope"] == "portfolio"
