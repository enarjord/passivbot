"""Only released-v8 approval authority is converted to the final shared layout."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from config import prepare_config
from config.migrations.released_v8 import get_released_v8_defaults
from config.universe import resolve_universe_sources, universe_source_view
from tools.migrate_config_v9 import migrate
from suite_runner import apply_scenario_overrides
from optimization.warmup import _apply_config_overrides


def released(kind):
    c=get_released_v8_defaults(strategy_kind=kind)
    c['backtest']['visible_metrics']=[x for x in c['backtest']['visible_metrics'] if not x.endswith('_w')]
    c['optimize']['scoring']=[x for x in c['optimize']['scoring'] if not x['metric'].endswith('_w')]
    return c


@pytest.mark.parametrize('kind',['trailing_martingale','ema_anchor'])
@pytest.mark.parametrize('value,shared,sides',[
    (['BTC','ETH'],['BTC','ETH'],{'long':None,'short':None}),
    ('all','all',{'long':None,'short':None}),
    ({'long':['BTC'],'short':['BTC']},[],{'long':['BTC'],'short':['BTC']}),
    ({'long':None,'short':[]},[],{'long':[],'short':[]}),
    ({'long':['BTC']},[],{'long':['BTC'],'short':[]}),
])
def test_root_mapping_preserves_shared_intent_explicit_pins_and_empty_sides(kind,value,shared,sides):
    c=released(kind);c['live']['approved_coins']=deepcopy(value);before=deepcopy(c)
    result=migrate(c,memory_span_minutes=123.5)
    out=result.config
    assert out['bot']['global']['universe']['approved_coins']==shared
    assert {side:out['bot'][side]['universe']['approved_coins'] for side in sides}==sides
    assert c==before
    assert any('approved_coins' in row['path'] for row in result.changes)
    assert migrate(out).config==out


@pytest.mark.parametrize('owner',['scenario','fixed'])
@pytest.mark.parametrize('rootvalue',[['BTC'],{'long':['BTC'],'short':['ETH']}])
@pytest.mark.parametrize('replacement',[['SOL'],{'long':['XRP'],'short':None}])
def test_whole_released_replacement_resets_preexisting_shared_or_side_authority(owner,rootvalue,replacement):
    c=released('trailing_martingale');c['live']['approved_coins']=rootvalue
    patch={'live.approved_coins':replacement}
    if owner=='fixed':c['optimize']['fixed_runtime_overrides']=patch
    else:c['backtest']['scenarios']=[{'label':'replace','overrides':patch}]
    out=migrate(c,memory_span_minutes=123.5).config
    effective=prepare_config(out,verbose=False)
    if owner=='fixed':
        _apply_config_overrides(effective,effective['optimize']['fixed_runtime_overrides'])
    else:
        apply_scenario_overrides(effective,effective['backtest']['scenarios'][0]['overrides'])
    expected={'long':['SOL'],'short':['SOL']} if isinstance(replacement,list) else {'long':['XRP'],'short':[]}
    assert resolve_universe_sources(effective,'approved_coins')[0]==expected


@pytest.mark.parametrize('kind',['trailing_martingale','ema_anchor'])
@pytest.mark.parametrize('wrapped',[False,True])
@pytest.mark.parametrize('filename',['coins.json','coins'])
def test_released_directional_file_remains_reloadable_and_explicit(tmp_path,kind,wrapped,filename):
    p=tmp_path/filename;p.write_text('{"long":["BTC"],"short":["ETH"]}')
    c=released(kind);old_source=[filename] if wrapped else filename;c['live']['approved_coins']=old_source;before=deepcopy(c);bytes_before=p.read_bytes()
    out=migrate(c,memory_span_minutes=123.5,base_config_path=str(tmp_path/'config.json')).config
    assert out['bot']['global']['universe']['approved_coins']==[]
    expected_file = str(p) if not Path(filename).suffix else filename
    expected_source = [expected_file] if wrapped else expected_file
    assert all(out['bot'][side]['universe']['approved_coins']==expected_source for side in ('long','short'))
    effective=prepare_config(out,base_config_path=str(tmp_path/'config.json'),verbose=False)
    assert resolve_universe_sources(effective,'approved_coins')[0]=={'long':['BTC'],'short':['ETH']}
    from tools.clean_config import render_source
    destination = tmp_path / 'moved' / 'v9.json'
    destination.parent.mkdir()
    (destination.parent / filename).write_text('{"long":["DECOY"],"short":[]}')
    relocated = json.loads(render_source(json.dumps(out), source=tmp_path/'config.json',
                                        destination=destination, mode='compact'))
    moved = prepare_config(relocated, base_config_path=str(destination), verbose=False)
    assert resolve_universe_sources(moved, 'approved_coins')[0] == {'long':['BTC'],'short':['ETH']}
    p.write_text('{"long":["SOL"],"short":[]}')
    assert resolve_universe_sources(effective,'approved_coins')[0]=={'long':['SOL'],'short':[]}
    assert c==before and bytes_before==b'{"long":["BTC"],"short":["ETH"]}'


def test_released_singleton_all_stays_shared_despite_same_named_file(tmp_path):
    (tmp_path/'all').write_text('["ETH"]')
    raw = released('trailing_martingale')
    raw['live']['approved_coins'] = ['all']
    result = migrate(raw, memory_span_minutes=123.5, base_config_path=str(tmp_path/'input.json'))
    assert result.config['bot']['global']['universe']['approved_coins'] == ['all']
    assert all(result.config['bot'][side]['universe']['approved_coins'] is None for side in ('long','short'))


ENCODED_RELEASED_SOURCES = [
    '{"long":["BTC"],"short":[]}',
    '{"long":null,"short":[]}',
    '["BTC","ETH"]',
    ['{"long":["BTC"],"short":["ETH"]}'],
    {'long':'["BTC","ETH"]','short':'[]'},
]


@pytest.mark.parametrize('kind', ['trailing_martingale', 'ema_anchor'])
@pytest.mark.parametrize('field', ['approved_coins', 'ignored_coins'])
@pytest.mark.parametrize('value', ENCODED_RELEASED_SOURCES)
def test_released_encoded_inline_coin_sources_keep_legacy_membership_and_report(kind, field, value):
    from utils import normalize_coins_source
    raw = released(kind)
    raw['live'][field] = deepcopy(value)
    before = deepcopy(raw)
    expected = normalize_coins_source(value, allow_all=field == 'approved_coins', resolve_files=False)
    result = migrate(raw, memory_span_minutes=123.5)
    effective = prepare_config(result.config, verbose=False)
    assert resolve_universe_sources(effective, field)[0] == expected
    assert any('decoded released' in row['reason'] for row in result.changes)
    assert raw == before
    assert migrate(result.config).config == result.config


@pytest.mark.parametrize('owner', ['scenario', 'fixed'])
@pytest.mark.parametrize('field', ['approved_coins', 'ignored_coins'])
@pytest.mark.parametrize('selected_side', [None, 'long', 'short'])
@pytest.mark.parametrize('value', ['["BTC","ETH"]', '{"long":["BTC"],"short":["ETH"]}', '{"long":null,"short":[]}'])
def test_released_encoded_override_sources_preserve_replacement_scope(owner, field, selected_side, value):
    from utils import normalize_coins_source
    raw = released('trailing_martingale')
    initial = {'long':['OLD'], 'short':['OTHER']}
    raw['live'][field] = deepcopy(initial)
    selector = f'live.{field}' + (f'.{selected_side}' if selected_side else '')
    replacement = {selector:value}
    if owner == 'fixed':
        raw['optimize']['fixed_runtime_overrides'] = replacement
    else:
        raw['backtest']['scenarios'] = [{'label':'encoded', 'overrides':replacement}]
    old_effective = deepcopy(initial)
    if selected_side:
        old_effective[selected_side] = value
    else:
        old_effective = value
    expected = normalize_coins_source(old_effective, allow_all=field == 'approved_coins', resolve_files=False)
    before = deepcopy(raw)
    result = migrate(raw, memory_span_minutes=123.5)
    effective = prepare_config(result.config, verbose=False)
    if owner == 'fixed':
        _apply_config_overrides(effective, effective['optimize']['fixed_runtime_overrides'])
    else:
        apply_scenario_overrides(effective, effective['backtest']['scenarios'][0]['overrides'])
    assert resolve_universe_sources(effective, field)[0] == expected
    assert any('decoded released' in row['reason'] for row in result.changes)
    assert raw == before


@pytest.mark.parametrize('owner', ['root', 'scenario', 'fixed'])
@pytest.mark.parametrize('field', ['approved_coins', 'ignored_coins'])
@pytest.mark.parametrize('directional', [False, True])
def test_released_encoded_file_shorthand_retains_dynamic_binding(tmp_path, owner, field, directional):
    selected = tmp_path/'coinlist'
    selected.write_text('["BTC"]')
    value = json.dumps({'long':'coinlist','short':[]}) if directional else json.dumps(['coinlist'])
    raw = released('trailing_martingale')
    if owner == 'root':
        raw['live'][field] = value
    elif owner == 'scenario':
        raw['backtest']['scenarios'] = [{'label':'file','overrides':{f'live.{field}':value}}]
    else:
        raw['optimize']['fixed_runtime_overrides'] = {f'live.{field}':value}
    before = deepcopy(raw)
    result = migrate(raw, memory_span_minutes=123.5, base_config_path=str(tmp_path/'input.json'))
    effective = prepare_config(result.config, base_config_path=str(tmp_path/'input.json'), verbose=False)
    if owner == 'scenario':
        apply_scenario_overrides(effective, effective['backtest']['scenarios'][0]['overrides'])
    elif owner == 'fixed':
        _apply_config_overrides(effective, effective['optimize']['fixed_runtime_overrides'])
    expected_short = [] if directional else ['BTC']
    assert resolve_universe_sources(effective, field)[0] == {'long':['BTC'],'short':expected_short}
    selected.write_text('["ETH"]')
    assert resolve_universe_sources(effective, field)[0] == {'long':['ETH'],'short':[] if directional else ['ETH']}
    assert raw == before
    assert any('decoded released' in row['reason'] for row in result.changes)


@pytest.mark.parametrize('field', ['approved_coins', 'ignored_coins'])
def test_released_encoded_directional_null_reports_final_empty_selection(field):
    from config.migrations.shared_policy import normalize_released_coin_source
    events = []
    result = normalize_released_coin_source('{"long":null,"short":["ETH"]}',
        field=field, side='long', path=f'live.{field}.long',
        record=lambda *event: events.append(event))
    assert result == []
    assert events[-1][0] == f'live.{field}.long'
    assert events[-1][2] == []


@pytest.mark.parametrize('kind', ['trailing_martingale', 'ema_anchor'])
@pytest.mark.parametrize('value', ['BTC', 'BTC,ETH'])
@pytest.mark.parametrize('owner', ['scenario', 'fixed'])
def test_released_nonfile_scalar_approval_becomes_shared_and_future_shared_edits_apply(kind, value, owner):
    raw = released(kind)
    raw['live']['approved_coins'] = value
    before = deepcopy(raw)
    result = migrate(raw, memory_span_minutes=123.5)
    assert result.config['bot']['global']['universe']['approved_coins'] == value.split(',')
    assert all(result.config['bot'][side]['universe']['approved_coins'] is None for side in ('long','short'))
    effective = prepare_config(result.config, verbose=False)
    patch = {'bot.global.universe.approved_coins':['SOL']}
    if owner == 'scenario':
        apply_scenario_overrides(effective, patch)
    else:
        _apply_config_overrides(effective, patch)
    assert resolve_universe_sources(effective, 'approved_coins')[0] == {'long':['SOL'],'short':['SOL']}
    assert raw == before


@pytest.mark.parametrize('path', ['./missing', 'missing.json', '/not-present/coins.txt'])
def test_released_missing_explicit_file_source_stays_directional(path):
    from config.migrations.v9 import migrate_v8_to_v9
    raw = released('trailing_martingale')
    raw['live']['approved_coins'] = path
    result = migrate_v8_to_v9(raw, memory_span_minutes=123.5).config
    assert result['bot']['global']['universe']['approved_coins'] == []
    assert all(result['bot'][side]['universe']['approved_coins'] == path for side in ('long','short'))


@pytest.mark.parametrize('owner', ['scenario', 'fixed'])
@pytest.mark.parametrize('value', ['SOL', 'SOL,BNB'])
def test_released_whole_scalar_override_resets_directional_pins_to_shared(owner, value):
    raw = released('trailing_martingale')
    raw['live']['approved_coins'] = {'long':['BTC'], 'short':['ETH']}
    override = {'live.approved_coins':value}
    if owner == 'scenario':
        raw['backtest']['scenarios'] = [{'label':'scalar','overrides':override}]
    else:
        raw['optimize']['fixed_runtime_overrides'] = override
    result = migrate(raw, memory_span_minutes=123.5).config
    effective = prepare_config(result, verbose=False)
    if owner == 'scenario':
        apply_scenario_overrides(effective, effective['backtest']['scenarios'][0]['overrides'])
    else:
        _apply_config_overrides(effective, effective['optimize']['fixed_runtime_overrides'])
    assert resolve_universe_sources(effective, 'approved_coins')[0] == {'long':value.split(','),'short':value.split(',')}
    _apply_config_overrides(effective, {'bot.global.universe.approved_coins':['XRP']})
    assert resolve_universe_sources(effective, 'approved_coins')[0] == {'long':['XRP'],'short':['XRP']}


@pytest.mark.parametrize('wrapped', [False, True])
def test_released_whitespace_file_binding_uses_resolver_and_stays_directional(tmp_path, monkeypatch, wrapped):
    monkeypatch.chdir(tmp_path)
    selected = tmp_path/'coinlist'
    selected.write_text('["BTC"]')
    value = [' coinlist '] if wrapped else ' coinlist '
    raw = released('trailing_martingale')
    raw['live']['approved_coins'] = value
    before = deepcopy(raw)
    effective = prepare_config(migrate(raw, memory_span_minutes=123.5).config, verbose=False)
    restored = universe_source_view(effective)
    assert restored['bot']['global']['universe']['approved_coins'] == []
    for side in ('long','short'):
        assert restored['bot'][side]['universe']['approved_coins'] == ([str(selected)] if wrapped else str(selected))
    _apply_config_overrides(effective, {'bot.global.universe.approved_coins':['ETH']})
    assert resolve_universe_sources(effective, 'approved_coins')[0] == {'long':['BTC'],'short':['BTC']}
    selected.write_text('["SOL"]')
    assert resolve_universe_sources(effective, 'approved_coins')[0] == {'long':['SOL'],'short':['SOL']}
    assert raw == before


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('fallback', ['cwd', 'repo'])
def test_released_fallback_coin_file_binding_preserves_reloadable_symlink(tmp_path, monkeypatch, wrapped, fallback):
    import utils
    repo, cwd, configs = (tmp_path/name for name in ('repo','cwd','configs'))
    for directory in (repo, cwd, configs):
        directory.mkdir()
    monkeypatch.setattr(utils, 'PROJECT_ROOT', repo)
    monkeypatch.chdir(cwd)
    selected = (cwd if fallback == 'cwd' else repo)/'coinlist'
    first, second = selected.with_name('first.json'), selected.with_name('second.json')
    first.write_text('["BTC"]')
    second.write_text('["ETH"]')
    selected.symlink_to(first.name)
    # The old helper's default physical-path result remains unchanged.
    assert utils._resolve_coins_file_path('coinlist') == first.resolve()
    raw = released('trailing_martingale')
    raw['live']['approved_coins'] = ['coinlist'] if wrapped else 'coinlist'
    before = deepcopy(raw)
    exported = migrate(raw, base_config_path=str(configs/'input.json'), memory_span_minutes=123.5).config
    for side in ('long','short'):
        assert exported['bot'][side]['universe']['approved_coins'] == ([str(selected)] if wrapped else str(selected))
    selected.unlink()
    selected.symlink_to(second.name)
    effective = prepare_config(exported, base_config_path=str(tmp_path/'moved.json'), verbose=False)
    assert resolve_universe_sources(effective, 'approved_coins')[0] == {'long':['ETH'],'short':['ETH']}
    assert raw == before
