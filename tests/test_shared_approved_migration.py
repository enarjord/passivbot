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
