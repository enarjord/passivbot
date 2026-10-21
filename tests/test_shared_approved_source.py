"""Shared approval binds by source, not by equal materialized membership."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import pytest

from config import prepare_config
from config.schema import get_template_config
from config.universe import (
    ResolvedCoinList, get_side_universe_sources, resolve_universe_sources,
    restore_universe_sources, universe_source_view,
)
from tools.clean_config import cleanup_config
from config.authoring import compact_config, resolved_config


def source(shared=None, long=None, short=None):
    c = {"config_version": "v9.0.0", "bot": {"global": {"universe": {}},
        "long": {"universe": {"approved_coins": long, "ignored_coins": []}},
        "short": {"universe": {"approved_coins": short, "ignored_coins": []}}}, "live": {}}
    if shared is not None:
        c["bot"]["global"]["universe"]["approved_coins"] = shared
    return c


@pytest.mark.parametrize("long,expected", [(None,["BTC","ETH"]), ([],[]), (["ETH"],["ETH"])])
def test_side_replacement_and_inheritance(long, expected):
    c = prepare_config(source(["BTC","ETH"], long), verbose=False)
    assert c["bot"]["long"]["universe"]["approved_coins"] == expected
    assert universe_source_view(c)["bot"]["long"]["universe"]["approved_coins"] == long
    c["bot"]["global"]["universe"]["approved_coins"] = ["SOL"]
    values, _ = resolve_universe_sources(c, "approved_coins")
    assert values["long"] == (["SOL"] if long is None else long)


@pytest.mark.parametrize("sections", [{}, {"live": {}}, {"bot": {}}, {"bot": {"long": {}}}])
def test_sparse_and_starter_are_distinct(sections):
    raw = {"config_version": "v9.0.0", **deepcopy(sections)}
    sparse = prepare_config(raw, verbose=False)
    assert sparse["bot"]["global"]["universe"]["approved_coins"] == []
    assert sparse["bot"]["long"]["universe"]["approved_coins"] == []
    from config.load import load_input_config
    for export in (compact_config(raw), resolved_config(raw), cleanup_config(raw, _allow_sparse=True)):
        effective = prepare_config(export, verbose=False)
        assert get_side_universe_sources(effective, 'approved_coins') == {'long': [], 'short': []}
    assert load_input_config(None, log_info=False)[0]['bot']['global']['universe']['approved_coins']
    starter = prepare_config(get_template_config(), verbose=False)
    assert starter["bot"]["global"]["universe"]["approved_coins"]
    assert universe_source_view(starter)["bot"]["long"]["universe"]["approved_coins"] is None


def test_input_metadata_cannot_grant_approval():
    c = source([])
    c["_coins_sources"] = {"approved_coins":{"long":["BTC"],"short":["BTC"]}}
    assert prepare_config(c, verbose=False)["bot"]["long"]["universe"]["approved_coins"] == []


@pytest.mark.parametrize("value", [None, {"long":["BTC"],"short":[]}])
def test_shared_source_rejects_null_or_directional_shape(value):
    c = source([])
    c["bot"]["global"]["universe"]["approved_coins"] = value
    with pytest.raises((TypeError, ValueError)):
        prepare_config(c, verbose=False)


def test_materialization_export_and_direct_edit_preserve_authority():
    c = prepare_config(source(["BTC"], ["BTC"]), verbose=False)
    for mode in ("full","compact","resolved"):
        export = cleanup_config(c, mode=mode)
        assert export["bot"]["long"]["universe"]["approved_coins"] == ["BTC"]
        assert export["bot"]["short"]["universe"]["approved_coins"] is None
        export["bot"]["global"]["universe"]["approved_coins"] = ["ETH"]
        values, _ = resolve_universe_sources(prepare_config(export, verbose=False), "approved_coins")
        assert values == {"long":["BTC"],"short":["ETH"]}
    c["bot"]["short"]["universe"]["approved_coins"] = []
    assert universe_source_view(c)["bot"]["short"]["universe"]["approved_coins"] == []


@pytest.mark.parametrize("owner", ["scenario","fixed"])
def test_post_materialization_patch_shared_and_side(owner):
    c = prepare_config(source(["BTC"]), verbose=False)
    if owner == "scenario":
        from suite_runner import apply_scenario_overrides as apply
    else:
        from optimization.warmup import _apply_config_overrides as apply
    apply(c, {"bot.global.universe.approved_coins":["ETH"], "bot.short.universe.approved_coins":[]})
    values, _ = resolve_universe_sources(c,"approved_coins")
    assert values == {"long":["ETH"],"short":[]}
    apply(c, {"bot.global.universe.approved_coins":["SOL"]})
    assert resolve_universe_sources(c,"approved_coins")[0] == {"long":["SOL"],"short":[]}


def test_shared_file_refresh_failure_recovery_and_independent_pin(tmp_path):
    config_path = tmp_path / "input.json"
    coins = tmp_path / "coins.json"
    coins.write_text('["BTC"]')
    c = prepare_config(source("coins.json", short=["ETH"]), base_config_path=str(config_path), live_only=True, verbose=False)
    assert resolve_universe_sources(c,"approved_coins")[0] == {"long":["BTC"],"short":["ETH"]}
    coins.write_text('["SOL"]')
    assert resolve_universe_sources(c,"approved_coins")[0]["long"] == ["SOL"]
    coins.unlink()
    values, errors = resolve_universe_sources(c,"approved_coins",allow_partial=True)
    assert values == {"short":["ETH"]} and isinstance(errors["long"],FileNotFoundError)
    coins.write_text('["XRP"]')
    assert resolve_universe_sources(c,"approved_coins")[0]["long"] == ["XRP"]
    coins.unlink()
    c["bot"]["long"]["universe"]["approved_coins"] = ["BTC"]
    assert resolve_universe_sources(c,"approved_coins")[0] == {"long":["BTC"],"short":["ETH"]}


def test_shared_file_read_once_per_context(tmp_path, monkeypatch):
    p=tmp_path/'coins.json';p.write_text('["BTC"]')
    c=prepare_config(source(str(p)),live_only=True,verbose=False)
    original=Path.open; calls=[]
    def opened(self,*a,**k):
        if self == p: calls.append(self)
        return original(self,*a,**k)
    monkeypatch.setattr(Path,'open',opened)
    assert resolve_universe_sources(c,'approved_coins')[0] == {'long':['BTC'],'short':['BTC']}
    assert calls == [p]


def test_relative_files_are_bound_to_selected_input(tmp_path):
    for name,coin in [('a','BTC'),('b','ETH')]:
        d=tmp_path/name;d.mkdir();(d/'coins.json').write_text(json.dumps([coin]))
        c=prepare_config(source('coins.json'),base_config_path=str(d/'config.json'),live_only=True,verbose=False)
        assert resolve_universe_sources(c,'approved_coins')[0]['long'] == [coin]


def test_shared_file_rejects_directional_object(tmp_path):
    p=tmp_path/'coins.json';p.write_text('{"long":["BTC"],"short":["ETH"]}')
    with pytest.raises(ValueError,match='one coin list'):
        prepare_config(source(str(p)),verbose=False)
    c=prepare_config(source([], str(p), str(p)),verbose=False)
    assert get_side_universe_sources(c,'approved_coins') == {'long':['BTC'],'short':['ETH']}


def test_resolved_files_preserve_binding_across_scenario_and_fixed(tmp_path):
    p=tmp_path/'coins.json';p.write_text('["BTC"]')
    c=source(str(p));c['optimize']={'fixed_runtime_overrides':{'bot.global.universe.approved_coins':['ETH']}}
    c['backtest']={'scenarios':[{'label':'replace','overrides':{'bot.global.universe.approved_coins':['SOL']}}]}
    exported=resolved_config(c)
    p.unlink()
    assert exported['bot']['global']['universe']['approved_coins']==['BTC']
    assert exported['bot']['long']['universe']['approved_coins'] is None
    from optimization.warmup import _apply_config_overrides
    from suite_runner import apply_scenario_overrides
    reloaded=prepare_config(exported,verbose=False)
    _apply_config_overrides(reloaded,reloaded['optimize']['fixed_runtime_overrides'])
    assert resolve_universe_sources(reloaded,'approved_coins')[0]['long']==['ETH']
    apply_scenario_overrides(reloaded,reloaded['backtest']['scenarios'][0]['overrides'])
    assert resolve_universe_sources(reloaded,'approved_coins')[0]['long']==['SOL']


@pytest.mark.parametrize('sort_first',[False,True])
def test_prepared_dump_reload_still_inherits_shared_scenario(tmp_path,sort_first):
    from config_utils import dump_config
    from pure_funcs import sort_dict_keys
    c=prepare_config(source(['BTC'],['BTC']),verbose=False)
    if sort_first: c=sort_dict_keys(c)
    path=tmp_path/'saved.json';dump_config(c,str(path),clean=True)
    raw=json.loads(path.read_text())
    assert raw['bot']['short']['universe']['approved_coins'] is None
    assert raw['bot']['long']['universe']['approved_coins']==['BTC']
    from suite_runner import apply_scenario_overrides
    c=prepare_config(raw,verbose=False)
    apply_scenario_overrides(c,{'bot.global.universe.approved_coins':['ETH']})
    assert resolve_universe_sources(c,'approved_coins')[0]=={'long':['BTC'],'short':['ETH']}


@pytest.mark.parametrize('command',['live','backtest','optimize'])
def test_cli_both_side_replacement_shared_and_directional_precedence(command):
    from config_utils import add_config_arguments,project_template_config_for_cli,update_config_with_args
    p=argparse.ArgumentParser()
    keys=add_config_arguments(p,project_template_config_for_cli(get_template_config(),command),command=command)
    c=prepare_config(source(['BTC'],['ETH'],['SOL']),live_only=True,verbose=False)
    args=p.parse_args(['--bot.short.universe.approved_coins','XRP','-s','BNB'])
    update_config_with_args(c,args,allowed_keys=keys)
    assert resolve_universe_sources(c,'approved_coins')[0]=={'long':['BNB'],'short':['XRP']}
    args=p.parse_args(['--bot.global.universe.approved_coins','DOGE'])
    update_config_with_args(c,args,allowed_keys=keys)
    assert resolve_universe_sources(c,'approved_coins')[0]=={'long':['BNB'],'short':['XRP']}
    args=p.parse_args(['--bot.long.universe.approved_coins','null'])
    update_config_with_args(c,args,allowed_keys=keys)
    assert resolve_universe_sources(c,'approved_coins')[0]=={'long':['DOGE'],'short':['XRP']}


def test_native_worker_projection_is_an_intentional_fixed_membership_view():
    from optimization.native_datasets import _worker_config
    c=prepare_config(source(['BTC'],['ETH']),verbose=False)
    snapshot=json.loads(json.dumps(_worker_config(c)))
    assert snapshot['bot']['short']['universe']['approved_coins'] == ['BTC']
    assert snapshot['bot']['long']['universe']['approved_coins'] == ['ETH']
    assert universe_source_view(c)['bot']['short']['universe']['approved_coins'] is None


@pytest.mark.parametrize('owner', ['root', 'scenario', 'fixed'])
@pytest.mark.parametrize('selector', ['bot.global.universe.approved_coins', 'bot.long.universe.approved_coins',
                                      'bot.global.universe', 'bot.long.universe'])
def test_compact_rebases_universe_files_at_each_authority_scope(tmp_path, owner, selector):
    original = tmp_path / 'original'
    original.mkdir()
    coins = original / 'coins.json'
    coins.write_text('["BTC"]')
    destination = tmp_path / 'moved' / 'output.json'
    destination.parent.mkdir()
    (destination.parent / 'coins.json').write_text('["DECOY"]')
    config = source([], short=[])
    replacement = 'coins.json' if selector.endswith('.approved_coins') else {'approved_coins': 'coins.json'}
    if owner == 'root':
        node = config
        parts = selector.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = replacement
    elif owner == 'scenario':
        config['backtest'] = {'scenarios': [{'label': 'file', 'overrides': {selector: replacement}}]}
    else:
        config['optimize'] = {'fixed_runtime_overrides': {selector: replacement}}
    original_path = original / 'input.json'
    original_path.write_text(json.dumps(config))
    before = original_path.read_bytes(), coins.read_bytes()
    if owner == 'fixed' and not selector.endswith('.approved_coins'):
        with pytest.raises(TypeError, match='target leaf settings'):
            compact_config(config, base_config_path=str(original_path), destination_path=str(destination))
        assert before == (original_path.read_bytes(), coins.read_bytes())
        return
    exported = compact_config(config, base_config_path=str(original_path), destination_path=str(destination))
    reloaded = prepare_config(exported, base_config_path=str(destination), live_only=True, verbose=False)
    if owner == 'scenario':
        from suite_runner import apply_scenario_overrides
        apply_scenario_overrides(reloaded, reloaded['backtest']['scenarios'][0]['overrides'])
    elif owner == 'fixed':
        from optimization.warmup import _apply_config_overrides
        _apply_config_overrides(reloaded, reloaded['optimize']['fixed_runtime_overrides'])
    assert resolve_universe_sources(reloaded, 'approved_coins')[0] == {'long': ['BTC'], 'short': []}
    assert before == (original_path.read_bytes(), coins.read_bytes())


@pytest.mark.parametrize('mode', ['full', 'live', 'backtest', 'optimize', 'compact'])
@pytest.mark.parametrize('suffix', ['.json', ''])
@pytest.mark.parametrize('wrapped', [False, True])
def test_semantic_cli_export_rebases_shared_and_side_sources_but_format_only_preserves_them(tmp_path, mode, suffix, wrapped):
    from tools.clean_config import render_source
    original = tmp_path / 'original'
    original.mkdir()
    shared_name, side_name = 'shared' + suffix, 'side' + suffix
    (original / shared_name).write_text('["BTC"]')
    (original / side_name).write_text('["ETH"]')
    shared_ref = './' + shared_name if wrapped and not suffix else shared_name
    side_ref = './' + side_name if wrapped and not suffix else side_name
    config = source([shared_ref] if wrapped else shared_ref, short=[side_ref] if wrapped else side_ref)
    text = json.dumps(config)
    selected = original / 'input.json'
    selected.write_text(text)
    destination = tmp_path / 'moved' / 'output.json'
    destination.parent.mkdir()
    for name in (shared_name, side_name):
        (destination.parent / name).write_text('["DECOY"]')
    exported = json.loads(render_source(text, source=selected, mode=mode, destination=destination))
    reloaded = prepare_config(exported, base_config_path=str(destination), live_only=True, verbose=False)
    assert resolve_universe_sources(reloaded, 'approved_coins')[0] == {'long': ['BTC'], 'short': ['ETH']}
    formatted = json.loads(render_source(text, source=selected, mode='format', destination=destination))
    assert formatted == config  # Formatting does not rebase, hydrate, or change source authority.
    assert selected.read_text() == text


@pytest.mark.parametrize('pinned', [False, True])
def test_explain_distinguishes_inherited_file_source_from_equal_explicit_pin(tmp_path, pinned):
    from config.authoring import explain_config
    coins = tmp_path / 'coins.json'
    coins.write_text('["BTC"]')
    config = source('coins.json', long=['BTC'] if pinned else None)
    report = explain_config(config, 'bot.long.universe.approved_coins', base_config_path=str(tmp_path / 'input.json'))
    assert report['value'] == ['BTC']
    inherited = [event for event in report['sources'] if event['kind'] == 'inherited']
    assert bool(inherited) is not pinned
    files = [event for event in report['sources'] if event['kind'] == 'coin_list_file']
    assert bool(files) is not pinned
    if files:
        assert files[0]['path'] == 'bot.global.universe.approved_coins'
        assert Path(files[0]['file']) == coins


@pytest.mark.parametrize('scope', ['global', 'long', 'ignored'])
@pytest.mark.parametrize('explicit', [False, True])
def test_mixed_inline_sources_distinguish_literal_tokens_and_explicit_files(tmp_path, scope, explicit):
    (tmp_path / 'coinlist').write_text('["ETH"]')
    items = ['BTC', './coinlist' if explicit else 'coinlist']
    config = source(items)
    if scope == 'long':
        config = source([], long=items)
    elif scope == 'ignored':
        config = source(['BTC'])
        config['bot']['long']['universe']['ignored_coins'] = items
    if explicit:
        with pytest.raises(ValueError, match='multi-item source'):
            prepare_config(config, base_config_path=str(tmp_path/'input.json'), verbose=False)
    else:
        prepared = prepare_config(config, base_config_path=str(tmp_path/'input.json'), verbose=False)
        field = 'ignored_coins' if scope == 'ignored' else 'approved_coins'
        side = 'global' if scope == 'global' else 'long'
        assert universe_source_view(prepared)['bot'][side]['universe'][field] == items


@pytest.mark.parametrize('owner', ['scenario', 'fixed'])
def test_unrelated_universe_filter_patch_does_not_discard_prepared_source_membership(owner):
    from config.universe import materialize_side_universe_sources
    config = prepare_config(source(['BTC']), verbose=False)
    materialize_side_universe_sources(config, 'approved_coins', {'long':['BTC'], 'short':[]})
    if owner == 'scenario':
        from suite_runner import apply_scenario_overrides as apply
    else:
        from optimization.warmup import _apply_config_overrides as apply
    apply(config, {'bot.global.universe.minimum_coin_age_days': 2.5})
    assert isinstance(config['bot']['long']['universe']['approved_coins'], ResolvedCoinList)
    assert get_side_universe_sources(config, 'approved_coins') == {'long':['BTC'], 'short':[]}
    assert universe_source_view(config)['bot']['long']['universe']['approved_coins'] is None


@pytest.mark.parametrize('wrapped', [False, True])
def test_all_sentinel_wins_over_same_named_file(tmp_path, wrapped):
    (tmp_path / 'all').write_text('["BTC"]')
    raw = ['all'] if wrapped else 'all'
    config = prepare_config(source(raw), base_config_path=str(tmp_path/'input.json'), live_only=True, verbose=False)
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['all'], 'short':['all']}
    config['bot']['global']['universe']['approved_coins'] = './all'
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['BTC'], 'short':['BTC']}


@pytest.mark.parametrize('backend', ['pymoo', 'legacy'])
@pytest.mark.parametrize('filename', ['coins.json', 'coinlist'])
def test_actual_optimizer_builder_saved_policy_reloads_outside_input_directory(tmp_path, backend, filename):
    import optimize
    from optimization.callback import build_pymoo_record_entry
    from optimization.evaluation_contract import build_evaluation_contract, CONTRACT_KEY
    from config_utils import dump_config
    from optimization.prepared_dataset_identity import build_prepared_dataset_identity, PREPARED_DATASET_KEY
    from shared_arrays import SharedArrayManager
    import numpy as np

    def prepared_identity(value):
        manager = SharedArrayManager()
        try:
            specs = [manager.create_from(array)[0] for array in (np.ones((8, 2, 4)), np.ones(8))]
            return build_prepared_dataset_identity(config=value, hlcvs_specs={'binance':specs[0]},
                btc_usd_specs={'binance':specs[1]}, msss={'binance':{'BTC':{},'ETH':{}}},
                timestamps={'binance':np.arange(8, dtype=np.int64) * 60_000})
        finally:
            manager.cleanup()

    original = tmp_path/'input'
    original.mkdir()
    coins = original/filename
    coins.write_text('["BTC"]')
    short = original/'short.json'
    short.write_text('["ETH"]')
    config = prepare_config(source(filename, short=['short.json']), base_config_path=str(original/'config.json'), verbose=False)
    config['backtest']['coins'] = {'binance':['BTC','ETH']}
    config[PREPARED_DATASET_KEY] = prepared_identity(config)
    vector = optimize.config_to_individual(config, optimize.extract_bounds_tuple_list_from_config(config), sig_digits=6)
    before = deepcopy(config)
    # Result creation never re-reads prepared source membership, even if temporarily unavailable.
    coins.unlink()
    short.unlink()
    if backend == 'pymoo':
        entry = build_pymoo_record_entry(vector=vector, metrics={}, template=config,
            build_config_fn=optimize.individual_to_config, overrides_fn=optimize.optimizer_overrides,
            overrides_list=[])
    else:
        class Candidate(list):
            pass
        candidate = Candidate(vector)
        candidate.evaluation_metrics = {}
        records = []
        optimize._record_individual_result(candidate, config, [], SimpleNamespace(record=records.append))
        entry = records[0]
    assert entry['bot']['global']['universe']['approved_coins'] == str(coins)
    assert entry['bot']['long']['universe']['approved_coins'] is None
    assert entry['bot']['short']['universe']['approved_coins'] == [str(short)]
    assert config == before
    coins.write_text('["BTC"]')
    short.write_text('["ETH"]')
    destination = tmp_path/'output'/'record.json'
    destination.parent.mkdir()
    (destination.parent/filename).write_text('["DECOY"]')
    dump_config(entry, str(destination))
    exported = cleanup_config(json.loads(destination.read_text()))
    reloaded = prepare_config(exported, base_config_path=str(destination), verbose=False)
    assert resolve_universe_sources(reloaded, 'approved_coins')[0] == {'long':['BTC'], 'short':['ETH']}
    reloaded['backtest']['coins'] = deepcopy(config['backtest']['coins'])
    reloaded[PREPARED_DATASET_KEY] = prepared_identity(reloaded)
    disk_record = json.loads(destination.read_text())
    assert disk_record[CONTRACT_KEY] == build_evaluation_contract(reloaded)
    # Same prepared candles and coin union, but changed directional membership, cannot reuse fitness.
    coins.write_text('["ETH"]')
    short.write_text('["BTC"]')
    changed = prepare_config(exported, base_config_path=str(destination), verbose=False)
    changed['backtest']['coins'] = deepcopy(config['backtest']['coins'])
    changed[PREPARED_DATASET_KEY] = prepared_identity(changed)
    assert any('prepared_data' in item for item in optimize._resume_config_mismatches(disk_record, changed))


def test_prepared_literal_approval_identity_ignores_later_same_named_local_file(tmp_path):
    from optimization.native_planning import execution_key
    from optimization.evaluation_contract import build_evaluation_contract
    config = prepare_config(source(['BTC']), base_config_path=str(tmp_path/'input.json'), verbose=False)
    config['backtest']['coins'] = {'binance':['BTC']}
    expected = execution_key(config), build_evaluation_contract(config)
    (tmp_path/'BTC').write_text('["ETH"]')
    assert (execution_key(config), build_evaluation_contract(config)) == expected


def test_cli_rebinds_only_explicitly_assigned_source_leaves(tmp_path):
    from config_utils import update_config_with_args
    config = prepare_config(source(['BTC'], long=['BTC']), base_config_path=str(tmp_path/'input.json'), live_only=True, verbose=False)
    (tmp_path/'BTC').write_text('["ETH"]')
    update_config_with_args(config, SimpleNamespace(**{'live.account.user':'example'}), verbose=False)
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['BTC'],'short':['BTC']}
    update_config_with_args(config, SimpleNamespace(**{'bot.global.universe.approved_coins':['SOL']}), verbose=False)
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['BTC'],'short':['SOL']}
    update_config_with_args(config, SimpleNamespace(**{'bot.long.universe.approved_coins':'BTC'}), verbose=False)
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['ETH'],'short':['SOL']}


def test_explicit_side_file_keeps_existing_plain_text_format(tmp_path):
    selected = tmp_path/'coins.txt'
    selected.write_text('BTC, ETH\nSOL')
    config = prepare_config(source([], long='coins.txt'), base_config_path=str(tmp_path/'input.json'), verbose=False)
    assert get_side_universe_sources(config, 'approved_coins') == {'long':['BTC','ETH','SOL'], 'short':[]}


@pytest.mark.parametrize('owner', ['helper', 'public'])
@pytest.mark.parametrize('literal', [['BTC'], 'BTC'])
def test_bounds_anchoring_keeps_prepared_literal_membership_after_late_file(tmp_path, owner, literal):
    from tools.optimizer_bounds import _anchor_file_references
    config = prepare_config(source(literal, long=literal),
        base_config_path=str(tmp_path/'input.json'), verbose=False)
    (tmp_path/'BTC').write_text('["ETH"]')
    if owner == 'helper':
        exported = universe_source_view(config)
        _anchor_file_references(exported, base_config_path=str(tmp_path/'input.json'))
    else:
        from tools.optimizer_bounds import apply_recommended_bounds
        exported = apply_recommended_bounds(config, base_config_path=str(tmp_path/'input.json'))
    assert exported['bot']['global']['universe']['approved_coins'] == ['BTC']
    assert exported['bot']['long']['universe']['approved_coins'] == ['BTC']
    assert resolve_universe_sources(exported, 'approved_coins')[0] == {'long':['BTC'], 'short':['BTC']}


@pytest.mark.parametrize('wrapped', [False, True])
def test_shared_source_rejects_json_encoded_directional_object(wrapped):
    encoded = '{"long":["BTC"],"short":[]}'
    with pytest.raises(TypeError, match='one common coin list'):
        prepare_config(source([encoded] if wrapped else encoded), verbose=False)


@pytest.mark.parametrize('mode', ['full', 'compact'])
def test_semantic_export_preserves_reloadable_coin_file_symlink_alias(tmp_path, mode):
    from tools.clean_config import render_source
    original = tmp_path/'input'
    original.mkdir()
    first, second = original/'first.json', original/'second.json'
    first.write_text('["BTC"]')
    second.write_text('["ETH"]')
    alias = original/'coins.json'
    alias.symlink_to(first.name)
    selected = original/'config.json'
    text = json.dumps(source('coins.json'))
    selected.write_text(text)
    destination = tmp_path/'output'/'config.json'
    destination.parent.mkdir()
    exported = json.loads(render_source(text, source=selected, mode=mode, destination=destination))
    retained = exported['bot']['global']['universe']['approved_coins']
    assert Path(retained).name == alias.name
    alias.unlink()
    alias.symlink_to(second.name)
    reloaded = prepare_config(exported, base_config_path=str(destination), verbose=False)
    assert resolve_universe_sources(reloaded, 'approved_coins')[0] == {'long':['ETH'], 'short':['ETH']}
    assert selected.read_text() == text


@pytest.mark.parametrize('encoded', [
    '{"long":["BTC"],"short":[]}',
    '{"long":["BTC"],"short":["BTC"]}',
    '["BTC"]',
])
def test_shared_file_members_remain_literal_tokens_not_json_shorthand(tmp_path, encoded):
    (tmp_path/'coins.json').write_text(json.dumps([encoded]))
    config = prepare_config(source('coins.json'), base_config_path=str(tmp_path/'input.json'), verbose=False)
    values, _ = resolve_universe_sources(config, 'approved_coins')
    assert values['long'] == values['short']
    assert 'BTC' not in values['long']  # An encoded structure grants no membership to its contents.
    assert universe_source_view(config)['bot']['global']['universe']['approved_coins'] == 'coins.json'


def test_literal_source_owner_does_not_change_legacy_jsonish_helper_default():
    from utils import normalize_coins_source
    encoded = '{"long":["BTC"],"short":[]}'
    assert normalize_coins_source(encoded, resolve_files=False) == {'long':['BTC'], 'short':[]}
    literal = normalize_coins_source(encoded, resolve_files=False, parse_jsonish=False)
    assert literal['long'] == literal['short']
    assert 'BTC' not in literal['long']


@pytest.mark.parametrize('content', ['BTC, ETH', 'BTC\nETH\n', '"BTC"\n"ETH"\n'])
def test_shared_plain_text_file_uses_existing_token_format_and_reads_once(tmp_path, monkeypatch, content):
    selected = tmp_path/'coins.txt'
    selected.write_text(content)
    config = prepare_config(source(str(selected)), live_only=True, verbose=False)
    opened, original = [], Path.open
    def counted(path, *args, **kwargs):
        if path == selected:
            opened.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['BTC','ETH'],'short':['BTC','ETH']}
    assert opened == [selected]
    selected.write_text('SOL\n')
    assert resolve_universe_sources(config, 'approved_coins')[0] == {'long':['SOL'],'short':['SOL']}


@pytest.mark.parametrize('coins', [['BTC'], []])
def test_shared_file_rejects_equal_directional_object_too(tmp_path, coins):
    selected = tmp_path/'coins.txt'
    selected.write_text(json.dumps({'long':coins,'short':coins}))
    with pytest.raises(ValueError, match='one coin list'):
        prepare_config(source(str(selected)), verbose=False)


@pytest.mark.parametrize('content', ['{long BTC', '# comment\n{long: [BTC]'])
def test_shared_file_never_demotes_malformed_mapping_to_plain_coins(tmp_path, content):
    selected = tmp_path/'coins.txt'
    selected.write_text(content)
    with pytest.raises(ValueError, match='one coin list'):
        prepare_config(source(str(selected)), verbose=False)
