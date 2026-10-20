"""Authoring views preserve effective policy and keep evidence out of config."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from config.universe import require_side_universe_sources, set_side_universe_sources
from config.authoring import compact_config, resolved_config, explain_config
from config.load import prepare_config
from config.overrides import parse_overrides
from config.schema import get_template_config
from config.hsl import generated_template
from config_utils import strip_config_metadata
from passivbot_cli.main import main as cli_main
from tools.clean_config import cleanup_config, main as clean_main


def policy():
    config = get_template_config()
    set_side_universe_sources(config, 'approved_coins', {'long': ['BTC', 'ETH'], 'short': []})
    set_side_universe_sources(config, 'ignored_coins', {'long': [], 'short': []})
    return config


def effective(source, base=""):
    prepared = prepare_config(source, base_config_path=base, verbose=False)
    result = strip_config_metadata(parse_overrides(prepared, verbose=False))
    result["live"].pop("base_config_path", None)
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_compact_default_view_is_smaller_and_reloads_with_same_policy():
    source = policy()
    before = deepcopy(source)
    compact = compact_config(source)
    assert source == before
    assert len(json.dumps(compact)) < len(json.dumps(source)) * 0.7
    assert "logging" not in compact
    assert "warmup" not in compact["bot"].get("global", {})
    assert effective(compact) == effective(source)
    assert compact_config(compact) == compact


@pytest.mark.parametrize("sections", [
    {}, {"bot": {}}, {"live": {}}, {"bot": {}, "live": {}},
    {"bot": {"long": {}}}, {"bot": {"global": {"strategy_kind": "ema_anchor"}}},
])
def test_authoring_views_preserve_whole_section_omission(sections):
    source = {"config_version": "v9.0.0", **deepcopy(sections)}
    original = deepcopy(source)
    expected = effective(source)
    compact = compact_config(source)
    assert ("bot" in compact) == ("bot" in source)
    assert ("live" in compact) == ("live" in source)
    assert effective(compact) == expected
    assert compact_config(compact) == compact
    assert effective(resolved_config(source)) == expected
    report = explain_config(source, "bot.long.risk.total_wallet_exposure_limit")
    assert report["value"] == expected["bot"]["long"]["risk"]["total_wallet_exposure_limit"]
    assert (report["sources"][0].get("rule") == "omitted_side_disabled") == (
        "bot" in source and "long" not in source["bot"]
    )
    assert source == original


@pytest.mark.parametrize("export", [compact_config, resolved_config,
                                    lambda config: explain_config(config, "bot.global.hedge_mode")])
@pytest.mark.parametrize("patch", [{"unknown": {}}, {"live": {"hsl_engine": "legacy"}},
                                   {"bot": {"global": {"unknown": True}}}])
def test_sparse_authoring_does_not_bypass_unknown_or_retired_validation(export, patch):
    source = {"config_version": "v9.0.0", **patch}
    with pytest.raises((ValueError, TypeError)):
        export(source)


def test_compact_preserves_nondefault_strategy_selector():
    source = policy()
    source['bot']['global']['strategy_kind'] = "ema_anchor"
    compact = compact_config(source)
    assert compact['bot']['global']['strategy_kind'] == "ema_anchor"
    assert effective(compact) == effective(source)


@pytest.mark.parametrize("side", ["long", "short"])
def test_compact_retains_side_presence_and_explicit_disable(side):
    source = {"config_version": "v9.0.0", "bot": {side: {}}, "live": {}}
    compact = compact_config(source)
    assert side in compact["bot"]
    assert effective(compact) == effective(source)
    source["bot"][side]["risk"] = {"total_wallet_exposure_limit": 0.0}
    assert effective(compact_config(source)) == effective(source)


@pytest.mark.parametrize("weights", [
    {"volume": 2.0, "ema_readiness": 1.0, "volatility": 0.0, "unilateralness": 0.0},
    {"volume": 0.0, "ema_readiness": 0.0, "volatility": 0.0, "unilateralness": 0.0},
])
def test_compact_preserves_effective_normalized_weights(weights):
    source = policy()
    source["bot"]["long"]["forager"]["score_weights"] = weights
    assert effective(compact_config(source)) == effective(source)


def test_compact_preserves_authored_search_authority_and_sparse_pins():
    source = policy()
    source["optimize"]["bounds"] = {"long": {"unstuck": {"ema_span_0": [17.5, 17.5]}}}
    source["optimize"]["fixed_params"] = ["long.unstuck.ema_span_0"]
    source["optimize"]["fixed_runtime_overrides"] = {"bot.long.unstuck.close_pct": 0.11}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {"bot.long.unstuck.close_pct": 0.12}}]
    source["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 790.0}}}}}
    compact = compact_config(source)
    for name in ("bounds", "fixed_params", "fixed_runtime_overrides"):
        assert compact["optimize"][name] == source["optimize"][name]
    assert compact["coin_overrides"] == source["coin_overrides"]
    assert compact["backtest"]["scenarios"] == source["backtest"]["scenarios"]
    assert effective(compact) == effective(source)


def test_compact_preserves_null_inheritance_and_eligibility_sources():
    source = policy()
    source["backtest"]["filter_by_min_effective_cost"] = None
    set_side_universe_sources(source, 'approved_coins', [])
    set_side_universe_sources(source, 'ignored_coins', {'long': ['ETH'], 'short': []})
    compact = compact_config(source)
    assert compact["backtest"]["filter_by_min_effective_cost"] is None
    assert require_side_universe_sources(compact, 'approved_coins') == {'long': [], 'short': []}
    assert require_side_universe_sources(compact, 'ignored_coins') == require_side_universe_sources(source, 'ignored_coins')
    for flag in (False, True):
        left, right = deepcopy(source), deepcopy(compact)
        left['bot']['global']['universe']['filter_by_min_effective_cost'] = flag
        right['bot'].setdefault('global', {}).setdefault('universe', {})['filter_by_min_effective_cost'] = flag
        assert effective(left) == effective(right)


def test_compact_retains_optional_portfolio_authorization():
    source = generated_template(policy(), "portfolio")
    compact = compact_config(source)
    assert compact["bot"]["global"]["hsl"] == source["bot"]["global"]["hsl"]
    assert effective(compact) == effective(source)


def test_compact_rebases_relative_override_file_when_output_moves(tmp_path):
    override = write_json(tmp_path / "inputs" / "coin.json", {"bot": {"long": {"unstuck": {"ema_span_0": 17.5}}}})
    source = policy()
    source["coin_overrides"] = {"BTC": {"override_config_path": override.name}}
    input_path = write_json(override.parent / "input.json", source)
    destination = tmp_path / "outputs" / "compact.json"
    assert clean_main([str(input_path), str(destination), "--mode", "compact"]) == 0
    compact = json.loads(destination.read_text())
    assert compact["coin_overrides"]["BTC"]["override_config_path"] == "../inputs/coin.json"
    assert effective(compact, str(destination))["coin_overrides"] == effective(source, str(input_path))["coin_overrides"]


def test_resolved_export_inlines_files_and_preserves_sparse_override_precedence(tmp_path):
    coins = write_json(tmp_path / "approved.json", {"long": ["BTC", "ETH"], "short": []})
    override = write_json(tmp_path / "coin.json", {"bot": {"long": {"unstuck": {"ema_span_0": 17.5, "close_pct": 0.12}}}})
    source = policy()
    set_side_universe_sources(source, 'approved_coins', str(coins))
    set_side_universe_sources(source, 'ignored_coins', {'long': ['ETH'], 'short': []})
    source["backtest"]["filter_by_min_effective_cost"] = None
    source["coin_overrides"] = {"BTC": {"override_config_path": override.name,
                                       "bot": {"long": {"unstuck": {"close_pct": 0.13}}}}}
    input_path = write_json(tmp_path / "input.json", source)
    before = deepcopy(source)
    resolved = resolved_config(source, base_config_path=str(input_path))
    assert source == before
    assert require_side_universe_sources(resolved, 'approved_coins') == {"long": ["BTC", "ETH"], "short": []}
    assert resolved["backtest"]["filter_by_min_effective_cost"] is None
    assert resolved["coin_overrides"]["BTC"] == {"bot": {"long": {"unstuck": {"ema_span_0": 17.5, "close_pct": 0.13}}}}
    assert effective(resolved) == effective(source, str(input_path))
    coins.unlink()
    override.unlink()
    assert resolved_config(resolved) == resolved


def test_resolved_export_materializes_scenario_file_sources(tmp_path):
    coins = write_json(tmp_path / "scenario-coins.json", ["ETH"])
    override = write_json(tmp_path / "scenario-coin.json", {"bot": {"long": {"unstuck": {"ema_span_0": 19.5}}}})
    source = policy()
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {
        "bot.long.universe.approved_coins": str(coins),
        "coin_overrides": {"ETH": {"override_config_path": override.name}},
    }}]
    resolved = resolved_config(source, base_config_path=str(tmp_path / "input.json"))
    patch = resolved["backtest"]["scenarios"][0]["overrides"]
    assert patch["bot.long.universe.approved_coins"] == ["ETH"]
    assert patch["coin_overrides"] == {"ETH": {"bot": {"long": {"unstuck": {"ema_span_0": 19.5}}}}}
    override.unlink()
    coins.unlink()
    assert resolved_config(resolved) == resolved


@pytest.mark.parametrize("exporter", [compact_config, resolved_config])
def test_exports_keep_strict_authored_and_result_boundaries(exporter):
    source = policy()
    source["unknown"] = {"do_not_prune": True}
    with pytest.raises(ValueError, match="unknown configuration field"):
        exporter(source)
    source = policy()
    wrapper = {"config": source, "analysis": {"adg": 1.0}}
    assert "analysis" not in exporter(wrapper)
    assert "sources" not in exporter(wrapper)


def test_resolved_refuses_unavailable_market_universe_and_missing_coin_file(tmp_path):
    source = policy()
    set_side_universe_sources(source, 'approved_coins', 'all')
    with pytest.raises(ValueError, match="cannot expand"):
        resolved_config(source)
    set_side_universe_sources(source, 'approved_coins', str(tmp_path / 'missing.json'))
    with pytest.raises(FileNotFoundError, match="coin-list file"):
        resolved_config(source)


def test_explain_defaults_declared_metadata_and_search_source():
    source = {"config_version": "v9.0.0", "bot": {"long": {}}, "live": {}}
    report = explain_config(source, "bot.long.strategy.trailing_martingale.entry.ema_span_0")
    assert report["value"] == 790.0
    assert report["types"] == ["float"]
    assert report["scope"] == "long"
    assert report["unit"] is None
    assert report["default"] == {"declared": True, "value": 790.0}
    assert report["sources"][0]["kind"] == "default_or_derived"
    assert report["search"]["status"] == "omitted_fixed"
    assert report["search"]["bounds_source"] == "configured_value"


def test_explain_applies_fixed_scenario_and_coin_precedence(tmp_path):
    override = write_json(tmp_path / "coin.json", {"bot": {"long": {"unstuck": {"ema_span_0": 17.5}}}})
    source = policy()
    source["optimize"]["fixed_runtime_overrides"] = {"bot.long.unstuck.close_pct": 0.11}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {"bot.long.unstuck.close_pct": 0.12}}]
    source["coin_overrides"] = {"BTC": {"override_config_path": override.name,
                                       "bot": {"long": {"unstuck": {"close_pct": 0.13}}}}}
    before = deepcopy(source)
    report = explain_config(source, "bot.long.unstuck.close_pct", coin="BTC", scenario="one", optimize=True,
                            base_config_path=str(tmp_path / "input.json"))
    assert source == before
    assert report["value"] == 0.13
    assert [item["kind"] for item in report["sources"]] == [
        "authored", "fixed_runtime_override", "scenario_override", "coin_override",
    ]
    assert report["search"]["status"] == "coin_override"
    report = explain_config(source, "bot.long.unstuck.ema_span_0", coin="BTC",
                            base_config_path=str(tmp_path / "input.json"))
    assert report["value"] == 17.5
    assert report["sources"][-1]["kind"] == "override_file"
    assert Path(report["sources"][-1]["file"]) == override


def test_search_treatment_reports_runtime_and_scenario_shadowing():
    source = policy()
    path = "bot.long.risk.total_exposure_enforcer.threshold"
    source["optimize"]["bounds"]["long"]["risk"]["total_exposure_enforcer"]["threshold"] = [0.5, 1.0]
    source["optimize"]["fixed_runtime_overrides"] = {path: 0.8}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {path: 0.9}}]
    ordinary = explain_config(source, path)
    assert ordinary["search"]["status"] == "searched"
    fixed = explain_config(source, path, optimize=True)
    assert fixed["value"] == 0.8
    assert fixed["search"]["status"] == "fixed_runtime_override"
    assert fixed["search"]["gene_status"] == "searched"
    assert fixed["search"]["bounds"] == [0.5, 1.0]
    scenario = explain_config(source, path, optimize=True, scenario="one")
    assert scenario["value"] == 0.9
    assert scenario["search"]["status"] == "scenario_override"
    assert scenario["search"]["shadowed_by"] == [
        {"kind": "fixed_runtime_override", "path": path},
        {"kind": "scenario_override", "path": path},
    ]
    bounds = explain_config(source, "optimize.bounds.long.risk.total_exposure_enforcer.threshold", optimize=True)
    assert bounds["search"]["status"] == "fixed_runtime_override"


@pytest.mark.parametrize("selected,expected", [
    ("BTC", 0.1), ("BTC/USDT:USDT", 0.125),
    ("bybit::BTC/USDT:USDT", 0.15), ("ETH", None),
])
def test_explain_coin_key_selection_preserves_exact_market_identity(selected, expected):
    source = policy()
    path = "bot.long.unstuck.close_pct"
    source["coin_overrides"] = {
        key: {"bot": {"long": {"unstuck": {"close_pct": value}}}}
        for key, value in (("BTC", 0.1), ("BTC/USDT:USDT", 0.125), ("bybit::BTC/USDT:USDT", 0.15))
    }
    report = explain_config(source, path, coin=selected)
    assert report["value"] == (source["bot"]["long"]["unstuck"]["close_pct"] if expected is None else expected)
    if expected is not None:
        assert report["sources"][-1]["kind"] == "coin_override"
        assert report["search"]["status"] == "coin_override"


def test_explain_reports_null_inheritance_fixed_selectors_and_fixed_bounds():
    source = policy()
    source["backtest"]["filter_by_min_effective_cost"] = None
    report = explain_config(source, "backtest.filter_by_min_effective_cost")
    assert report["value"] is True
    assert report["sources"][-1] == {"kind": "inherited", "path": "bot.global.universe.filter_by_min_effective_cost", "value": True}
    source["optimize"]["fixed_params"] = ["long.unstuck.ema_span_0", "short.unstuck.close_pct"]
    search = explain_config(source, "bot.long.unstuck.ema_span_0")["search"]
    assert search["status"] == "fixed_selector"
    assert search["fixed_selectors"] == ["long.unstuck.ema_span_0"]
    source["optimize"]["fixed_params"] = []
    source["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] = [17.5, 17.5]
    assert explain_config(source, "bot.long.unstuck.ema_span_0")["search"]["status"] == "fixed_bound"


def test_explain_scenario_dates_and_dataset_limit_are_explicit():
    source = policy()
    source["backtest"]["scenarios"] = [{"label": "one", "start_date": "2024-01-01", "coins": ["BTC"]}]
    report = explain_config(source, "backtest.start_date", scenario="one")
    assert report["value"] == "2024-01-01"
    assert report["sources"][-1]["kind"] == "scenario_selection"
    assert report["dataset_selection"] == {"coins": ["BTC"]}
    with pytest.raises(ValueError, match="prepared dataset"):
        explain_config(source, "bot.long.universe.approved_coins", scenario="one")


def test_explain_keeps_unexpanded_all_visible_and_does_not_invent_units():
    source = policy()
    set_side_universe_sources(source, 'approved_coins', 'all')
    report = explain_config(source, "bot.long.universe.approved_coins")
    assert report["unresolved"] == ["bot.long.universe.approved_coins", "bot.short.universe.approved_coins"]
    assert report["unit"] is None


@pytest.mark.parametrize("path,target,element_type", [
    ("optimize.bounds.long.risk.max_positions", "bot.long.risk.max_positions", "integer"),
    ("optimize.bounds.long.forager.score_weights.volume", "bot.long.forager.score_weights.volume", "float"),
])
def test_explain_bounds_reuses_existing_parameter_path_adapter(path, target, element_type):
    report = explain_config(policy(), path)
    assert report["types"] == ["array"]
    assert report["bound_target"] == target
    assert report["bound_element_types"] == [element_type]
    assert report["search"]["status"] in {"searched", "fixed_bound"}


def test_explain_bounds_container_does_not_claim_one_parameter_type():
    report = explain_config(policy(), "optimize.bounds.long")
    assert report["bound_target"] is None
    assert report["types"] is None


def test_explain_dynamic_coin_patch_reuses_root_type_without_inventing_a_pin_default():
    source = policy()
    source["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {"ema_span_0": 17.5}}}}}
    report = explain_config(source, "coin_overrides.BTC.bot.long.unstuck.ema_span_0")
    assert report["types"] == ["float"]
    assert report["default"] == {"declared": False}
    assert report["search"]["status"] == "coin_override"


def test_explain_omitted_side_disable_and_normalized_weights_are_visible():
    source = {"config_version": "v9.0.0", "bot": {}, "live": {}}
    report = explain_config(source, "bot.long.risk.total_wallet_exposure_limit")
    assert report["value"] == 0.0
    assert report["sources"][0]["rule"] == "omitted_side_disabled"
    source = policy()
    source["bot"]["long"]["forager"]["score_weights"] = {
        "ema_readiness": 1.0, "volume": 2.0, "volatility": 0.0, "unilateralness": 0.0,
    }
    report = explain_config(source, "bot.long.forager.score_weights.volume")
    assert report["value"] == pytest.approx(2 / 3)
    assert report["sources"][-1]["kind"] == "normalization"


@pytest.mark.parametrize("path", ["live.unknown", "bot.long.strategy.ema_anchor.offset"])
def test_explain_rejects_unknown_or_inactive_fields(path):
    with pytest.raises(KeyError, match="unknown or inactive"):
        explain_config(policy(), path)


def test_explain_local_coin_list_source_is_report_evidence_only(tmp_path):
    coins = write_json(tmp_path / "coins.json", ["BTC", "ETH"])
    source = policy()
    set_side_universe_sources(source, 'approved_coins', str(coins))
    report = explain_config(source, "bot.long.universe.approved_coins")
    assert report["value"] == ["BTC", "ETH"]
    assert report["sources"][-1] == {"kind": "coin_list_file", "path": "bot.long.universe.approved_coins", "file": str(coins)}
    assert "sources" not in source


def test_authoring_tools_never_connect_to_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("offline authoring attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    compact_config(policy())
    resolved_config(policy())
    explain_config(policy(), "bot.long.unstuck.close_pct", optimize=True)


def test_explain_cli_prints_report_and_never_rewrites_input(tmp_path, capsys):
    path = write_json(tmp_path / "input.json", policy())
    before = path.read_bytes()
    assert cli_main(["tool", "config-explain", str(path), "bot.global.warmup.ratio", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["value"] == 0.3
    assert report["context"]["scope"] == "offline_policy"
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["compact", "resolved"])
def test_new_clean_modes_reload_and_are_idempotent(tmp_path, mode):
    source = write_json(tmp_path / "input.json", policy())
    destination = tmp_path / f"{mode}.json"
    assert clean_main([str(source), str(destination), "--mode", mode]) == 0
    exported = json.loads(destination.read_text())
    assert effective(exported) == effective(policy())
    assert clean_main([str(destination), "--mode", mode, "--check"]) == 0


@pytest.mark.parametrize("path", sorted((Path(__file__).resolve().parents[1] / "configs" / "examples").glob("*.json")), ids=lambda path: path.name)
@pytest.mark.parametrize("exporter", [compact_config, resolved_config])
def test_public_examples_keep_effective_policy_in_new_views(path, exporter):
    source = json.loads(path.read_text())
    result = exporter(source, base_config_path=str(path))
    assert effective(result, str(path)) == effective(source, str(path))


@pytest.mark.parametrize('transform,target,dependency,value,independent', [
    ('couple_unstuck_ema_spans', 'bot.long.unstuck.ema_span_0',
     'bot.long.strategy.trailing_martingale.entry.ema_span_0', 123.5, False),
    ('mirror_short_from_long', 'bot.short.risk.total_wallet_exposure_limit',
     'bot.long.risk.total_wallet_exposure_limit', 1.25, True),
    ('lossless_close_trailing', 'bot.long.strategy.trailing_martingale.close.threshold_base_pct',
     'bot.long.strategy.trailing_martingale.close.retracement_base_pct', 0.02, True),
])
def test_explain_distinguishes_explicit_relationship_from_target_gene(transform, target, dependency, value, independent):
    from config.param_paths import resolve_dotted_config_path
    source = policy()
    source['optimize']['bounds'] = {}
    source['optimize']['enable_overrides'] = [transform]
    def assign(path, value):
        parts = resolve_dotted_config_path(source, path)
        owner = source
        for part in parts[:-1]:
            owner = owner[part]
        owner[parts[-1]] = value
    assign(dependency, value)
    assign(target, 0.0 if transform != 'couple_unstuck_ema_spans' else 42.5)
    original = deepcopy(source)
    report = explain_config(source, target, optimize=True)
    assert report['value'] == value
    assert report['search']['status'] == 'optimizer_transform'
    assert report['search']['independent_gene'] is independent
    assert report['search']['gene_status'] == ('omitted_fixed' if independent else 'not_in_genome')
    event = report['sources'][-1]
    assert event['kind'] == 'optimizer_transform'
    assert event['name'] == transform
    assert dependency in [item['path'] for item in event['dependencies']]
    assert source == original
    # A relationship remains a derivation even when it writes the same number.
    assign(target, value)
    assert explain_config(source, target, optimize=True)['search']['status'] == 'optimizer_transform'


@pytest.mark.parametrize('transform,target,dependency,late_coupling', [
    ('couple_unstuck_ema_spans', 'bot.long.unstuck.ema_span_0',
     'bot.long.strategy.trailing_martingale.entry.ema_span_0', True),
    ('mirror_short_from_long', 'bot.short.risk.total_wallet_exposure_limit',
     'bot.long.risk.total_wallet_exposure_limit', False),
    ('lossless_close_trailing', 'bot.long.strategy.trailing_martingale.close.threshold_base_pct',
     'bot.long.strategy.trailing_martingale.close.retracement_base_pct', False),
])
def test_explain_reports_scenario_order_relative_to_optimizer_relationships(transform, target, dependency, late_coupling):
    source = policy()
    source['optimize']['bounds'] = {}
    source['optimize']['enable_overrides'] = [transform]
    source['backtest']['scenarios'] = [{'label': 'one', 'overrides': {target: 0.25}}]
    report = explain_config(source, target, optimize=True, scenario='one')
    assert report['search']['status'] == ('optimizer_transform' if late_coupling else 'scenario_override')
    if late_coupling:
        owner = source
        for part in dependency.split('.'):
            owner = owner[part]
        assert report['value'] == owner
    else:
        assert report['value'] == 0.25


def test_explain_coupled_coin_span_uses_exact_coin_dependency_and_is_not_authored_pin():
    source = policy()
    source['optimize']['bounds'] = {}
    source['optimize']['enable_overrides'] = ['couple_unstuck_ema_spans']
    coin = 'bybit::BTC/USDT:USDT'
    source['coin_overrides'] = {coin: {'bot': {'long': {'strategy': {
        'trailing_martingale': {'entry': {'ema_span_0': 321.5}},
    }, 'unstuck': {'ema_span_0': 42.5}}}}}
    report = explain_config(source, 'bot.long.unstuck.ema_span_0', coin=coin, optimize=True)
    assert report['value'] == 321.5
    assert report['search']['status'] == 'optimizer_transform'
    assert report['sources'][-1]['name'] == 'couple_unstuck_ema_spans'
    dependency = report['sources'][-1]['dependencies'][0]
    assert dependency['path_parts'] == ['coin_overrides', coin, 'bot', 'long', 'strategy',
                                       'trailing_martingale', 'entry', 'ema_span_0']
    assert source['coin_overrides'][coin]['bot']['long']['unstuck']['ema_span_0'] == 42.5


def test_explain_mirroring_does_not_derive_directional_selection():
    source = policy()
    source['optimize']['enable_overrides'] = ['mirror_short_from_long']
    source['bot']['long']['forager']['always_include'] = ['BTC']
    source['bot']['short']['forager']['always_include'] = ['ETH']
    report = explain_config(source, 'bot.short.forager.always_include', optimize=True)
    assert report['value'] == ['ETH']
    assert all(event['kind'] != 'optimizer_transform' for event in report['sources'])


@pytest.mark.parametrize('metadata,value', [
    ('fixed_params', ['long.unstuck.ema_span_0']),
    ('bounds.long.unstuck.ema_span_0', [300.0, 400.0]),
    ('enable_overrides', ['couple_unstuck_ema_spans']),
])
def test_explain_scenario_metadata_does_not_redefine_outer_genome(metadata, value):
    source = policy()
    source['optimize']['bounds'] = {'long': {'unstuck': {'ema_span_0': [100.0, 200.0]}}}
    source['backtest']['scenarios'] = [{'label': 'one', 'overrides': {'optimize.' + metadata: value}}]
    report = explain_config(source, 'bot.long.unstuck.ema_span_0', optimize=True, scenario='one')
    assert report['search']['gene_status'] == 'searched'
    assert report['search']['bounds'] == [100.0, 200.0]
    assert report['search']['fixed_selectors'] == []
    assert report['search']['independent_gene'] is True


def test_explain_bound_declaration_is_not_normalized_by_a_target_transform():
    source = policy()
    source['optimize']['bounds'] = {}
    source['optimize']['enable_overrides'] = ['mirror_short_from_long']
    source['bot']['long']['risk']['total_wallet_exposure_limit'] = 1.25
    source['bot']['short']['risk']['total_wallet_exposure_limit'] = 0.0
    report = explain_config(source, 'optimize.bounds.short.risk.total_wallet_exposure_limit', optimize=True)
    assert report['value'] == [0.0, 0.0]
    assert report['search']['status'] == 'optimizer_transform'
    assert any(event['kind'] == 'optimizer_transform' and event['value'] == 1.25 for event in report['sources'])
    assert all(event['kind'] != 'normalization' for event in report['sources'])


def test_explain_complete_object_override_projects_value_before_reporting_normalization():
    source = policy()
    weights = {'volume': 0.5, 'volatility': 0.25, 'ema_readiness': 0.25, 'unilateralness': 0.0}
    source['backtest']['scenarios'] = [{'label': 'one', 'overrides': {'bot.long.forager.score_weights': weights}}]
    report = explain_config(source, 'bot.long.forager.score_weights.volume', scenario='one')
    assert report['value'] == 0.5
    assert report['sources'][-1]['kind'] == 'scenario_override'
