"""Config migration is explicit, offline and preserves source files and policy choices."""
from copy import deepcopy
import json

import pytest

from config.schema import get_template_config
from config.hsl_revised import generated_template
from tools.migrate_hsl_config import main, migrate


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    import socket
    def denied(*args, **kwargs):
        pytest.fail("offline config migration attempted network access")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def legacy(mode="coin"):
    cfg = get_template_config()
    cfg["live"]["hsl_signal_mode"] = mode
    cfg["bot"]["long"]["hsl"]["enabled"] = True
    # An authored retired optimizer override must be edited, never silently dropped.
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    if mode == "unified":
        for side in ("long", "short"):
            cfg["optimize"]["bounds"][side].pop("hsl", None)
    return cfg


@pytest.mark.parametrize("mode", ["coin", "pside"])
@pytest.mark.parametrize("policy", ["always", "never"])
def test_explicit_conversion_is_idempotent_and_preserves_other_settings(mode, policy):
    source = legacy(mode)
    original = deepcopy(source)
    result = migrate(source, restart_policies={"long": policy})
    assert source == original
    assert result["live"]["hsl_engine"] == "revised"
    assert result["bot"]["long"]["hsl"]["restart_after_red_policy"] == policy
    assert result["bot"]["long"]["hsl"]["red_threshold"] == source["bot"]["long"]["hsl"]["red_threshold"]
    assert result["bot"]["long"]["risk"] == source["bot"]["long"]["risk"]
    assert "tier_ratios" not in result["bot"]["long"]["hsl"]
    assert "hsl_position_during_cooldown_policy" not in result["live"]
    assert migrate(result) == result


def test_enabled_legacy_threshold_cannot_be_guessed():
    with pytest.raises(ValueError, match="explicit choice"):
        migrate(legacy())


def test_unified_policy_is_not_copied_from_side_or_template():
    cfg = legacy("unified")
    with pytest.raises(ValueError, match="explicit bot.hsl"):
        migrate(cfg)
    policy = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    policy.update(enabled=True, restart_after_red_policy="never", red_threshold=.123)
    result = migrate(cfg, portfolio=policy)
    assert result["bot"]["hsl"] == policy
    with pytest.raises(ValueError, match="already exists"):
        migrate(result, portfolio=policy)


@pytest.mark.parametrize("choices", [{"bad": "always"}, {"long": "threshold"}, {"portfolio": "always"}])
def test_invalid_or_inactive_choices_fail(choices):
    with pytest.raises(ValueError):
        migrate(legacy(), restart_policies=choices)


def test_retired_search_dimension_is_rejected_not_silently_removed():
    cfg = legacy()
    cfg["optimize"]["fixed_runtime_overrides"]["bot.long.hsl.no_restart_drawdown_threshold"] = .25
    with pytest.raises(ValueError, match="removed"):
        migrate(cfg, restart_policies={"long": "always"})


def test_coin_override_still_requires_its_own_explicit_restart_choice():
    cfg = legacy()
    cfg["coin_overrides"] = {"BTC": {"bot": {"long": {"hsl": {"restart_after_red_policy": "threshold"}}}}}
    with pytest.raises(ValueError, match="coin_overrides.*explicit choice"):
        migrate(cfg, restart_policies={"long": "always"})


def test_nested_current_input_supported_without_mutation():
    cfg = {"config": legacy()}
    original = deepcopy(cfg)
    assert migrate(cfg, restart_policies={"long": "always"})["live"]["hsl_engine"] == "revised"
    assert cfg == original


def test_cli_writes_new_file_and_refuses_overwrite(tmp_path, capsys):
    src, dst = tmp_path/"source.json", tmp_path/"converted.json"
    src.write_text(json.dumps(legacy()))
    before = src.read_bytes()
    assert main([str(src), str(dst), "--restart-policy", "long=always"]) == 0
    assert src.read_bytes() == before
    assert json.loads(dst.read_text())["live"]["hsl_engine"] == "revised"
    assert "not equivalent" in capsys.readouterr().err
    result = dst.read_bytes()
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=never"])
    assert dst.read_bytes() == result
    with pytest.raises(SystemExit):
        main([str(src), str(src), "--restart-policy", "long=always"])
    assert src.read_bytes() == before


@pytest.mark.parametrize("extra", [[], ["--restart-policy", "long=always", "--restart-policy", "long=never"]])
def test_failed_cli_does_not_create_output(tmp_path, extra):
    src, dst = tmp_path/"source.json", tmp_path/"converted.json"
    src.write_text(json.dumps(legacy()))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), *extra])
    assert not dst.exists()


@pytest.mark.parametrize('patch', [
    {'restart_after_red_policy': 'threshold'},
    {'no_restart_drawdown_threshold': .25},
    {'red_threshold': 0.0},
])
def test_file_backed_invalid_policy_fails_before_output(tmp_path, patch):
    src, dst, override = tmp_path/'source.json', tmp_path/'output.json', tmp_path/'coin.json'
    override.write_text(json.dumps({'bot': {'long': {'hsl': patch}}}))
    cfg = legacy()
    cfg['coin_overrides'] = {'BTC': {'override_config_path': str(override)}}
    src.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), '--restart-policy', 'long=always'])
    assert not dst.exists()


def test_relative_override_materialized_with_inline_precedence_and_reloaded_elsewhere(tmp_path, monkeypatch):
    from config.load import load_prepared_config
    from config.overrides import parse_overrides
    source_dir, output_dir, cwd = (tmp_path/p for p in ('source', 'output', 'cwd'))
    for folder in (source_dir, output_dir, cwd):
        folder.mkdir()
    patch = {'bot': {'long': {'hsl': {'red_threshold': .13, 'restart_after_red_policy': 'never'}}}}
    override = source_dir/'coin.json'
    override.write_text(json.dumps(patch))
    # A different same-named file must never supply the converted policy.
    (output_dir/'coin.json').write_text(json.dumps({'bot': {'long': {'hsl': {'red_threshold': .9}}}}))
    cfg = legacy()
    cfg['coin_overrides'] = {'BTC': {'override_config_path': 'coin.json',
                                    'bot': {'long': {'hsl': {'red_threshold': .17}}}}}
    src, dst = source_dir/'source.json', output_dir/'converted.json'
    src.write_text(json.dumps(cfg))
    original = override.read_bytes()
    monkeypatch.chdir(cwd)
    assert main([str(src), str(dst), '--restart-policy', 'long=always']) == 0
    assert override.read_bytes() == original
    output = json.loads(dst.read_text())
    assert 'override_config_path' not in output['coin_overrides']['BTC']
    override.unlink()  # Result must be self-contained, not dependent on old files.
    reloaded = parse_overrides(load_prepared_config(str(dst), verbose=False), verbose=False)
    assert reloaded['coin_overrides']['BTC']['bot']['long']['hsl'] == {
        'red_threshold': .17, 'restart_after_red_policy': 'never'}


@pytest.mark.parametrize('mode', ['Unified', ' unified '])
def test_canonical_mode_normalization_precedes_scope_choices(mode):
    cfg = legacy('unified')
    cfg['live']['hsl_signal_mode'] = mode
    policy = generated_template(get_template_config(), 'unified')['bot']['hsl']
    result = migrate(cfg, portfolio=policy, restart_policies={'portfolio': 'never'})
    assert result['live']['hsl_signal_mode'] == 'unified'
    assert result['bot']['hsl']['restart_after_red_policy'] == 'never'
    with pytest.raises(ValueError, match='side restart choices are inactive'):
        migrate(cfg, portfolio=policy, restart_policies={'long': 'always'})


@pytest.mark.parametrize('alias', ['bot.long.hsl.restart_after_red_policy', 'long.hsl.restart_after_red_policy',
                                   'bot.long.hsl_restart_after_red_policy'])
def test_explicit_restart_choice_survives_optimizer_fixed_override(alias):
    from optimization.warmup import _apply_config_overrides
    cfg = legacy()
    cfg['optimize']['fixed_runtime_overrides'] = {alias: 'always'}
    result = migrate(cfg, restart_policies={'long': 'never'})
    assert result['optimize']['fixed_runtime_overrides'][alias] == 'never'
    effective = deepcopy(result)
    _apply_config_overrides(effective, effective["optimize"]["fixed_runtime_overrides"])
    assert effective['bot']['long']['hsl']['restart_after_red_policy'] == 'never'


@pytest.mark.parametrize('selector', ['long.hsl.no_restart_drawdown_threshold',
                                    '*.hsl.tier_ratios', 'long_hsl_no_restart_drawdown_threshold',
                                    'long.hsl.misspelled_threshold'])
def test_inactive_fixed_parameter_selector_rejected(selector):
    cfg = legacy()
    cfg['optimize']['fixed_params'] = [selector]
    with pytest.raises(ValueError, match='fixed_params|removed'):
        migrate(cfg, restart_policies={'long': 'always'})


def test_valid_fixed_groups_and_leaf_selectors_remain_supported():
    cfg = legacy()
    cfg['optimize']['fixed_params'] = ['long.hsl', '*.hsl.red_threshold', 'short.risk']
    result = migrate(cfg, restart_policies={'long': 'always'})
    assert result['optimize']['fixed_params'] == cfg['optimize']['fixed_params']


@pytest.mark.parametrize("mode,scope", [("pside", "short"), ("unified", "portfolio")])
def test_restart_choice_updates_only_matching_optimizer_scope(mode, scope):
    cfg = legacy(mode)
    policy = generated_template(get_template_config(), "unified")["bot"]["hsl"] if mode == "unified" else None
    chosen = "bot.hsl.restart_after_red_policy" if scope == "portfolio" else "bot.short.hsl.restart_after_red_policy"
    fixed = {chosen: "always", "bot.long.risk.n_positions": 3}
    cfg["optimize"]["fixed_runtime_overrides"] = fixed
    choices = {scope: "never"}
    if mode == "pside":
        choices["long"] = "always"
    result = migrate(cfg, portfolio=policy, restart_policies=choices)
    assert result["optimize"]["fixed_runtime_overrides"] == {**fixed, chosen: "never"}
    assert cfg["optimize"]["fixed_runtime_overrides"][chosen] == "always"


def test_missing_override_file_does_not_create_output(tmp_path):
    cfg = legacy()
    cfg["coin_overrides"] = {"BTC": {"override_config_path": "missing.json"}}
    src, dst = tmp_path/"source.json", tmp_path/"output.json"
    src.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=always"])
    assert not dst.exists()


def test_valid_selector_cannot_mask_unmatched_selector():
    cfg = legacy()
    cfg["optimize"]["fixed_params"] = ["long.hsl", "short.hsl.missing"]
    with pytest.raises(ValueError, match="short.hsl.missing.*matches no active bounds"):
        migrate(cfg, restart_policies={"long": "always"})
