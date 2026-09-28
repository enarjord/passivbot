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
