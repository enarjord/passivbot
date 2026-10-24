"""Preflight and diff must retain loader HSL validation in inactive topologies."""

from copy import deepcopy
import json
import re

import pytest

from config import prepare_config
from tools.live_config_preflight import build_live_config_diff_report, build_live_config_preflight_report


def _policy(enabled=False):
    return {"enabled": enabled, "scale_budget_with_excess_allowance": False,
            "drawdown_ratio_threshold": 0.05, "drawdown_ratio_ema_span_minutes": 120.0,
            "memory_ema_span_minutes": 43200.0, "panic_close_order_type": "limit",
            "cooldown_after_red_minutes": 60.0}


def _config(scope="portfolio"):
    return {"config_version": "v9.0.0",
            "live": {"account": {"balance_override": None},
                     "startup": {"risk_bootstrap_max_lookback_minutes": 129600.0}},
            "bot": {"portfolio": {"hsl": {"scope": scope, **(_policy() if scope == "portfolio" else {})}},
                    "global": {"hsl": _policy()}}}


def _inspect(tmp_path, baseline, target, inspection):
    paths = []
    for name, value in (("baseline", baseline), ("target", target)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    original = [path.read_bytes() for path in paths]
    report = (build_live_config_preflight_report(paths[1]) if inspection == "preflight"
              else build_live_config_diff_report(*paths))
    assert [path.read_bytes() for path in paths] == original
    return report


@pytest.mark.parametrize("inspection", ["preflight", "diff"])
@pytest.mark.parametrize("scope", ["global", "long", "short"])
@pytest.mark.parametrize("field,value,message", [
    ("drawdown_ratio_threshold", -0.1, "outside its finite supported range"),
    ("scale_budget_with_excess_allowance", True, "requires coin HSL mode"),
])
def test_portfolio_scope_still_validates_resolved_directional_hsl(tmp_path, inspection, scope, field, value, message):
    baseline = _config()
    target = deepcopy(baseline)
    target["bot"].setdefault(scope, {}).setdefault("hsl", {})[field] = value
    resolved_side = "long" if scope == "global" else scope
    path = f"bot.{resolved_side}.hsl.{field}"
    # Normal startup already rejects these policies even though entry and HSL
    # are disabled on the directional side. Inspection must report the same error.
    with pytest.raises(ValueError, match=re.escape(path)):
        prepare_config(deepcopy(target), live_only=True, verbose=False)
    report = _inspect(tmp_path, baseline, target, inspection)
    assert report["ok"] is False
    assert any(path in issue["message"] and message in issue["message"]
               for issue in report["issues"])
    assert "hsl" not in report


@pytest.mark.parametrize("inspection", ["preflight", "diff"])
@pytest.mark.parametrize("scope", ["global", "long", "short"])
def test_position_side_scope_rejects_coin_only_directional_scaling(tmp_path, inspection, scope):
    baseline = _config("position_side")
    target = deepcopy(baseline)
    target["bot"].setdefault(scope, {}).setdefault("hsl", {})["scale_budget_with_excess_allowance"] = True
    resolved_side = "long" if scope == "global" else scope
    path = f"bot.{resolved_side}.hsl.scale_budget_with_excess_allowance"
    with pytest.raises(ValueError, match=re.escape(path)):
        prepare_config(deepcopy(target), live_only=True, verbose=False)
    report = _inspect(tmp_path, baseline, target, inspection)
    assert report["ok"] is False
    assert any(path in issue["message"] and "requires coin HSL mode" in issue["message"]
               for issue in report["issues"])


@pytest.mark.parametrize("inspection", ["preflight", "diff"])
@pytest.mark.parametrize("enabled", [False, True])
def test_portfolio_scaling_rejection_matches_loader_even_when_disabled(tmp_path, inspection, enabled):
    baseline = _config()
    target = deepcopy(baseline)
    target["bot"]["portfolio"]["hsl"].update(enabled=enabled, scale_budget_with_excess_allowance=True)
    path = "bot.portfolio.hsl.scale_budget_with_excess_allowance"
    with pytest.raises(ValueError, match=re.escape(path)):
        prepare_config(deepcopy(target), live_only=True, verbose=False)
    report = _inspect(tmp_path, baseline, target, inspection)
    assert report["ok"] is False
    assert any(path in issue["message"] and "requires coin HSL mode" in issue["message"]
               for issue in report["issues"])


@pytest.mark.parametrize("enabled", [False, True])
def test_coin_scoped_scaling_remains_valid_with_activation_disabled(tmp_path, enabled):
    config = _config("coin")
    config["bot"]["global"]["hsl"].update(enabled=enabled, scale_budget_with_excess_allowance=True)
    prepared = prepare_config(deepcopy(config), live_only=True, verbose=False)
    assert all(prepared["bot"][side]["enabled"] is False for side in ("long", "short"))
    report = _inspect(tmp_path, config, config, "preflight")
    assert report["ok"] is True
    assert all(policy["enabled"] is enabled and policy["scale_budget_with_excess_allowance"] is True
               for policy in report["hsl"]["policies"].values())
