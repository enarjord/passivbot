"""Final risk authoring and optimizer boundaries."""

from copy import deepcopy
import json

import pytest

from config import compile_runtime_config, prepare_config
from config.authoring import compact_config
from config.inherited_bounds import get_bound_plan, resolve_plan_selectors
from config.risk_memory import (
    DEFAULT_REALIZED_LOSS_MEMORY_EMA_SPAN_MINUTES,
    LOSS_GATE_CAP_PATH,
    require_loss_realization_policy,
    risk_memory_spans,
)
from config.schema import get_template_config
from config_utils import clean_config
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


CAP = LOSS_GATE_CAP_PATH[-1]
GATE_DOTTED = ".".join(LOSS_GATE_CAP_PATH[:-1])


def risk(config):
    return config["bot"]["portfolio"]["risk"]["loss_realization"]


def cap_bounds(config, bounds):
    config["optimize"]["bounds"]["portfolio"] = {
        "risk": {"loss_realization": {"loss_gate": {CAP: bounds}}},
    }


def test_gate_authority_memory_activation_and_zero_round_trip():
    source = get_template_config()
    source["bot"]["global"]["risk"]["unstuck"]["enabled"] = False
    for authored_gate, policy, spans in (
        ({"enabled": False}, (False, None), ()),
        ({"enabled": False, CAP: 0}, (False, 0), ()),
        ({"enabled": True, CAP: 0}, (True, 0), (DEFAULT_REALIZED_LOSS_MEMORY_EMA_SPAN_MINUTES,)),
    ):
        risk(source)["loss_gate"] = authored_gate
        prepared = prepare_config(source, verbose=False)
        assert require_loss_realization_policy(prepared)[1:] == policy
        assert risk_memory_spans(prepared) == spans
        assert risk(clean_config(prepared))["loss_gate"] == authored_gate
        assert risk(compact_config(source))["loss_gate"] == authored_gate


def test_runtime_projection_rejects_a_missing_enablement_flag():
    prepared = prepare_config(get_template_config(), verbose=False)
    del risk(prepared)["loss_gate"]["enabled"]
    with pytest.raises(ValueError, match="explicit boolean"):
        compile_runtime_config(prepared)


@pytest.mark.parametrize("cap", [None, -0.1, 1, float("nan"), True])
def test_even_dormant_authored_caps_validate(cap):
    config = get_template_config()
    risk(config)["loss_gate"] = {"enabled": False, CAP: cap}
    with pytest.raises((TypeError, ValueError), match=CAP):
        prepare_config(config, verbose=False)


def test_only_eligible_scenarios_open_an_explicit_cap_coordinate():
    source = get_template_config()
    cap_bounds(source, [0, 0.1])
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "off"}])
    prepared = prepare_config(source, verbose=False)
    assert LOSS_GATE_CAP_PATH not in {b.authoring_path for b in get_bound_plan(prepared).bindings}
    source["backtest"]["scenarios"].append({
        "label": "enabled",
        "overrides": {GATE_DOTTED + ".enabled": True, GATE_DOTTED + "." + CAP: 0},
    })
    prepared = prepare_config(source, verbose=False)
    cap = next(b for b in get_bound_plan(prepared).bindings if b.authoring_path == LOSS_GATE_CAP_PATH)
    assert (cap.bound.low, cap.bound.high) == (0, 0.1)
    memory = next(b for b in get_bound_plan(prepared).bindings
                  if b.native_ids == ("realized_loss_memory_ema_span_minutes",))
    assert memory.bound.low == memory.bound.high == DEFAULT_REALIZED_LOSS_MEMORY_EMA_SPAN_MINUTES
    del source["backtest"]["scenarios"][1]["overrides"][GATE_DOTTED + "." + CAP]
    with pytest.raises(ValueError, match="required when the loss gate is enabled"):
        prepare_config(source, verbose=False)


def test_retired_sources_rejected_before_default_resolution(tmp_path):
    source = get_template_config()
    for retired in (
        {"bot.long.unstuck.enabled": False},
        {"bot": {"long": {"hsl": {}}}},
        {"bot.long.risk.unstuck.close_fraction": 0},
        {"bot.portfolio.loss_gate": {}},
    ):
        config = deepcopy(source)
        config["backtest"]["scenarios"] = [{"label": "old", "overrides": retired}]
        with pytest.raises(ValueError, match="risk group|removed|shared span"):
            prepare_config(config, verbose=False)
    prepared = prepare_config(source, verbose=False)
    with pytest.raises(ValueError, match="removed"):
        resolve_plan_selectors(prepared, ["long_unstuck_close_fraction"])
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"bot": {"long": {"risk": {"unstuck": {"close_fraction": 0.1}}}}}))
    source["coin_overrides"] = {"BTC": {"override_config_path": str(file)}}
    with pytest.raises(ValueError, match="close_fraction was removed"):
        prepare_config(source, verbose=False)


def test_v8_gate_off_omits_cap_and_zero_keeps_explicit_enablement():
    source = released_v8_policy()
    source["live"]["max_realized_loss_pct"] = 1
    output = migrate(source, memory_span_minutes=43200).config
    assert risk(output)["loss_gate"] == {"enabled": False}
    source["live"]["max_realized_loss_pct"] = 0
    with pytest.raises(ValueError, match="explicit --lookback-ema-span-minutes"):
        migrate(source, best_effort=True, acknowledge_realized_loss_gate_memory_change=True)
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate(source, best_effort=True, memory_span_minutes=43200)
    output = migrate(source, memory_span_minutes=43200,
                     acknowledge_realized_loss_gate_memory_change=True).config
    assert risk(output)["loss_gate"] == {"enabled": True, CAP: 0}


def test_v8_reports_removed_unstuck_size_and_rejects_mixed_bounds():
    source = released_v8_policy()
    source["bot"]["long"]["unstuck"]["close_pct"] = 0
    source["optimize"]["bounds"]["long"]["unstuck"]["close_pct"] = [0, 0]
    result = migrate(source, memory_span_minutes=43200)
    assert result.config["bot"]["long"]["risk"]["unstuck"]["enabled"] is False
    assert "close_fraction" not in result.config["optimize"]["bounds"]["long"]["risk"]["unstuck"]
    assert any("discarded the retired unstuck size" in item["reason"] for item in result.changes)
    source["optimize"]["bounds"]["long"]["unstuck"]["close_pct"] = [0, 0.1]
    with pytest.raises(ValueError, match="both disabled and enabled"):
        migrate(source, memory_span_minutes=43200)


def test_optimizer_candidates_supply_only_enabled_scenarios():
    from types import SimpleNamespace

    from optimization.warmup import build_optimizer_max_config
    from optimize import SuiteEvaluator
    from suite_runner import apply_scenario_overrides

    source = get_template_config()
    cap_bounds(source, [0, 0.1])
    enabled = {GATE_DOTTED + ".enabled": True, GATE_DOTTED + "." + CAP: 0}
    cases = [{"label": "off"}, {"label": "on", "overrides": enabled}]
    source["backtest"].update(suite_enabled=True, scenarios=cases)
    prepared = prepare_config(source, verbose=False)
    candidate = build_optimizer_max_config(prepared)
    evaluator = SimpleNamespace(
        base=SimpleNamespace(config=prepared, scoring_specs=[], limit_checks=[]),
        objective_bases=[],
    )

    for scenario, expected in zip(cases, [(False, None), (True, 0.1)], strict=True):
        # Dataset preparation can use the maximal genome. It must not supply
        # authored permissions or a dormant cap to the evaluated scenario.
        context_config = deepcopy(candidate)
        overrides = scenario.get("overrides", {})
        apply_scenario_overrides(context_config, overrides)
        context = SimpleNamespace(label=scenario["label"], config=context_config,
                                  overrides=overrides, msss={})
        actual = SuiteEvaluator.build_scenario_candidate_config(evaluator, candidate, context)
        assert require_loss_realization_policy(actual)[1:] == expected


def test_v8_mixed_referenced_and_inline_sizes_require_an_explicit_choice(tmp_path):
    source = released_v8_policy()
    sidecar = tmp_path / "coin.json"
    sidecar.write_text(json.dumps({"bot": {"long": {"unstuck": {"close_pct": 0}}}}))
    policy = {"close_pct": 0.08}
    source["coin_overrides"] = {
        "BTC": {"override_config_path": str(sidecar.name), "bot": {"long": {"unstuck": policy}}},
    }
    with pytest.raises(ValueError, match="overrides a disabled v8 coin-file size"):
        migrate(source, memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    policy["enabled"] = True
    output = migrate(source, memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json")).config
    assert output["coin_overrides"]["BTC"]["bot"]["long"]["risk"]["unstuck"] == {"enabled": True}
