"""Released config removal and explicit opt-in loss-budget policy."""

from copy import deepcopy
import json

import pytest
from config.load import prepare_config
from config.schema import get_template_config
from config.overrides import parse_overrides
from risk_limits import (
    wallet_exposure_limit_with_allowance,
)


@pytest.mark.parametrize("location", ["root", "coin", "fixed", "scenario", "wrapper"])
@pytest.mark.parametrize("mode", [" BOUNDED ", " LEGACY_RAW "])
def test_current_schema_rejects_retired_mode_on_all_authored_surfaces(
    location, mode
):
    source = get_template_config()
    risk = {"we_excess_allowance_mode": mode}
    if location == "root":
        source["bot"]["long"]["risk"].update(risk)
    elif location == "coin":
        source["coin_overrides"] = {"BTC": {"bot": {"long": {"risk": risk}}}}
    elif location == "fixed":
        source["optimize"]["fixed_runtime_overrides"][
            "bot.long.risk.we_excess_allowance_mode"
        ] = mode
    elif location == "scenario":
        source["backtest"]["scenarios"] = [
            {
                "label": "old",
                "overrides": {"bot.long.risk.we_excess_allowance_mode": mode},
            }
        ]
    else:
        source["bot"]["long"]["risk"].update(risk)
        source = {"config": source}
    original = deepcopy(source)
    match = "config_version.*explicit migration" if location == "wrapper" else "we_excess_allowance_mode"
    with pytest.raises(ValueError, match=match):
        prepare_config(source, verbose=False)
    assert source == original


@pytest.mark.parametrize("mode", ["bounded", "legacy_raw"])
def test_external_config_cannot_hide_removed_raw_policy(tmp_path, mode):
    path = tmp_path / "coin.json"
    path.write_text(
        json.dumps(
            {
                "bot": {
                    "long": {
                        "risk": {
                            "we_excess_allowance_mode": mode,
                            "we_excess_allowance_pct": 0.5,
                        }
                    }
                }
            }
        )
    )
    source = get_template_config()
    source["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
    with pytest.raises(ValueError, match=r"coin_overrides\.BTC\.override_config_path.*config\.bot\.long\.risk\.we_excess_allowance_mode is an unknown configuration field"):
        parse_overrides(prepare_config(source, verbose=False), verbose=False)
    assert (
        json.loads(path.read_text())["bot"]["long"]["risk"]["we_excess_allowance_mode"]
        == mode
    )


@pytest.mark.parametrize("value", [True, False, "true", 1, None])
def test_scaling_is_strict_boolean_and_defaults_off(value):
    source = get_template_config()
    assert source["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"] is False
    source["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"] = value
    if type(value) is bool:
        result = prepare_config(source, verbose=False)
        assert (
            result["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"] is value
        )
    else:
        with pytest.raises(TypeError, match="must be a boolean"):
            prepare_config(source, verbose=False)


def test_coin_scaling_switch_is_local_and_preserves_base_policy():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"hsl": {"scale_budget_with_excess_allowance": True}}}}
    }
    original = deepcopy(source)
    prepared = prepare_config(source, verbose=False)
    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"] is True
    assert prepared["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"] is False
    assert source == original


@pytest.mark.parametrize("mode", ["position_side", "portfolio"])
def test_scaling_rejects_aggregate_scopes(mode):
    from config.hsl import generated_template

    source = generated_template(get_template_config(), mode)
    source["bot"]["long"]["hsl"]["scale_budget_with_excess_allowance"] = True
    with pytest.raises(ValueError, match="requires coin HSL mode"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize(
    "base,total,expected",
    [
        (0, 1, 0),
        (float("nan"), 1, 0),
        (0.25, 0, 0.25),
        (0.25, float("inf"), 0.25),
        (0.25, 1, 0.36),
        (1, 1, 1),
    ],
)
def test_python_admission_matches_bounded_rust_contract(base, total, expected):
    assert wallet_exposure_limit_with_allowance(
        wallet_exposure_limit=base,
        total_wallet_exposure_limit=total,
        risk_we_excess_allowance_pct=0.44,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("surface", ["helper", "full", "live", "backtest", "optimize"])
@pytest.mark.parametrize("old_value", ["bounded", "legacy_raw"])
def test_authored_cleanup_rejects_removed_policy_and_runtime_sanitizer_has_no_authority(surface, old_value):
    from config_utils import clean_config
    from tools.clean_config import cleanup_config

    source = get_template_config()
    source["bot"]["long"]["risk"]["we_excess_allowance_mode"] = old_value
    original = deepcopy(source)
    clean = (
        (lambda: clean_config(source))
        if surface == "helper"
        else (lambda: cleanup_config(source, mode=surface))
    )
    if surface == "helper":
        assert "we_excess_allowance_mode" not in clean()["bot"]["long"]["risk"]
        # Internal sanitization does not authorize the original authored policy.
        with pytest.raises(ValueError, match="we_excess_allowance_mode"):
            prepare_config(source, verbose=False)
    else:
        with pytest.raises(ValueError, match="we_excess_allowance_mode"):
            clean()
    assert source == original


@pytest.mark.parametrize("mode", ["bounded", "legacy_raw"])
def test_released_v8_requires_explicit_selector_removal_before_conversion(mode):
    from tools.migrate_config_v9 import migrate as migrate_config_v9

    source = {
        "config_version": "v8.4.0",
        "live": {"strategy_kind": "trailing_martingale", "pnls_max_lookback_days": 30.0},
        "bot": {
            "long": {"risk": {"n_positions": 1.0, "total_wallet_exposure_limit": 1.0,
                                "we_excess_allowance_pct": 0.5, "we_excess_allowance_mode": mode},
                     "hsl": {"enabled": False}},
            "short": {"risk": {"n_positions": 0.0, "total_wallet_exposure_limit": 0.0},
                      "hsl": {"enabled": False}},
        },
        "optimize": {"backend": "pymoo", "bounds": {}},
    }
    original = deepcopy(source)
    with pytest.raises(ValueError, match="we_excess_allowance_mode"):
        migrate_config_v9(source, memory_span_minutes=43200.0)
    reviewed = deepcopy(source)
    reviewed["bot"]["long"]["risk"].pop("we_excess_allowance_mode")
    converted = migrate_config_v9(reviewed, memory_span_minutes=43200.0).config
    prepared = prepare_config(converted, verbose=False)
    assert prepared["bot"]["long"]["risk"]["we_excess_allowance_pct"] == 0.5
    assert "we_excess_allowance_mode" not in prepared["bot"]["long"]["risk"]
    assert source == original
