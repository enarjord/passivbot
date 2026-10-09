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
def test_retired_mode_is_removed_or_fails_with_actionable_context(
    location, mode, caplog
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
                "name": "old",
                "overrides": {"bot.long.risk.we_excess_allowance_mode": mode},
            }
        ]
    else:
        source["bot"]["long"]["risk"].update(risk)
        source = {"config": source}
    original = deepcopy(source)
    if "LEGACY_RAW" in mode:
        with pytest.raises(
            ValueError,
            match="no longer supported.*re-backtest.*cannot be preserved automatically",
        ):
            prepare_config(source, verbose=False)
        assert "we_excess_allowance_mode" in caplog.text
    else:
        result = prepare_config(source, verbose=False)
        assert "we_excess_allowance_mode" not in result["bot"]["long"]["risk"]
        assert "Removed obsolete" in caplog.text
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
    if mode == "legacy_raw":
        with pytest.raises(ValueError, match="legacy_raw.*no longer supported"):
            parse_overrides(prepare_config(source, verbose=False), verbose=False)
    else:
        result = parse_overrides(prepare_config(source, verbose=False), verbose=False)
        assert result["coin_overrides"]["BTC"]["bot"]["long"]["risk"] == {
            "we_excess_allowance_pct": 0.5
        }
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


def test_coin_scaling_switch_is_global_per_side():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"hsl": {"scale_budget_with_excess_allowance": True}}}}
    }
    with pytest.raises(ValueError, match="global per-side"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("mode", ["pside", "unified"])
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
def test_cleanup_cannot_drop_a_raw_policy_before_validation(surface, old_value):
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
    if old_value == "legacy_raw":
        with pytest.raises(ValueError, match="no longer supported.*re-backtest"):
            clean()
    else:
        assert "we_excess_allowance_mode" not in clean()["bot"]["long"]["risk"]
    assert source == original
