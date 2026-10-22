from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from config.load import prepare_config
from config.overrides import CONDITIONAL_HSL_OVERRIDE_PATHS, parse_overrides
from config.schema import get_template_config
from config.hsl import generated_template


def _parse(overrides, *, mode="coin", loaded=None, configure_source=None):
    source = generated_template(get_template_config(), mode)
    source["live"]["account"]["user"] = "tester"
    source["bot"]["portfolio"]["hsl"]["scope"] = mode
    if configure_source is not None:
        configure_source(source)
    # Schema validation must see the same real authored file that the resolver
    # sees; a nonexistent test path would now correctly fail before its hook.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "override.json"
        path.write_text(json.dumps(loaded or {}))
        source["coin_overrides"] = deepcopy(overrides)
        for patch in source["coin_overrides"].values():
            if "override_config_path" in patch:
                patch["override_config_path"] = str(path)
        prepared = prepare_config(source, verbose=False, log_config_transforms=False)
        return parse_overrides(
            prepared,
            verbose=False,
            override_loader=lambda config, coin: deepcopy(loaded or {}),
            symbol_normalizer=lambda coin: coin,
        )


HSL_LEAF_CASES = [
    (("cooldown_after_red_minutes",), 12.5),
    (("drawdown_ratio_ema_span_minutes",), 5.5),
    (("memory_ema_span_minutes",), 19.5),
    (("scale_budget_with_excess_allowance",), True),
    (("enabled",), False),
    (("panic_close_order_type",), "market"),
    (("drawdown_ratio_threshold",), 0.2),
]


def _nested_hsl_patch(path, value):
    result = current = {}
    for part in path[:-1]:
        current[part] = {}
        current = current[part]
    current[path[-1]] = value
    return result


def _get_path(value, path):
    current = value
    for part in path:
        current = current[part]
    return current


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize(("path", "value"), HSL_LEAF_CASES)
def test_complete_hsl_group_is_allowed_for_coin_signal_mode(pside, path, value):
    parsed = _parse({"BTC": {"bot": {pside: {"hsl": _nested_hsl_patch(path, value)}}}})

    hsl = parsed["coin_overrides"]["BTC"]["bot"][pside]["hsl"]
    assert _get_path(hsl, path) == value


def test_hsl_policy_registry_covers_every_canonical_leaf():
    assert CONDITIONAL_HSL_OVERRIDE_PATHS == {
        "hsl.cooldown_after_red_minutes",
        "hsl.drawdown_ratio_ema_span_minutes",
        "hsl.memory_ema_span_minutes",
        "hsl.scale_budget_with_excess_allowance",
        "hsl.enabled",
        "hsl.panic_close_order_type",
        "hsl.drawdown_ratio_threshold",
    }


@pytest.mark.parametrize("field,value", [
    ("hsl_red_threshold", 0.2),
    ("unstuck_ema_span_0", 12.5),
    ("risk_we_excess_allowance_pct", 0.1),
])
@pytest.mark.parametrize("location", ["inline", "file"])
def test_final_coin_patch_rejects_retired_flat_shared_fields(field, value, location):
    patch = {"bot": {"long": {field: value}}}
    with pytest.raises(ValueError, match="(?:not overridable|unknown|canonical)"):
        if location == "file":
            _parse({"BTC": {"override_config_path": "override.json"}}, loaded=patch)
        else:
            _parse({"BTC": patch})


@pytest.mark.parametrize("mode", ["position_side", "portfolio"])
@pytest.mark.parametrize("spelling", ["grouped", "flat"])
def test_inline_hsl_patch_is_rejected_outside_coin_mode(mode, spelling):
    side = (
        {"hsl": {"drawdown_ratio_threshold": 0.2}}
        if spelling == "grouped"
        else {"hsl_red_threshold": 0.2}
    )

    with pytest.raises(
        ValueError,
        match=r"bot\.long\.hsl.*(inactive|only when)",
    ):
        _parse({"BTC": {"bot": {"long": side}}}, mode=mode)


def test_full_file_hsl_patch_is_ignored_outside_coin_mode(caplog):
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        mode="position_side",
        loaded={
            "bot": {
                "long": {
                    "hsl": {"drawdown_ratio_threshold": 0.2},
                    "unstuck": {"loss_allowance_fraction": 0.023},
                }
            }
        },
    )
    assert parsed["coin_overrides"]["BTC"] == {
        "bot": {"long": {"unstuck": {"loss_allowance_fraction": 0.023}}}
    }
    assert "hsl is overridable only" in caplog.text
    assert "the file HSL values are ignored" in caplog.text

@pytest.mark.parametrize(
    ("global_mode", "file_mode"),
    [("coin", "portfolio"), ("position_side", "coin")],
)
def test_override_file_cannot_author_global_signal_scope(
    global_mode, file_mode, caplog
):
    loaded = generated_template(get_template_config(), file_mode)
    loaded["bot"]["short"]["hsl"]["panic_close_order_type"] = "market"
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        mode=global_mode,
        loaded=loaded,
    )
    assert parsed["bot"]["portfolio"]["hsl"]["scope"] == global_mode
    patch = parsed["coin_overrides"]["BTC"]["bot"]
    assert "risk" not in patch
    if global_mode == "coin":
        assert patch["short"]["hsl"]["panic_close_order_type"] == "market"
    else:
        assert "hsl" not in patch.get("short", {})
        assert "the file HSL values are ignored" in caplog.text

def test_inline_cannot_switch_signal_mode_to_authorize_hsl_patch():
    with pytest.raises(
        ValueError, match=r"(effective global|inactive outside coin mode|not overridable)"
    ):
        _parse(
            {
                "BTC": {
                    "bot": {"portfolio": {'hsl': {'scope': 'coin'}}, 'long': {'hsl': {'enabled': False}}},
                }
            },
            mode="position_side",
        )


def test_file_then_inline_hsl_precedence_is_independent_by_side():
    parsed = _parse(
        {
            "BTC": {
                "override_config_path": "unused-by-test.json",
                "bot": {
                    "long": {"hsl": {"drawdown_ratio_threshold": 0.25}},
                    "short": {"hsl": {"panic_close_order_type": "market"}},
                },
            }
        },
        loaded={
            "bot": {
                "long": {
                    "hsl": {
                        "drawdown_ratio_threshold": 0.15,
                        "drawdown_ratio_ema_span_minutes": 4.5,
                    }
                },
                "short": {
                    "hsl": {
                        "panic_close_order_type": "limit",
                        "cooldown_after_red_minutes": 17.5,
                    }
                },
            }
        },
    )

    bot = parsed["coin_overrides"]["BTC"]["bot"]
    assert bot["long"]["hsl"]["drawdown_ratio_threshold"] == 0.25
    assert bot["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] == 4.5
    assert bot["short"]["hsl"]["panic_close_order_type"] == "market"
    assert bot["short"]["hsl"]["cooldown_after_red_minutes"] == 17.5


def test_threshold_override_does_not_add_unrequested_policy_fields():
    parsed = _parse({"BTC": {"bot": {"long": {"hsl": {"drawdown_ratio_threshold": 0.2}}}}})
    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["hsl"] == {
        "drawdown_ratio_threshold": 0.2
    }


@pytest.mark.parametrize(
    "hsl_patch",
    [
        {"panic_close_order_type": "invalid"},
        {"drawdown_ratio_threshold": 1.1},
    ],
)
def test_invalid_hsl_combinations_fail_effective_validation(hsl_patch):
    with pytest.raises(ValueError, match=r"coin_overrides\.BTC"):
        _parse({"BTC": {"bot": {"long": {"hsl": hsl_patch}}}})


def test_unknown_hsl_policy_is_rejected_at_patch_boundary():
    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC.*(unknown|not overridable)",
    ):
        _parse({"BTC": {"bot": {"long": {"hsl": {"unknown_policy": 1.0}}}}})
