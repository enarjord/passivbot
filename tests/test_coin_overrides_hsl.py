from copy import deepcopy

import pytest

from config.load import prepare_config
from config.overrides import CONDITIONAL_HSL_OVERRIDE_PATHS, parse_overrides
from config.schema import get_template_config
from config.hsl import generated_template


def _parse(overrides, *, mode="coin", loaded=None, configure_source=None):
    source = generated_template(get_template_config(), mode)
    source["live"]["user"] = "tester"
    source["live"]["hsl_signal_mode"] = mode
    if configure_source is not None:
        configure_source(source)
    source["coin_overrides"] = deepcopy(overrides)
    prepared = prepare_config(source, verbose=False, log_config_transforms=False)
    return parse_overrides(
        prepared,
        verbose=False,
        override_loader=lambda config, coin: deepcopy(loaded or {}),
        symbol_normalizer=lambda coin: coin,
    )


HSL_LEAF_CASES = [
    (("cooldown_minutes_after_red",), 12.5),
    (("ema_span_minutes",), 5.5),
    (("enabled",), False),
    (("panic_close_order_type",), "market"),
    (("red_threshold",), 0.2),
    (("restart_after_red_policy",), "always"),
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
        "hsl.cooldown_minutes_after_red",
        "hsl.ema_span_minutes",
        "hsl.enabled",
        "hsl.panic_close_order_type",
        "hsl.red_threshold",
        "hsl.restart_after_red_policy",
    }


@pytest.mark.parametrize("mode", ["pside", "unified"])
@pytest.mark.parametrize("spelling", ["grouped", "flat"])
def test_inline_hsl_patch_is_rejected_outside_coin_mode(mode, spelling):
    side = (
        {"hsl": {"red_threshold": 0.2}}
        if spelling == "grouped"
        else {"hsl_red_threshold": 0.2}
    )

    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC\.bot\.long\.hsl.*(inactive|only when)",
    ):
        _parse({"BTC": {"bot": {"long": side}}}, mode=mode)


def test_full_file_hsl_patch_is_warned_and_ignored_outside_coin_mode(caplog):
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        mode="pside",
        loaded={
            "bot": {
                "long": {
                    "hsl": {"red_threshold": 0.2},
                    "unstuck": {"loss_allowance_pct": 0.023},
                }
            }
        },
    )

    side = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    assert "hsl" not in side
    assert side["unstuck"]["loss_allowance_pct"] == 0.023
    assert "file HSL values are ignored" in caplog.text


@pytest.mark.parametrize(
    ("global_mode", "file_mode", "accepted"),
    [("coin", "unified", True), ("pside", "coin", False)],
)
def test_global_signal_mode_wins_over_override_file_mode(
    global_mode, file_mode, accepted, caplog
):
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        mode=global_mode,
        loaded={
            "live": {"hsl_signal_mode": file_mode},
            "bot": {"short": {"hsl": {"panic_close_order_type": "market"}}},
        },
    )

    short = parsed["coin_overrides"]["BTC"].get("bot", {}).get("short", {})
    assert ("hsl" in short) is accepted
    if accepted:
        assert short["hsl"]["panic_close_order_type"] == "market"
    else:
        assert "file HSL values are ignored" in caplog.text


def test_inline_cannot_switch_signal_mode_to_authorize_hsl_patch():
    with pytest.raises(
        ValueError, match=r"(effective global|inactive outside coin mode)"
    ):
        _parse(
            {
                "BTC": {
                    "live": {"hsl_signal_mode": "coin"},
                    "bot": {"long": {"hsl": {"enabled": False}}},
                }
            },
            mode="pside",
        )


def test_file_then_inline_hsl_precedence_is_independent_by_side():
    parsed = _parse(
        {
            "BTC": {
                "override_config_path": "unused-by-test.json",
                "bot": {
                    "long": {"hsl": {"red_threshold": 0.25}},
                    "short": {"hsl_panic_close_order_type": "market"},
                },
            }
        },
        loaded={
            "bot": {
                "long": {
                    "hsl": {
                        "red_threshold": 0.15,
                        "ema_span_minutes": 4.5,
                    }
                },
                "short": {
                    "hsl": {
                        "panic_close_order_type": "limit",
                        "restart_after_red_policy": "always",
                    }
                },
            }
        },
    )

    bot = parsed["coin_overrides"]["BTC"]["bot"]
    assert bot["long"]["hsl"]["red_threshold"] == 0.25
    assert bot["long"]["hsl"]["ema_span_minutes"] == 4.5
    assert bot["short"]["hsl"]["panic_close_order_type"] == "market"
    assert bot["short"]["hsl"]["restart_after_red_policy"] == "always"


def test_threshold_override_does_not_add_unrequested_policy_fields():
    parsed = _parse({"BTC": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}})
    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["hsl"] == {
        "red_threshold": 0.2
    }


@pytest.mark.parametrize(
    "hsl_patch",
    [
        {"panic_close_order_type": "invalid"},
        {"red_threshold": 1.1},
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
