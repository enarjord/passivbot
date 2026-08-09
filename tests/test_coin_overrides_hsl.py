from copy import deepcopy

import pytest

from config.load import prepare_config
from config.overrides import CONDITIONAL_HSL_OVERRIDE_PATHS, parse_overrides
from config.schema import get_template_config


def _parse(overrides, *, mode="coin", loaded=None, configure_source=None):
    source = get_template_config()
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
    (("no_restart_drawdown_threshold",), 0.8),
    (("orange_tier_mode",), "graceful_stop"),
    (("panic_close_order_type",), "market"),
    (("red_threshold",), 0.2),
    (("restart_after_red_policy",), "always"),
    (("tier_ratios", "orange"), 0.8),
    (("tier_ratios", "yellow"), 0.4),
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
        "hsl.no_restart_drawdown_threshold",
        "hsl.orange_tier_mode",
        "hsl.panic_close_order_type",
        "hsl.red_threshold",
        "hsl.restart_after_red_policy",
        "hsl.tier_ratios.orange",
        "hsl.tier_ratios.yellow",
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
        match=rf"coin_overrides\.BTC\.bot\.long\.hsl.*only when.*got '{mode}'",
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
    with pytest.raises(ValueError, match=r"effective global.*got 'pside'"):
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
                        "tier_ratios": {"yellow": 0.4},
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
    assert bot["long"]["hsl"]["tier_ratios"]["yellow"] == 0.4
    assert bot["short"]["hsl"]["panic_close_order_type"] == "market"
    assert bot["short"]["hsl"]["restart_after_red_policy"] == "always"


def test_hsl_dependent_normalization_is_retained_in_resolved_patch():
    def configure(source):
        source["bot"]["long"]["hsl"]["red_threshold"] = 0.05
        source["bot"]["long"]["hsl"]["no_restart_drawdown_threshold"] = 0.1

    parsed = _parse(
        {"BTC": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}},
        configure_source=configure,
    )

    hsl = parsed["coin_overrides"]["BTC"]["bot"]["long"]["hsl"]
    assert hsl["red_threshold"] == 0.2
    assert hsl["no_restart_drawdown_threshold"] == 0.2


def test_hsl_dependent_normalization_runs_after_file_inline_precedence():
    def configure(source):
        source["bot"]["long"]["hsl"]["red_threshold"] = 0.05
        source["bot"]["long"]["hsl"]["no_restart_drawdown_threshold"] = 0.1

    parsed = _parse(
        {
            "BTC": {
                "override_config_path": "unused-by-test.json",
                "bot": {"long": {"hsl": {"red_threshold": 0.05}}},
            }
        },
        loaded={"bot": {"long": {"hsl": {"red_threshold": 0.2}}}},
        configure_source=configure,
    )

    hsl = parsed["coin_overrides"]["BTC"]["bot"]["long"]["hsl"]
    assert hsl["red_threshold"] == 0.05
    assert "no_restart_drawdown_threshold" not in hsl


@pytest.mark.parametrize(
    "hsl_patch",
    [
        {"tier_ratios": {"yellow": 0.0, "orange": 0.8}},
        {"tier_ratios": {"yellow": 0.8, "orange": 0.8}},
        {"tier_ratios": {"yellow": 0.8, "orange": 1.0}},
        {"tier_ratios": {"yellow": 0.9, "orange": 0.8}},
        {"orange_tier_mode": "invalid"},
        {"panic_close_order_type": "invalid"},
        {"red_threshold": 1.1},
    ],
)
def test_invalid_hsl_combinations_fail_effective_validation(hsl_patch):
    with pytest.raises(
        ValueError, match=r"coin_overrides\.BTC produces an invalid config"
    ):
        _parse({"BTC": {"bot": {"long": {"hsl": hsl_patch}}}})


def test_unknown_hsl_tier_ratio_is_rejected_at_patch_boundary():
    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC\.bot\.long\.hsl\.tier_ratios\.red is not overridable",
    ):
        _parse({"BTC": {"bot": {"long": {"hsl": {"tier_ratios": {"red": 1.0}}}}}})
