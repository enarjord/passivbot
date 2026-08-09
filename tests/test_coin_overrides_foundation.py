from copy import deepcopy

import pytest

from config.load import prepare_config
from config.overrides import parse_overrides
from config.runtime_compile import compile_runtime_config
from config.schema import get_template_config
from config.strategy import merge_runtime_bot_side
from passivbot import Passivbot


def _prepared_config(overrides):
    source = get_template_config()
    source["live"]["user"] = "tester"
    source["coin_overrides"] = deepcopy(overrides)
    return prepare_config(source, verbose=False, log_config_transforms=False)


def _parse(overrides, *, loaded=None, symbol_normalizer=lambda coin: coin):
    prepared = _prepared_config(overrides)
    return parse_overrides(
        prepared,
        verbose=False,
        override_loader=lambda config, coin: deepcopy(loaded or {}),
        symbol_normalizer=symbol_normalizer,
    )


def test_explicit_inline_values_equal_to_global_are_retained():
    template = get_template_config()
    threshold = template["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ]
    enabled = template["bot"]["long"]["unstuck"]["enabled"]
    parsed = _parse(
        {
            "BTC": {
                "bot": {
                    "long": {
                        "strategy": {
                            "trailing_martingale": {
                                "entry": {"threshold_base_pct": threshold}
                            }
                        },
                        "unstuck": {"enabled": enabled},
                    }
                }
            }
        }
    )

    override = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    assert (
        override["strategy"]["trailing_martingale"]["entry"]["threshold_base_pct"]
        == threshold
    )
    assert override["unstuck"]["enabled"] is enabled


def test_partial_file_extracts_only_explicit_values_and_inline_wins():
    parsed = _parse(
        {
            "BTC": {
                "override_config_path": "unused-by-test.json",
                "bot": {"long": {"unstuck": {"enabled": False}}},
            }
        },
        loaded={
            "bot": {
                "long": {
                    "unstuck": {
                        "enabled": True,
                        "loss_allowance_pct": 0.02,
                    }
                }
            }
        },
    )

    override = parsed["coin_overrides"]["BTC"]
    assert override == {
        "bot": {
            "long": {
                "unstuck": {
                    "enabled": False,
                    "loss_allowance_pct": 0.02,
                }
            }
        }
    }


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ("false", TypeError, "must be numeric or boolean, not a string"),
        (None, TypeError, "may not be null"),
    ],
)
def test_invalid_boolean_override_types_fail(value, error, message):
    with pytest.raises(error, match=message):
        _parse({"BTC": {"bot": {"long": {"unstuck": {"enabled": value}}}}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_override_values_fail(value):
    with pytest.raises(ValueError, match="must be finite"):
        _parse(
            {
                "BTC": {
                    "bot": {
                        "long": {
                            "strategy": {
                                "trailing_martingale": {
                                    "entry": {"threshold_base_pct": value}
                                }
                            }
                        }
                    }
                }
            }
        )


def test_effective_config_cross_field_validation_fails_with_coin_context():
    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC produces an invalid config after .*threshold.*> 0",
    ):
        _parse(
            {
                "BTC": {
                    "bot": {
                        "long": {
                            "risk": {
                                "position_exposure_enforcer_enabled": True,
                                "position_exposure_enforcer_threshold": 0.0,
                            }
                        }
                    }
                }
            }
        )


def test_unstuck_ema_distance_validation_applies_to_effective_config():
    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC produces an invalid config after file and inline precedence resolution",
    ):
        _parse({"BTC": {"bot": {"long": {"unstuck": {"ema_dist": -1.0}}}}})


def test_effective_validation_normalization_is_retained_in_patch():
    parsed = _parse({"BTC": {"bot": {"long": {"unstuck": {"threshold": 1e-12}}}}})

    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["threshold"] == 0.0


def test_runtime_compilation_must_follow_override_parsing():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"filter_volume_drop_pct": 0.25}}}
    }
    runtime_config = prepare_config(
        source,
        verbose=False,
        log_config_transforms=False,
        target="live",
        runtime="live",
    )

    with pytest.raises(ValueError, match=r"must run before compile_runtime_config"):
        parse_overrides(runtime_config, verbose=False)

    canonical_config = prepare_config(
        source,
        verbose=False,
        log_config_transforms=False,
        target="live",
    )
    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC\.bot\.long\.filter_volume_drop_pct is not overridable",
    ):
        parse_overrides(canonical_config, verbose=False)


def test_runtime_compilation_after_parsing_preserves_normalized_patch():
    parsed = _parse({"BTC": {"bot": {"long": {"unstuck": {"threshold": 1e-12}}}}})

    runtime_config = compile_runtime_config(parsed, runtime="live")

    assert (
        runtime_config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["threshold"]
        == 0.0
    )


def test_unknown_and_disallowed_inline_paths_fail():
    with pytest.raises(ValueError, match=r"coin_overrides\.BTC\.bot\.long\.mystery"):
        _parse({"BTC": {"bot": {"long": {"mystery": 1.0}}}})
    with pytest.raises(
        ValueError, match=r"total_wallet_exposure_limit.*not overridable"
    ):
        _parse(
            {"BTC": {"bot": {"long": {"risk": {"total_wallet_exposure_limit": 1.0}}}}}
        )


def test_strategy_kind_mismatch_fails_in_inline_patch():
    with pytest.raises(
        ValueError, match=r"ema_anchor cannot override active strategy_kind"
    ):
        _parse(
            {
                "BTC": {
                    "bot": {
                        "long": {
                            "strategy": {"ema_anchor": {"entry": {"ema_dist": -0.01}}}
                        }
                    }
                }
            }
        )


def test_inactive_strategy_subtrees_in_full_override_file_are_ignored():
    loaded = get_template_config()
    loaded["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_pct"
    ] = 0.123
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        loaded=loaded,
    )

    strategy = parsed["coin_overrides"]["BTC"]["bot"]["long"]["strategy"]
    assert set(strategy) == {"trailing_martingale"}
    assert strategy["trailing_martingale"]["entry"]["threshold_base_pct"] == 0.123


def test_override_file_strategy_kind_mismatch_fails():
    loaded = get_template_config()
    loaded["live"]["strategy_kind"] = "ema_anchor"
    with pytest.raises(ValueError, match=r"live\.strategy_kind.*does not match"):
        _parse(
            {"BTC": {"override_config_path": "unused-by-test.json"}},
            loaded=loaded,
        )

    with pytest.raises(
        ValueError, match=r"ema_anchor cannot override active strategy_kind"
    ):
        _parse(
            {"BTC": {"override_config_path": "unused-by-test.json"}},
            loaded={
                "bot": {
                    "long": {"strategy": {"ema_anchor": {"entry": {"ema_dist": -0.01}}}}
                }
            },
        )


@pytest.mark.parametrize("value", [0, -1.0])
def test_non_positive_leverage_override_fails(value):
    with pytest.raises(ValueError, match=r"live\.leverage .*must be > 0\.0"):
        _parse({"BTC": {"live": {"leverage": value}}})


def test_normalized_coin_collision_fails():
    with pytest.raises(ValueError, match=r"both normalize to 'BTC'"):
        _parse(
            {"BTC": {}, "XBT": {}},
            symbol_normalizer=lambda coin: "BTC",
        )


def test_invalid_normalized_coin_fails():
    with pytest.raises(ValueError, match="not a valid coin or symbol"):
        _parse({"???": {}}, symbol_normalizer=lambda coin: "")


def test_wallet_exposure_limit_from_partial_file_is_retained_and_validated():
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        loaded={"bot": {"short": {"wallet_exposure_limit": 0.125}}},
    )
    assert (
        parsed["coin_overrides"]["BTC"]["bot"]["short"]["wallet_exposure_limit"]
        == 0.125
    )

    with pytest.raises(
        ValueError, match=r"wallet_exposure_limit .*must be finite and >= 0\.0"
    ):
        _parse(
            {"BTC": {"override_config_path": "unused-by-test.json"}},
            loaded={"bot": {"short": {"wallet_exposure_limit": -0.1}}},
        )


def test_parse_overrides_is_idempotent():
    parsed_once = _parse(
        {"BTC": {"bot": {"long": {"unstuck": {"loss_allowance_pct": 0.02}}}}}
    )
    parsed_twice = parse_overrides(
        parsed_once,
        verbose=False,
        override_loader=lambda config, coin: {},
        symbol_normalizer=lambda coin: coin,
    )
    assert parsed_twice["coin_overrides"] == parsed_once["coin_overrides"]


def test_programmatic_override_added_after_prepare_is_retained():
    prepared = _prepared_config({})
    assert prepared["_raw"]["coin_overrides"] == {}
    prepared["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"loss_allowance_pct": 0.027}}}}
    }

    parsed = parse_overrides(
        prepared,
        verbose=False,
        override_loader=lambda config, coin: {},
        symbol_normalizer=lambda coin: coin,
    )

    assert (
        parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["loss_allowance_pct"]
        == 0.027
    )


def test_canonical_grouped_patch_has_live_and_backtest_consumer_parity():
    parsed = _parse(
        {"BTC": {"bot": {"long": {"unstuck": {"loss_allowance_pct": 0.031}}}}}
    )
    override_side = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    merged = merge_runtime_bot_side(
        parsed["bot"]["long"],
        pside="long",
        override_side=override_side,
        strategy_kind=parsed["live"]["strategy_kind"],
    )

    bot = Passivbot.__new__(Passivbot)
    bot.config = parsed
    bot.coin_overrides = {"BTC/USDT:USDT": parsed["coin_overrides"]["BTC"]}

    assert merged["unstuck_loss_allowance_pct"] == 0.031
    assert (
        bot.config_get(
            ["bot", "long", "unstuck_loss_allowance_pct"],
            "BTC/USDT:USDT",
        )
        == 0.031
    )


@pytest.mark.parametrize("pside", ["long", "short"])
def test_complete_current_active_strategy_and_shared_allowlist_is_accepted(pside):
    template = get_template_config()
    side = template["bot"][pside]
    parsed = _parse(
        {
            "BTC": {
                "bot": {
                    pside: {
                        "strategy": {
                            "trailing_martingale": deepcopy(
                                side["strategy"]["trailing_martingale"]
                            )
                        },
                        "risk": {
                            "entry_cooldown_minutes": side["risk"][
                                "entry_cooldown_minutes"
                            ],
                            "position_exposure_enforcer_enabled": side["risk"][
                                "position_exposure_enforcer_enabled"
                            ],
                            "position_exposure_enforcer_threshold": side["risk"][
                                "position_exposure_enforcer_threshold"
                            ],
                            "we_excess_allowance_pct": side["risk"][
                                "we_excess_allowance_pct"
                            ],
                        },
                        "unstuck": {
                            "close_pct": side["unstuck"]["close_pct"],
                            "ema_dist": side["unstuck"]["ema_dist"],
                            "ema_gating_enabled": side["unstuck"]["ema_gating_enabled"],
                            "enabled": side["unstuck"]["enabled"],
                            "loss_allowance_pct": side["unstuck"]["loss_allowance_pct"],
                            "threshold": side["unstuck"]["threshold"],
                        },
                        "wallet_exposure_limit": 0.1,
                    }
                },
                "live": {
                    "forced_mode_long": "",
                    "forced_mode_short": "normal",
                    "leverage": template["live"]["leverage"],
                },
            }
        }
    )

    assert parsed["coin_overrides"]["BTC"]["bot"][pside]["wallet_exposure_limit"] == 0.1
