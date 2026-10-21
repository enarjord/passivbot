from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from config.load import prepare_config
from config.overrides import parse_overrides
from config.runtime_compile import compile_runtime_config
from config.schema import get_template_config
from config.strategy import merge_runtime_bot_side
from passivbot import Passivbot


def _prepared_config(overrides):
    source = get_template_config()
    source["live"]["account"]["user"] = "tester"
    source["coin_overrides"] = deepcopy(overrides)
    return prepare_config(source, verbose=False, log_config_transforms=False)


def _parse(overrides, *, loaded=None, symbol_normalizer=lambda coin: coin):
    with TemporaryDirectory() as directory:
        overrides = deepcopy(overrides)
        for patch in overrides.values():
            if "override_config_path" in patch:
                path = Path(directory) / "override.json"
                path.write_text(json.dumps(loaded or {}))
                patch["override_config_path"] = str(path)
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
        "threshold_base_fraction"
    ]
    enabled = template["bot"]["long"]["unstuck"]["enabled"]
    parsed = _parse(
        {
            "BTC": {
                "bot": {
                    "long": {
                        "strategy": {
                            "trailing_martingale": {
                                "entry": {"threshold_base_fraction": threshold}
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
        override["strategy"]["trailing_martingale"]["entry"]["threshold_base_fraction"]
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
                        "loss_allowance_fraction": 0.02,
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
                    "loss_allowance_fraction": 0.02,
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
                                    "entry": {"threshold_base_fraction": value}
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
                                "position_exposure_enforcer": {
                                    "enabled": True,
                                    'threshold_ratio': 0.0,
                                },
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
        _parse({"BTC": {"bot": {"long": {"unstuck": {"ema_distance_fraction": -1.0}}}}})


def test_effective_validation_normalization_is_retained_in_patch():
    parsed = _parse({"BTC": {"bot": {"long": {"unstuck": {"exposure_trigger_ratio": 1e-12}}}}})

    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["exposure_trigger_ratio"] == 0.0


def test_runtime_compilation_must_follow_override_parsing():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"unstuck": {"loss_allowance_fraction": 0.25}}}}
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
    parsed = parse_overrides(canonical_config, verbose=False)
    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["loss_allowance_fraction"] == 0.25


def test_retired_volume_filter_fails_before_runtime_compilation():
    source = get_template_config()
    source["coin_overrides"] = {"BTC": {"bot": {"long": {"filter_volume_drop_pct": 0.25}}}}
    original = deepcopy(source)
    with pytest.raises(ValueError, match=r"filter_volume_drop_pct.*retired"):
        prepare_config(source, verbose=False, runtime="live")
    assert source == original


def test_runtime_compilation_after_parsing_preserves_normalized_patch():
    parsed = _parse({"BTC": {"bot": {"long": {"unstuck": {"exposure_trigger_ratio": 1e-12}}}}})

    runtime_config = compile_runtime_config(parsed, runtime="live")

    assert (
        runtime_config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["exposure_trigger_ratio"]
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
                            "strategy": {"ema_anchor": {"entry": {"ema_distance_fraction": -0.01}}}
                        }
                    }
                }
            }
        )


def test_inactive_strategy_subtrees_in_full_override_file_are_ignored():
    loaded = get_template_config()
    loaded["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "threshold_base_fraction"
    ] = 0.123
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        loaded=loaded,
    )

    strategy = parsed["coin_overrides"]["BTC"]["bot"]["long"]["strategy"]
    assert set(strategy) == {"trailing_martingale"}
    assert strategy["trailing_martingale"]["entry"]["threshold_base_fraction"] == 0.123


def test_override_file_strategy_kind_mismatch_fails():
    loaded = get_template_config()
    loaded['bot']['global']['strategy_kind'] = "ema_anchor"
    with pytest.raises(ValueError, match=r"bot\.global\.strategy_kind.*does not match"):
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
                    "long": {"strategy": {"ema_anchor": {"ema_span_0_minutes": 17.5}}}
                }
            },
        )


@pytest.mark.parametrize("value", [0, -1.0])
def test_non_positive_leverage_override_fails(value):
    with pytest.raises(ValueError, match="live\\.account\\.leverage .*must be > 0\\.0"):
        _parse({"BTC": {"live": {"account": {"leverage": value}}}})


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
        {"BTC": {"bot": {"long": {"unstuck": {"loss_allowance_fraction": 0.02}}}}}
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
        "BTC": {"bot": {"long": {"unstuck": {"loss_allowance_fraction": 0.027}}}}
    }

    parsed = parse_overrides(
        prepared,
        verbose=False,
        override_loader=lambda config, coin: {},
        symbol_normalizer=lambda coin: coin,
    )

    assert (
        parsed["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["loss_allowance_fraction"]
        == 0.027
    )


def test_canonical_grouped_patch_has_live_and_backtest_consumer_parity():
    parsed = _parse(
        {"BTC": {"bot": {"long": {"unstuck": {"loss_allowance_fraction": 0.031}}}}}
    )
    override_side = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    merged = merge_runtime_bot_side(
        parsed["bot"]["long"],
        pside="long",
        override_side=override_side,
        strategy_kind=parsed['bot']['global']['strategy_kind'],
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
                        "entry_cooldown": deepcopy(side["entry_cooldown"]),
                        "risk": {
                            "we_excess_allowance_fraction": side["risk"][
                                "we_excess_allowance_fraction"
                            ],
                            "position_exposure_enforcer": {
                                "enabled": side["risk"]["position_exposure_enforcer"][
                                    "enabled"
                                ],
                                'threshold_ratio': side["risk"]["position_exposure_enforcer"][
                                    "threshold_ratio"
                                ],
                            },
                        },
                        "unstuck": {
                            "close_fraction": side["unstuck"]["close_fraction"],
                            "ema_distance_fraction": side["unstuck"]["ema_distance_fraction"],
                            "ema_gating_enabled": side["unstuck"]["ema_gating_enabled"],
                            "enabled": side["unstuck"]["enabled"],
                            "loss_allowance_fraction": side["unstuck"]["loss_allowance_fraction"],
                            "exposure_trigger_ratio": side["unstuck"][
                                "exposure_trigger_ratio"
                            ],
                        },
                        "wallet_exposure_limit": 0.1,
                    }
                },
                "live": {
                    "modes": {"long": None, "short": "normal"},
                    "account": {"leverage": template["live"]["account"]["leverage"]},
                },
            }
        }
    )

    assert parsed["coin_overrides"]["BTC"]["bot"][pside]["wallet_exposure_limit"] == 0.1
