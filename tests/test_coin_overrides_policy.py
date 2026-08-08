from copy import deepcopy

import pytest

from config.load import prepare_config
from config.overrides import (
    OVERRIDABLE_SHARED_BOT_PATHS,
    allowed_flat_bot_side_modification_keys,
    parse_overrides,
)
from config.schema import get_template_config


def _parse(overrides, *, loaded=None):
    source = get_template_config()
    source["live"]["user"] = "tester"
    source["coin_overrides"] = deepcopy(overrides)
    prepared = prepare_config(source, verbose=False, log_config_transforms=False)
    return parse_overrides(
        prepared,
        verbose=False,
        override_loader=lambda config, coin: deepcopy(loaded or {}),
        symbol_normalizer=lambda coin: coin,
    )


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize(
    ("group", "key", "value"),
    [
        ("risk", "entry_cooldown_minutes", 2.5),
        ("unstuck", "ema_gating_enabled", False),
        ("unstuck", "loss_allowance_pct", 0.027),
    ],
)
def test_added_and_retained_canonical_policy_paths_are_allowed(
    pside, group, key, value
):
    parsed = _parse({"BTC": {"bot": {pside: {group: {key: value}}}}})

    assert parsed["coin_overrides"]["BTC"]["bot"][pside][group][key] == value


def test_policy_registry_and_legacy_aliases_are_derived_consistently():
    assert "risk.entry_cooldown_minutes" in OVERRIDABLE_SHARED_BOT_PATHS
    assert "unstuck.ema_gating_enabled" in OVERRIDABLE_SHARED_BOT_PATHS
    assert "unstuck.loss_allowance_pct" in OVERRIDABLE_SHARED_BOT_PATHS
    assert "risk.we_excess_allowance_mode" not in OVERRIDABLE_SHARED_BOT_PATHS

    flat = allowed_flat_bot_side_modification_keys()
    assert "risk_entry_cooldown_minutes" in flat
    assert "unstuck_ema_gating_enabled" in flat
    assert "unstuck_loss_allowance_pct" in flat
    assert "risk_we_excess_allowance_mode" not in flat


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize("spelling", ["grouped", "flat"])
def test_removed_allowance_mode_has_actionable_inline_error(pside, spelling):
    side = (
        {"risk": {"we_excess_allowance_mode": "legacy_raw"}}
        if spelling == "grouped"
        else {"risk_we_excess_allowance_mode": "legacy_raw"}
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"coin_overrides\.BTC\.bot\.{pside}\.risk\.we_excess_allowance_mode "
            rf"is no longer overridable.*configure bot\.{pside}\.risk\."
            r"we_excess_allowance_mode globally"
        ),
    ):
        _parse({"BTC": {"bot": {pside: side}}})


def test_removed_allowance_mode_in_full_file_is_warned_and_ignored(caplog):
    parsed = _parse(
        {"BTC": {"override_config_path": "unused-by-test.json"}},
        loaded={
            "bot": {
                "long": {
                    "risk": {
                        "entry_cooldown_minutes": 1.5,
                        "we_excess_allowance_mode": "legacy_raw",
                    }
                }
            }
        },
    )

    risk = parsed["coin_overrides"]["BTC"]["bot"]["long"]["risk"]
    assert risk == {"entry_cooldown_minutes": 1.5}
    assert "is no longer overridable" in caplog.text
    assert "the file value is ignored" in caplog.text


def test_file_then_inline_precedence_and_long_short_independence():
    parsed = _parse(
        {
            "BTC": {
                "override_config_path": "unused-by-test.json",
                "bot": {
                    "long": {"risk_entry_cooldown_minutes": 0.05},
                    "short": {
                        "unstuck": {
                            "ema_gating_enabled": True,
                            "loss_allowance_pct": 0.031,
                        }
                    },
                },
            }
        },
        loaded={
            "bot": {
                "long": {
                    "risk": {"entry_cooldown_minutes": 3.0},
                    "unstuck": {"ema_gating_enabled": False},
                },
                "short": {
                    "risk": {"entry_cooldown_minutes": 7.0},
                    "unstuck": {
                        "ema_gating_enabled": False,
                        "loss_allowance_pct": 0.029,
                    },
                },
            }
        },
    )

    bot = parsed["coin_overrides"]["BTC"]["bot"]
    assert bot["long"]["risk"]["entry_cooldown_minutes"] == 0.05
    assert bot["long"]["unstuck"]["ema_gating_enabled"] is False
    assert bot["short"]["risk"]["entry_cooldown_minutes"] == 7.0
    assert bot["short"]["unstuck"]["ema_gating_enabled"] is True
    assert bot["short"]["unstuck"]["loss_allowance_pct"] == 0.031


def test_new_policy_fields_use_effective_config_and_type_validation():
    with pytest.raises(
        ValueError,
        match=r"coin_overrides\.BTC produces an invalid config.*entry_cooldown_minutes",
    ):
        _parse({"BTC": {"bot": {"long": {"risk": {"entry_cooldown_minutes": -0.1}}}}})

    with pytest.raises(TypeError, match=r"ema_gating_enabled.*numeric or boolean"):
        _parse(
            {"BTC": {"bot": {"short": {"unstuck": {"ema_gating_enabled": "false"}}}}}
        )
