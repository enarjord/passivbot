"""Validate inherited and pinned score margins before runtime consumption."""

from copy import deepcopy
import re

import pytest

from config import prepare_config
from config.bot_inheritance import bot_policy_source_view
from config.forager_hysteresis import forager_score_hysteresis
from config.validate import validate_config


KINDS = ("trailing_martingale", "ema_anchor")
INVALID = [
    pytest.param(-0.1, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param("0.1", id="string"),
    pytest.param(None, id="null"),
]


def source(kind, margins):
    return {
        "config_version": "v9.0.0",
        "bot": {
            "portfolio": {"strategy_kind": kind},
            **{scope: {"forager": {"score_hysteresis_margin": value}}
               for scope, value in margins.items()},
        },
    }


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("scope", ("global", "long", "short"))
@pytest.mark.parametrize("value", INVALID)
def test_preparation_rejects_invalid_inherited_or_directional_margin(kind, scope, value):
    config = source(kind, {scope: value})
    with pytest.raises((TypeError, ValueError), match=re.escape(
        f"bot.{scope}.forager.score_hysteresis_margin"
    )):
        prepare_config(config, live_only=True, verbose=False)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("margins, expected", [
    ({"global": 0.125}, {"long": 0.125, "short": 0.125}),
    ({"global": 0.125, "long": 0}, {"long": 0.0, "short": 0.125}),
    ({"global": 0.125, "short": 0.25}, {"long": 0.125, "short": 0.25}),
    ({"global": 0.125, "long": 0.125}, {"long": 0.125, "short": 0.125}),
])
def test_valid_inheritance_and_zero_or_equal_pins_survive_preparation(kind, margins, expected):
    raw = source(kind, margins)
    before = deepcopy(raw)
    prepared = prepare_config(raw, live_only=True, verbose=False)
    assert forager_score_hysteresis(prepared) == expected
    assert forager_score_hysteresis(prepare_config(prepared, live_only=True, verbose=False)) == expected
    authored = bot_policy_source_view(prepared)["bot"]
    for side in ("long", "short"):
        assert prepared["bot"][side]["enabled"] is False
        if side in margins:
            assert authored[side]["forager"]["score_hysteresis_margin"] == margins[side]
        else:
            assert "forager" not in authored.get(side, {})
    assert raw == before


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_global_margin_is_rejected_even_when_both_sides_pin_valid_values(kind):
    raw = source(kind, {"global": -1.0, "long": 0.0, "short": 0.25})
    with pytest.raises(ValueError, match=r"bot\.global\.forager\.score_hysteresis_margin"):
        prepare_config(raw, live_only=True, verbose=False)


@pytest.mark.parametrize("side", ("long", "short"))
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "0.1"])
def test_canonical_validation_checks_each_resolved_side_without_coercion(side, value):
    config = prepare_config(source("trailing_martingale", {"global": 0.125}),
                            live_only=True, verbose=False)
    config["bot"][side]["forager"]["score_hysteresis_margin"] = value
    with pytest.raises((TypeError, ValueError), match=re.escape(
        f"bot.{side}.forager.score_hysteresis_margin"
    )):
        validate_config(config, verbose=False)
