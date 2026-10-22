"""The new common profile changes fallback ownership, not the released v8 snapshot."""

from copy import deepcopy

import pytest

from config.bot_defaults import get_common_bot_defaults
from config.migrations.released_v8 import get_released_v8_defaults
from config.schema import get_expanded_config_prototype
from config.strategy_spec import (
    get_common_strategy_defaults,
    get_strategy_defaults,
    get_strategy_spec,
)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_native_metadata_declares_one_common_profile_with_directional_ids(kind):
    spec = get_strategy_spec(kind)
    before = deepcopy(spec)
    common = get_common_strategy_defaults()[kind]
    directional = get_strategy_defaults(kind)
    assert common == directional["long"] == directional["short"]
    parameters = {parameter["optimize_key"]: parameter for parameter in spec["parameters"]}
    for key, parameter in parameters.items():
        if key.startswith("long_"):
            other = parameters["short_" + key[len("long_"):]]
            assert parameter["default"] == other["default"]
    common.clear()
    assert spec == before == get_strategy_spec(kind)


def test_common_shared_profile_retains_former_long_values_and_separate_permission():
    common = get_common_bot_defaults()
    prototype = get_expanded_config_prototype()["bot"]
    assert prototype["long"] == prototype["short"] == prototype["global"] == common
    assert common["enabled"] is False
    assert common["universe"]["approved_coins"] == []
    assert common["risk"]["total_wallet_exposure_limit"] == 1.5
    assert common["risk"]["max_positions"] == 7
    assert common["forager"]["score_hysteresis_margin"] == 0.02
    common["risk"]["total_wallet_exposure_limit"] = 0.0
    assert get_common_bot_defaults()["risk"]["total_wallet_exposure_limit"] == 1.5


def test_frozen_v8_defaults_keep_their_original_asymmetric_policy_and_ranges():
    v8 = get_released_v8_defaults(strategy_kind="trailing_martingale")
    assert v8["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] == 790.0
    assert v8["bot"]["short"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] == 60.0
    assert v8["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert v8["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert v8["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"] != (
        v8["optimize"]["bounds"]["short"]["strategy"]["trailing_martingale"]
    )
