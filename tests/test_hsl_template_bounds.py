"""Generated HSL scopes keep one active controller's declared search ranges."""
from copy import deepcopy

import pytest

from config import prepare_config
from config.hsl import generated_template
from config.schema import get_expanded_config_prototype


def source():
    config = get_expanded_config_prototype()
    config["bot"]["global"]["hsl"]["drawdown_ratio_threshold"] = 0.21
    config["optimize"]["bounds"]["global"] = {
        "hsl": {"drawdown_ratio_threshold": [0.1, 0.3]},
        "risk": {"we_excess_allowance_fraction": [0.1, 0.4]},
    }
    return config


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_generated_scope_preserves_defaults_and_only_its_active_hsl_bounds(mode):
    config = source()
    original = deepcopy(config)
    generated = generated_template(config, mode)
    assert config == original
    assert generated["bot"]["global"] == original["bot"]["global"]
    assert generated["optimize"]["bounds"]["global"]["risk"] == original["optimize"]["bounds"]["global"]["risk"]
    for scope in ("global", "long", "short"):
        if mode == "portfolio":
            assert "hsl" not in generated["optimize"]["bounds"][scope]
        else:
            assert generated["optimize"]["bounds"][scope]["hsl"] == original["optimize"]["bounds"][scope]["hsl"]
    if mode == "portfolio":
        assert generated["optimize"]["bounds"]["portfolio"]["hsl"]
    prepare_config(generated, verbose=False)


def test_authored_global_hsl_bounds_still_reject_inactive_portfolio_scope():
    config = generated_template(source(), "portfolio")
    config["optimize"]["bounds"]["global"]["hsl"] = {"drawdown_ratio_threshold": [0.1, 0.3]}
    with pytest.raises(ValueError, match=r"optimize.bounds.global.hsl.*inactive in portfolio"):
        prepare_config(config, verbose=False)
