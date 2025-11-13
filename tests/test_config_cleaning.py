from copy import deepcopy

from config_utils import clean_config, get_template_config


def test_clean_config_removes_internal_sections_and_keeps_user_values():
    template = get_template_config()
    config = {
        "bot": {
            "long": {"n_positions": 5, "_note": "keep me?"},
            "short": {"n_positions": 3},
        },
        "coin_overrides": {
            "BTC": {
                "_meta": "do not keep",
                "live": {"forced_mode_long": "manual"},
            }
        },
        "_transform_log": ["noise"],
        "_raw": {"bot": {}},
    }

    cleaned = clean_config(deepcopy(config))

    assert "_transform_log" not in cleaned
    assert "_raw" not in cleaned
    assert cleaned["bot"]["long"]["n_positions"] == 5
    assert cleaned["bot"]["short"]["n_positions"] == 3
    assert cleaned["bot"]["long"]["ema_span_0"] == template["bot"]["long"]["ema_span_0"]
    assert "BTC" in cleaned["coin_overrides"]
    assert "_meta" not in cleaned["coin_overrides"]["BTC"]
    assert (
        cleaned["coin_overrides"]["BTC"]["live"]["forced_mode_long"]
        == config["coin_overrides"]["BTC"]["live"]["forced_mode_long"]
    )


def test_clean_config_fills_missing_values_from_template():
    config = {}
    cleaned = clean_config(config)
    template = get_template_config()
    assert cleaned == template
