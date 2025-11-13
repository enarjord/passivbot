import pytest


from config_utils import format_config, get_template_config


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "DOGEUSDT": {"live": {"forced_mode_long": "manual"}},
            "SOLUSDT": {"live": {"forced_mode_short": "manual"}},
        },
        {
            "DOGEUSDT": {
                "live": {"forced_mode_long": "manual"},
                "bot": {"long": {"wallet_exposure_limit": 0.25}},
            },
        },
    ],
)
def test_coin_overrides_not_pruned(overrides):
    cfg = get_template_config()
    cfg["live"]["user"] = "tester"
    cfg["coin_overrides"] = overrides

    formatted = format_config(cfg, live_only=True, verbose=False)

    assert "coin_overrides" in formatted
    for coin_key, payload in overrides.items():
        assert coin_key in formatted["coin_overrides"], f"{coin_key} missing after formatting"
        for section, expected_section in payload.items():
            assert section in formatted["coin_overrides"][coin_key]
            for key, value in expected_section.items():
                assert formatted["coin_overrides"][coin_key][section][key] == value
