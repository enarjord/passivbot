from copy import deepcopy

import pytest

from config import (
    compile_runtime_config,
    get_template_config,
    load_prepared_config,
    prepare_config,
    project_config,
)


@pytest.mark.parametrize(
    ("target", "expected_sections"),
    [
        ("canonical", {"backtest", "bot", "coin_overrides", "live", "logging", "monitor", "optimize"}),
        ("live", {"bot", "coin_overrides", "live", "logging", "monitor"}),
        ("backtest", {"backtest", "bot", "coin_overrides", "live", "logging"}),
        ("optimize", {"backtest", "bot", "coin_overrides", "live", "logging", "optimize"}),
        ("monitor", {"live", "logging", "monitor"}),
    ],
)
def test_project_config_keeps_only_target_sections_and_metadata(target, expected_sections):
    cfg = get_template_config()
    cfg["_raw"] = {"live": {"user": "raw_user"}}
    cfg["_raw_effective"] = {"live": {"user": "effective_user"}}
    cfg["_transform_log"] = [{"step": "seed"}]
    cfg["_coins_sources"] = {"approved_coins": "configs/approved_coins.json"}

    projected = project_config(cfg, target)

    metadata_keys = {"_raw", "_raw_effective", "_transform_log", "_coins_sources"}
    assert set(projected) == expected_sections | metadata_keys
    for section in ("backtest", "bot", "coin_overrides", "live", "logging", "monitor", "optimize"):
        if section in expected_sections:
            assert section in projected
        else:
            assert section not in projected


def test_prepare_config_canonical_omits_runtime_aliases():
    prepared = prepare_config(
        get_template_config(),
        verbose=False,
        target="canonical",
        runtime=None,
    )

    assert "filter_volume_ema_span" not in prepared["bot"]["long"]
    assert "filter_volatility_ema_span" not in prepared["bot"]["long"]
    assert "long_filter_volume_ema_span" not in prepared["optimize"]["bounds"]
    assert "long_filter_volatility_ema_span" not in prepared["optimize"]["bounds"]


def test_compile_runtime_config_adds_runtime_aliases_without_removing_canonical_keys():
    canonical = prepare_config(
        get_template_config(),
        verbose=False,
        target="optimize",
        runtime=None,
    )

    compiled = compile_runtime_config(canonical, runtime="optimize")

    assert compiled["bot"]["long"]["forager_volume_ema_span"] == canonical["bot"]["long"]["forager_volume_ema_span"]
    assert compiled["bot"]["long"]["filter_volume_ema_span"] == canonical["bot"]["long"]["forager_volume_ema_span"]
    assert compiled["bot"]["long"]["filter_volatility_ema_span"] == canonical["bot"]["long"]["forager_volatility_ema_span"]
    assert compiled["bot"]["long"]["filter_volatility_drop_pct"] == pytest.approx(0.0)
    assert (
        compiled["optimize"]["bounds"]["long_filter_volume_ema_span"]
        == canonical["optimize"]["bounds"]["long_forager_volume_ema_span"]
    )
    assert (
        compiled["optimize"]["bounds"]["long_filter_volatility_ema_span"]
        == canonical["optimize"]["bounds"]["long_forager_volatility_ema_span"]
    )


def test_prepare_config_target_and_runtime_preserve_raw_metadata_and_record_steps():
    source = get_template_config()
    source["live"]["user"] = "override_user"
    raw_snapshot = deepcopy(get_template_config())

    prepared = prepare_config(
        source,
        verbose=False,
        target="live",
        runtime="live",
        raw_snapshot=raw_snapshot,
        effective_snapshot=deepcopy(source),
    )

    assert prepared["_raw"]["live"]["user"] == raw_snapshot["live"]["user"]
    assert prepared["_raw_effective"]["live"]["user"] == "override_user"
    steps = [entry["step"] for entry in prepared["_transform_log"]]
    assert "normalize_config" in steps
    assert "project_config" in steps
    assert "compile_runtime_config" in steps
    assert "backtest" not in prepared
    assert "optimize" not in prepared
    assert "monitor" in prepared


def test_load_prepared_config_without_path_uses_schema_defaults_pipeline():
    prepared = load_prepared_config(
        None,
        verbose=False,
        target="backtest",
        runtime="backtest",
        log_info=False,
    )

    template = get_template_config()
    assert prepared["backtest"]["market_order_slippage_pct"] == template["backtest"]["market_order_slippage_pct"]
    assert prepared["bot"]["long"]["filter_volume_ema_span"] == template["bot"]["long"]["forager_volume_ema_span"]
    assert prepared["_raw"] == template
    assert prepared["_raw_effective"] == template


def test_prepare_config_migrates_legacy_backtest_market_slippage_key():
    source = {
        "backtest": {"panic_market_slippage_pct": 0.0015},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["market_order_slippage_pct"] == pytest.approx(0.0015)
    assert "panic_market_slippage_pct" not in prepared["backtest"]


def test_prepare_config_removes_empty_means_all_approved_from_canonical_shape():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {
            "approved_coins": [],
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        },
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert "empty_means_all_approved" not in prepared["live"]
    assert prepared["live"]["approved_coins"] == {"long": ["all"], "short": ["all"]}
    assert prepared["_coins_sources"]["approved_coins"] == "all"
    assert prepared["_raw"]["live"]["empty_means_all_approved"] is True
    assert prepared["_raw_effective"]["live"]["empty_means_all_approved"] is True
