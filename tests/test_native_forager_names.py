"""Native forager quantities preserve ranking, score domains and released sources."""

from copy import deepcopy

import pytest

from config import get_template_config, prepare_config
from config.optimize_bounds import flatten_optimize_bounds
from config.shared_bot import flatten_shared_bot_side, set_grouped_bot_value
from released_v8_fixtures import released_v8_policy
from test_orchestrator_json_api import compute, make_input, make_symbol
from tools.migrate_config_v9 import migrate


@pytest.fixture(autouse=True)
def require_real_native_extension():
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(native, "__is_stub__", False)
    verify_loaded_runtime_extension()


@pytest.mark.parametrize("old", [
    "filter_volume_ema_span_1m", "forager_volume_ema_span_1m",
    "filter_volatility_ema_span_1m", "forager_volatility_ema_span_1m",
    "unilateralness_ema_span_1m",
])
@pytest.mark.parametrize("location", ["global", "symbol"])
def test_native_forager_packet_rejects_retired_span_aliases(old, location):
    import passivbot_rust as native

    packet = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    compute(native, packet)
    params = (packet["global"]["global_bot_params"]["long"] if location == "global"
              else packet["symbols"][0]["long"]["bot_params"])
    params[old] = 12.5
    with pytest.raises(ValueError, match=rf"unknown field `{old}`"):
        compute(native, packet)


@pytest.mark.parametrize("location", ["global", "symbol"])
def test_native_forager_weights_reject_retired_volatility_coordinate(location):
    import passivbot_rust as native

    packet = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    params = (packet["global"]["global_bot_params"]["long"] if location == "global"
              else packet["symbols"][0]["long"]["bot_params"])
    params["forager_score_weights"] = dict(
        volume=0.0, ema_readiness=0.0, log_range=1.0, unilateralness=0.0,
    )
    compute(native, packet)
    params["forager_score_weights"]["volatility"] = 1.0
    with pytest.raises(ValueError, match="unknown field `volatility`"):
        compute(native, packet)


@pytest.mark.parametrize("old,new", [
    ("forager_volume_ema_span_1m", "forager_volume_ema_span_minutes"),
    ("filter_volatility_ema_span_1m", "forager_log_range_ema_span_minutes"),
])
@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
def test_cpu_packet_requires_current_forager_duration_names_without_fallback(old, new, kind):
    from backtest import build_backtest_payload
    from tools.gpu_parity import build_parser, fixture_inputs
    import passivbot_rust as native

    args = build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--hsl", "disabled",
        "--coins", "2", "--bars", "65",
    ])
    source, candles, markets, btc, timestamps = fixture_inputs(args)
    payload = build_backtest_payload(candles, markets, source, args.exchange, btc, timestamps)
    bot = deepcopy(payload.bot_params_list)
    bot[0]["long"][old] = bot[0]["long"].pop(new)
    with pytest.raises((KeyError, ValueError), match=new):
        native.run_backtest_bundle(payload.bundle, bot, payload.strategy_params_list,
                                   payload.exchange_params, payload.backtest_params)


def candidates():
    return [dict(index=i, enabled=True, volume_ema=volume, log_range_ema=log_range,
                 unilateralness_score=0.0, bid=100.0, ask=100.0,
                 ema_lower=lower, ema_upper=100.0, entry_initial_ema_distance_fraction=0.0)
            for i, volume, log_range, lower in (
                (0, 10.0, 0.2, 100.0), (1, 100.0, 0.7, 90.0),
            )]


@pytest.mark.parametrize("weights,expected", [
    (dict(volume=1.0, ema_readiness=0.0, log_range=0.0), [1]),
    (dict(volume=0.0, ema_readiness=0.0, log_range=1.0), [1]),
    (dict(volume=0.0, ema_readiness=1.0, log_range=0.0), [0]),
    (dict(volume=0.0, ema_readiness=0.0, log_range=0.0), [0]),
])
def test_native_raw_quantities_preserve_each_ranking_and_zero_weight_fallback(weights, expected):
    import passivbot_rust as native

    assert native.select_forager_candidates_py(candidates(), "long", 1, weights, True) == expected


@pytest.mark.parametrize("old,new", [
    ("volume_score", "volume_ema"), ("volatility_score", "log_range_ema"),
])
def test_native_candidate_requires_raw_quantity_names_without_alias_fallback(old, new):
    import passivbot_rust as native

    values = candidates()
    values[0][old] = values[0].pop(new)
    with pytest.raises((KeyError, ValueError), match=new):
        native.select_forager_candidates_py(
            values, "long", 1, dict(volume=0.0, ema_readiness=0.0, log_range=1.0), True,
        )


def test_matching_public_and_native_forager_names_preserve_independent_weight_values():
    config = prepare_config(get_template_config(), verbose=False)
    side = config["bot"]["long"]
    public = deepcopy(side["forager"]["score_weights"])
    flat = flatten_shared_bot_side(side)
    assert flat["forager_score_weights"]["log_range"] == public["log_range"]
    assert "volatility" not in flat["forager_score_weights"]
    flat["forager_score_weights"]["log_range"] = 0.375
    assert side["forager"]["score_weights"] == public
    set_grouped_bot_value(side, "forager_score_weights", flat["forager_score_weights"])
    assert side["forager"]["score_weights"]["log_range"] == 0.375
    assert "volatility" not in side["forager"]["score_weights"]
    flat["forager_score_weights"]["log_range"] = 0.5
    assert side["forager"]["score_weights"]["log_range"] == 0.375


@pytest.mark.parametrize("old,path,native", [
    ("forager_volume_ema_span_1m", ("volume_ema_span_minutes",), "forager_volume_ema_span_minutes"),
    ("forager_volatility_ema_span_1m", ("log_range_ema_span_minutes",), "forager_log_range_ema_span_minutes"),
    ("forager_score_weights_volatility", ("score_weights", "log_range"), "forager_score_weights_log_range"),
])
def test_released_flat_forager_bounds_preserve_values_and_report_final_paths(old, path, native):
    source = released_v8_policy()
    value = [12.5, 88.75, 0.25] if "span" in old else [0.125, 0.875, 0.125]
    source["optimize"]["bounds"] = {f"long_{old}": value}
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == original
    flat = flatten_optimize_bounds(result.config["optimize"]["bounds"],
                                   strategy_kind="trailing_martingale")
    assert flat[f"long_{native}"] == value
    assert f"long_{old}" not in flat
    target = "optimize.bounds.long.forager." + ".".join(path)
    assert any(isinstance(row["new"], dict) and target in row["new"] for row in result.changes)


def test_released_weight_optimizer_id_does_not_gain_bot_field_authority():
    source = released_v8_policy()
    source["bot"]["long"]["forager_score_weights_volatility"] = 0.25
    with pytest.raises(ValueError, match="forager_score_weights_volatility"):
        migrate(source, memory_span_minutes=123.5)


@pytest.mark.parametrize("selector", [
    "forager_score_weights_volatility", "bot.long.forager_score_weights_volatility",
])
def test_released_unmatched_weight_id_selector_does_not_gain_new_freeze_authority(selector):
    from config.param_paths import resolve_bound_selectors

    source = released_v8_policy()
    source["optimize"]["fixed_params"] = [selector]
    result = migrate(source, memory_span_minutes=123.5)
    assert result.config["optimize"]["fixed_params"] == [selector]
    flat = flatten_optimize_bounds(result.config["optimize"]["bounds"],
                                   strategy_kind="trailing_martingale")
    assert resolve_bound_selectors(result.config, [selector], flat) == {}


def test_normalized_score_margin_has_no_fraction_upper_bound():
    from test_side_forager_hysteresis import selection_snapshot
    import passivbot_rust as native

    packet = selection_snapshot("ema_anchor", high_side="long")
    packet["global"]["forager_score_hysteresis"]["long"] = 1.25
    result = compute(native, packet)
    row = next(row for row in result["diagnostics"]["forager_selections"] if row["pside"] == "long")
    assert row["score_hysteresis_margin"] == 1.25
    assert "score_hysteresis_pct" not in row


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_unilateralness_retains_unit_interval_validation_when_weighted(value):
    import passivbot_rust as native

    values = candidates()
    values[0]["unilateralness_score"] = value
    with pytest.raises(ValueError, match="unilateralness_score"):
        native.select_forager_candidates_py(
            values, "long", 1,
            dict(volume=0.0, ema_readiness=0.0, log_range=0.0, unilateralness=1.0), True,
        )


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("shape", ["old_only", "both", "old_none"])
def test_raw_cpu_rejects_retired_optional_unilateralness_span_even_with_default(kind, shape):
    from backtest import build_backtest_payload
    from tools.gpu_parity import build_parser, fixture_inputs
    import passivbot_rust as native

    args = build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--hsl", "disabled",
        "--coins", "2", "--bars", "65",
    ])
    source, candles, markets, btc, timestamps = fixture_inputs(args)
    payload = build_backtest_payload(candles, markets, source, args.exchange, btc, timestamps)
    bot = deepcopy(payload.bot_params_list)
    params = bot[0]["long"]
    if shape == "old_only":
        params.pop("unilateralness_ema_span_minutes")
    params["unilateralness_ema_span_1m"] = None if shape == "old_none" else 3.75
    with pytest.raises(ValueError, match="unilateralness_ema_span_1m.*retired"):
        native.run_backtest_bundle(payload.bundle, bot, payload.strategy_params_list,
                                   payload.exchange_params, payload.backtest_params)


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
def test_genuinely_omitted_current_unilateralness_duration_keeps_native_default(kind):
    import numpy as np
    import passivbot_rust as native
    from backtest import build_backtest_payload
    from tools.gpu_parity import build_parser, fixture_inputs

    args = build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--hsl", "disabled",
        "--coins", "2", "--bars", "65",
    ])
    source, candles, markets, btc, timestamps = fixture_inputs(args)
    payload = build_backtest_payload(candles, markets, source, args.exchange, btc, timestamps)
    explicit = deepcopy(payload.bot_params_list)
    explicit[0]["long"]["unilateralness_ema_span_minutes"] = 60.0
    omitted = deepcopy(explicit)
    omitted[0]["long"].pop("unilateralness_ema_span_minutes")
    outputs = [native.run_backtest_bundle(payload.bundle, bot, payload.strategy_params_list,
               payload.exchange_params, payload.backtest_params) for bot in (explicit, omitted)]
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("shape", ["grouped", "flat"])
def test_released_unilateralness_duration_and_bounds_use_frozen_source_names(shape):
    source = released_v8_policy()
    if shape == "flat":
        source["bot"]["long"]["forager"].pop("unilateralness_ema_span_1m")
        source["bot"]["long"]["unilateralness_ema_span_1m"] = 42.5
    else:
        source["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 42.5
    source["optimize"]["bounds"]["long_unilateralness_ema_span_1m"] = [20.5, 80.75, 0.25]
    source["optimize"]["fixed_params"] = ["bot.long.unilateralness_ema_span_1m"]
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == original
    assert result.config["bot"]["long"]["forager"]["unilateralness_ema_span_minutes"] == 42.5
    native = flatten_optimize_bounds(result.config["optimize"]["bounds"],
                                     strategy_kind="trailing_martingale")
    assert native["long_unilateralness_ema_span_minutes"] == [20.5, 80.75, 0.25]
    assert "long_unilateralness_ema_span_1m" not in native
    assert result.config["optimize"]["fixed_params"] == [
        "bot.long.forager.unilateralness_ema_span_minutes",
    ]


def test_fractional_unilateralness_span_keeps_derived_window_and_minute_admission():
    import numpy as np
    from optimization.gpu.model import adaptive_params, EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    from optimization.gpu.mps_kernel import _scale_directional_minute_parameters

    derived = adaptive_params({"unilateralness_ema_span_minutes": 3.75})
    assert derived["unilateralness_ema_span_minutes"] == 3.75
    assert derived["unilateralness_window"] == 75
    keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    values = np.ones((1, len(keys)), dtype=np.float64)
    with pytest.raises(ValueError, match="1m|one-minute|1-minute"):
        _scale_directional_minute_parameters(values, keys, sides=1, interval_minutes=2.0)
