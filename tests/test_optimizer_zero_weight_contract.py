"""Raw fixed genes and canonical effective ranking share native replay behavior."""

from copy import deepcopy
import json

import numpy as np
import pytest

from backtest import build_backtest_payload, execute_backtest
from config import prepare_config
from config.bot import normalize_forager_score_weights
from config.result_records import extract_result_config
from config_utils import clean_config, strip_config_metadata
from optimization.callback import build_pymoo_record_entry
from optimization.shape import build_optimization_shape
from optimization.warmup import build_optimizer_vector_config
from optimizer_overrides import optimizer_overrides
from tools.gpu_parity import build_parser, fixture_inputs


ZERO_WEIGHTS = dict(volume=0.0, volatility=0.0, ema_readiness=0.0, unilateralness=0.0)
FALLBACK_WEIGHTS = {**ZERO_WEIGHTS, "ema_readiness": 1.0}


def _assert_native_inputs_equal(left, right):
    """Only relative score-weight transport may differ before Rust canonicalizes it."""
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key == "forager_score_weights":
                assert normalize_forager_score_weights(left[key], path=key) == normalize_forager_score_weights(right[key], path=key)
            else:
                _assert_native_inputs_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_native_inputs_equal(a, b)
    else:
        assert left == right


def _candidate(strategy, policy, mirrored):
    args = build_parser().parse_args([
        "--fixture", strategy, "--hsl", "disabled", "--sides", "both",
        "--coins", "3", "--bars", "512", "--seed", "2026",
    ])
    source, candles, markets, btc, timestamps = fixture_inputs(args)
    coins = source["backtest"].pop("coins")
    for side in ("long", "short"):
        source["bot"][side]["risk"]["max_positions"] = 1
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"] if mirrored else []
    target = "short" if policy == "activation" and not mirrored else "long"
    if policy == "runtime":
        source["optimize"]["bounds"] = {target: {"forager": {"score_weights": {
            name: [0.0, 1.0] for name in ZERO_WEIGHTS
        }}}}
        source["optimize"]["fixed_runtime_overrides"] = {
            f"bot.{target}.forager.score_weights.{name}": 0.0 for name in ZERO_WEIGHTS
        }
    else:
        for side in ("long", "short"):
            if policy == "anchor" or mirrored or side == target:
                source["bot"][side]["risk"]["total_wallet_exposure_limit"] = 0.0
                source["bot"][side]["forager"]["score_weights"] = deepcopy(ZERO_WEIGHTS)
        source["optimize"]["bounds"] = {target: {"risk": {
            "max_positions": [1], "total_wallet_exposure_limit": [0.0, 1.0],
        }}}
        if policy == "anchor":
            source["optimize"]["bounds"][target]["forager"] = {"score_weights": {
                name: [0.0, 1.0] for name in ZERO_WEIGHTS
            }}
    source = strip_config_metadata(source)
    source["live"].pop("base_config_path", None)
    config = prepare_config(source, verbose=False)
    if policy == "anchor":
        from optimize import install_anchored_fine_tune_plan

        install_anchored_fine_tune_plan(
            config, [f"{target}.risk.total_wallet_exposure_limit"], "synthetic",
            starting_configs_override=[source],
        )
    shape = build_optimization_shape(config)
    vector = [bound.high for bound in shape.bounds]
    candidate = build_optimizer_vector_config(vector, config, key_paths=shape.key_paths)
    candidate["backtest"]["coins"] = deepcopy(coins)
    return config, candidate, vector, target, coins, candles, markets, btc, timestamps


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("policy", ["activation", "runtime", "anchor"])
@pytest.mark.parametrize("mirrored", [False, True])
def test_fixed_zero_genes_keep_native_behavior_through_actual_writer_and_reload(strategy, policy, mirrored, monkeypatch):
    import socket
    import passivbot_rust

    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("requires the native backtest extension")

    def denied(*_args, **_kwargs):
        pytest.fail("synthetic config replay must not use a network")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    config, candidate, vector, target, coins, candles, markets, btc, timestamps = _candidate(strategy, policy, mirrored)
    before = deepcopy(candidate)
    assert candidate["bot"][target]["risk"]["total_wallet_exposure_limit"] == 1.0
    assert candidate["bot"][target]["forager"]["score_weights"] == ZERO_WEIGHTS
    entry = build_pymoo_record_entry(
        vector=vector, metrics={}, template=config, build_config_fn=None,
        overrides_fn=optimizer_overrides, overrides_list=config["optimize"]["enable_overrides"],
        prepared_config=candidate,
    )
    # Cleaning deliberately keeps authored/raw bot values. Loading the saved
    # result displays the existing effective EMA-readiness normalization instead.
    assert entry["bot"][target]["forager"]["score_weights"] == ZERO_WEIGHTS
    assert clean_config(strip_config_metadata(candidate))["bot"][target]["forager"]["score_weights"] == ZERO_WEIGHTS
    loaded = prepare_config(extract_result_config(json.loads(json.dumps(entry))), verbose=False)
    assert loaded["bot"][target]["forager"]["score_weights"] == FALLBACK_WEIGHTS
    loaded["backtest"]["coins"] = deepcopy(coins)
    assert loaded["optimize"] == entry["optimize"]
    raw_payload = build_backtest_payload(candles, deepcopy(markets), candidate, "binance", btc, timestamps)
    loaded_payload = build_backtest_payload(candles, deepcopy(markets), loaded, "binance", btc, timestamps)
    for field in ("bot_params_list", "strategy_params_list", "exchange_params", "backtest_params"):
        _assert_native_inputs_equal(getattr(raw_payload, field), getattr(loaded_payload, field))
    raw_result = execute_backtest(raw_payload, candidate)
    loaded_result = execute_backtest(loaded_payload, loaded)
    assert len(raw_result[0]) > 0
    np.testing.assert_array_equal(raw_result[0], loaded_result[0])
    np.testing.assert_array_equal(raw_result[1], loaded_result[1])
    # Native parallel metric reductions may differ by a rounding bit.
    assert raw_result[2] == pytest.approx(loaded_result[2], rel=1e-14, abs=1e-14, nan_ok=True)
    assert candidate == before
