"""Hourly input admission must precede shader compilation or replay allocation."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from optimization.gpu import model, mps_kernel


class CompilationReached(RuntimeError):
    pass


def runner_and_params(monkeypatch, kind, sides=1, *, first_observed=10):
    keys = getattr(model, kind + "_MULTICOIN_PARAM_KEYS")
    columns = getattr(model, kind + "_COIN_OVERRIDE_COLS")
    strategy = (model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS if kind == "EMA_ANCHOR" else
                tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS))
    hour = "offset_volatility_ema_span_1h" if kind == "EMA_ANCHOR" else "volatility_ema_span_1h"
    weight = "offset_volatility_1h_weight" if kind == "EMA_ANCHOR" else "entry_threshold_volatility_1h_weight"
    matrix = np.zeros((2, sides * len(keys)), dtype=np.float64)
    for side in range(sides):
        for key, value in ((hour, 3.0), (weight, 1.0), ("n_positions", 1),
                           ("total_wallet_exposure_limit", 1.0),
                           ("unstuck_memory_ema_span_minutes", 3.0),
                           ("loss_gate_memory_ema_span_minutes", 7.0)):
            matrix[:, side * len(keys) + keys.index(key)] = value
    overrides = tuple(np.full((1, columns), np.nan) for _ in range(sides))
    runner = SimpleNamespace(
        parameter_keys=keys, _bootstrap_coin_overrides=overrides, interval_minutes=60,
        n=20, replay_sides=sides, n_coins=1, requested_start_idx=0,
        run_config=SimpleNamespace(warmup_bars=1),
        _hourly_first_observed=np.array([first_observed]),
        _hourly_coin_ranges=np.array([[0, 19, 2]]),
        _pack_params=lambda params: params,
        _use_disabled_hsl_specialization=lambda params: False,
        _unstuck_ema_required=lambda params: False,
        _unstuck_required=lambda params: False,
    )
    def compile_gate():
        raise CompilationReached("pre-dispatch boundary reached")
    runner._library_cache_call = compile_gate
    monkeypatch.setattr(mps_kernel, "pack_bootstrap_counts", lambda *args, **kwargs: None)
    return runner, matrix, keys, strategy, hour, weight


@pytest.mark.parametrize("kind", ["EMA_ANCHOR", "TRAILING_MARTINGALE"])
@pytest.mark.parametrize("sides", [1, 2])
def test_missing_hourly_basis_rejects_before_compilation(monkeypatch, kind, sides):
    runner, matrix, *_ = runner_and_params(monkeypatch, kind, sides)
    with pytest.raises(ValueError, match="unavailable hourly EMA.*candidate.*coin"):
        mps_kernel._MulticoinReplayRunner._run_factual_attempt(runner, matrix)


@pytest.mark.parametrize("kind", ["EMA_ANCHOR", "TRAILING_MARTINGALE"])
@pytest.mark.parametrize("case", ["ready", "zero_weight", "zero_span", "disabled_side", "disabled_coin", "before_planning", "late_listing"])
def test_hourly_admission_preserves_valid_and_nonconsuming_policies(monkeypatch, kind, case):
    runner, matrix, keys, strategy, hour, weight = runner_and_params(monkeypatch, kind)
    end_steps = None
    if case == "ready":
        runner._hourly_first_observed[:] = 2
    elif case == "zero_weight":
        matrix[:, keys.index(weight)] = 0.0
    elif case == "zero_span":
        matrix[:, keys.index(hour)] = 0.0
    elif case == "disabled_side":
        matrix[:, keys.index("total_wallet_exposure_limit")] = 0.0
    elif case == "disabled_coin":
        column = getattr(model, kind + "_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN")
        runner._bootstrap_coin_overrides[0][:, column] = 0.0
    elif case == "before_planning":
        end_steps = np.array([2, 2])
    elif case == "late_listing":
        runner._hourly_coin_ranges[:] = [0, 19, 10]
    with pytest.raises(CompilationReached):
        mps_kernel._MulticoinReplayRunner._run_factual_attempt(runner, matrix, end_steps=end_steps)


@pytest.mark.parametrize("kind", ["EMA_ANCHOR", "TRAILING_MARTINGALE"])
def test_original_candidate_and_coin_pins_control_hourly_requirement(monkeypatch, kind):
    runner, matrix, keys, strategy, hour, weight = runner_and_params(monkeypatch, kind)
    matrix[:, keys.index(weight)] = 0.0
    runner._bootstrap_coin_overrides[0][:, strategy.index(weight)] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="unavailable hourly EMA"):
        mps_kernel._MulticoinReplayRunner._run_factual_attempt(runner, matrix)
    runner._bootstrap_coin_overrides[0][:, strategy.index(hour)] = 0.0
    with pytest.raises(CompilationReached):
        mps_kernel._MulticoinReplayRunner._run_factual_attempt(runner, matrix)


@pytest.mark.parametrize("kind", ["EMA_ANCHOR", "TRAILING_MARTINGALE"])
@pytest.mark.parametrize("end", [None, 3, 20, 2.5])
def test_hourly_admission_uses_the_executed_exclusive_endpoint(monkeypatch, kind, end):
    runner, matrix, *_ = runner_and_params(monkeypatch, kind, first_observed=3)
    runner.n = 3
    runner._hourly_coin_ranges[:] = [0, 2, 2]
    ends = None if end is None else np.full(2, end)
    # The final candle is excluded, so planning at index2 never executes.
    with pytest.raises(CompilationReached):
        mps_kernel._MulticoinReplayRunner._run_factual_attempt(runner, matrix, end_steps=ends)
