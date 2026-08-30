from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time

import numpy as np

from optimization.gpu.model import (
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
    build_mps_data,
    build_mps_multicoin_data,
)

CASES = (
    "ema-single-long",
    "tm-single-long",
    "tm-single-long-entry-ladder",
    "tm-single-long-static-close",
    "tm-single-long-close-ladder",
    "tm-single-long-hsl",
    "ema-multicoin-overhead",
    "ema-multicoin-overrides",
)
SINGLE_COIN_CASES = frozenset(
    {
        "ema-single-long",
        "tm-single-long",
        "tm-single-long-entry-ladder",
        "tm-single-long-static-close",
        "tm-single-long-close-ladder",
        "tm-single-long-hsl",
    }
)
DEFAULT_SEED = 7
MAX_CANDIDATES = 4096
MAX_WARM_RUNS = 20
MAX_SINGLE_BARS = 525_600
MAX_MULTICOIN_BARS = 100_000
MAX_COINS = 64
MAX_DISPATCH_CANDIDATE_BARS = 500_000_000
HSL_PNL_LOOKBACK_BARS = 30 * 24 * 60
HSL_SIGNAL_MODE_COIN = 2.0


def _base_parameter_values() -> dict[str, float]:
    return {
        "base_qty_pct": 0.08,
        "ema_span_0": 60.0,
        "ema_span_1": 240.0,
        "entry_double_down_factor": 1.25,
        "offset": 0.01,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 120.0,
        "offset_volatility_ema_span_1m": 60.0,
        "volatility_ema_span_1h": 120.0,
        "volatility_ema_span_1m": 60.0,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.08,
        "entry_threshold_base_pct": 0.01,
        "entry_threshold_we_weight": 0.0,
        "entry_threshold_volatility_1h_weight": 0.0,
        "entry_threshold_volatility_1m_weight": 0.0,
        "entry_retracement_base_pct": 0.001,
        "entry_retracement_we_weight": 0.0,
        "entry_retracement_volatility_1h_weight": 0.0,
        "entry_retracement_volatility_1m_weight": 0.0,
        "close_qty_pct": 0.5,
        "close_threshold_base_pct": 0.01,
        "close_threshold_we_weight": 0.0,
        "close_threshold_volatility_1h_weight": 0.0,
        "close_threshold_volatility_1m_weight": 0.0,
        "close_retracement_base_pct": 0.001,
        "close_retracement_volatility_1h_weight": 0.0,
        "close_retracement_volatility_1m_weight": 0.0,
        "entry_cooldown_minutes": 0.0,
        "total_wallet_exposure_limit": 1.0,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
        "forager_volume_ema_span_1m": 120.0,
        "forager_volatility_ema_span_1m": 60.0,
        "forager_volume_drop_pct": 0.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 4.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        "unstuck_enabled": 0.0,
        "unstuck_ema_gating_enabled": 1.0,
        "unstuck_close_pct": 0.1,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.1,
        "unstuck_threshold": 0.5,
        "hsl_enabled": 0.0,
        "hsl_red_threshold": 0.2,
        "hsl_ema_span_minutes": 60.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 1.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_mode": 0.0,
        "hsl_slot_count": 1.0,
        "wallet_exposure_limit": -1.0,
    }


def _single_coin_value_overrides(name: str) -> dict[str, float]:
    if name == "tm-single-long-entry-ladder":
        return {
            "entry_retracement_base_pct": 0.0,
        }
    if name in {
        "tm-single-long-static-close",
        "tm-single-long-close-ladder",
    }:
        return {
            "close_qty_pct": 0.25,
            "close_retracement_base_pct": 0.0,
            "close_threshold_we_weight": (
                0.02 if name == "tm-single-long-close-ladder" else 0.0
            ),
        }
    if name != "tm-single-long-hsl":
        return {}
    return {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.02,
        "hsl_ema_span_minutes": 60.0,
        "hsl_cooldown_minutes_after_red": 1_440.0,
        "hsl_restart_policy": 1.0,
        "hsl_signal_mode": HSL_SIGNAL_MODE_COIN,
        "entry_double_down_factor": 2.0,
        "total_wallet_exposure_limit": 5.0,
    }


def _parameter_matrix(
    keys,
    candidates: int,
    seed: int,
    *,
    value_overrides: dict[str, float] | None = None,
) -> np.ndarray:
    values = _base_parameter_values()
    values.update(value_overrides or {})
    base = np.asarray([values[key] for key in keys], dtype=np.float64)
    matrix = np.repeat(base[None, :], candidates, axis=0)
    rng = np.random.default_rng(seed)
    variable_keys = [
        key
        for key in (
            "base_qty_pct",
            "ema_span_0",
            "ema_span_1",
            "offset",
            "entry_initial_ema_dist",
            "entry_initial_qty_pct",
            "entry_threshold_base_pct",
            "close_threshold_base_pct",
        )
        if key in keys
    ]
    for key in variable_keys:
        column = keys.index(key)
        matrix[:, column] *= rng.uniform(0.75, 1.25, size=candidates)
    return np.ascontiguousarray(matrix)


def _synthetic_hlcvs(bars: int, coins: int, seed: int):
    rng = np.random.default_rng(seed)
    steps = np.arange(bars, dtype=np.float64)
    values = np.empty((bars, coins, 4), dtype=np.float64)
    for coin in range(coins):
        phase = coin * 0.73
        trend = steps * (0.0004 + coin * 0.00002)
        wave = np.sin(steps / (37.0 + coin * 3.0) + phase) * (1.5 + coin * 0.1)
        noise = rng.normal(0.0, 0.03, size=bars).cumsum()
        close = 100.0 + coin * 15.0 + trend + wave + noise
        spread = 0.15 + np.abs(np.sin(steps / 19.0 + phase)) * 0.2
        values[:, coin, 0] = close + spread
        values[:, coin, 1] = close - spread
        values[:, coin, 2] = close
        values[:, coin, 3] = 100.0 + coin * 10.0
    timestamps = 1_700_000_000_000 + steps.astype(np.int64) * 60_000
    return values, timestamps


def _fixture_sha256(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _recursive_close_ladder_candidate_count(candidates) -> int:
    return sum(
        float(candidate.get("long_close_retracement_base_pct", 1.0)) <= 0.0
        and float(candidate.get("long_close_threshold_we_weight", 0.0)) != 0.0
        and float(candidate.get("long_close_qty_pct", 1.0)) < 1.0
        for candidate in candidates
    )


def _recursive_entry_ladder_candidate_count(candidates) -> int:
    return sum(
        float(candidate.get("long_entry_retracement_base_pct", 1.0)) <= 0.0
        and float(candidate.get("long_entry_cooldown_minutes", 0.0)) == 0.0
        for candidate in candidates
    )


def _market_and_run(timestamps, bars: int):
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        60,
        60,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        bars - 1,
    )
    return market, run


def _build_case(
    name: str,
    *,
    candidates: int,
    dispatch_batch_size: int,
    single_bars: int,
    multicoin_bars: int,
    coins: int,
    seed: int,
):
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorMulticoinRunner,
        MpsEmaAnchorRunner,
        MpsTrailingMartingaleRunner,
    )
    from optimization.gpu.metrics import compute_objectives
    from optimization.gpu.service import (
        MpsMulticoinProxy,
        MpsSingleCoinProxy,
        mps_requested_metric_features,
    )

    needed_metrics = {
        "adg_strategy_eq",
        "drawdown_worst_strategy_eq",
        "fills_per_day",
    }
    base_values = _base_parameter_values()

    if name in SINGLE_COIN_CASES:
        hlcvs, timestamps = _synthetic_hlcvs(single_bars, 1, seed)
        market, run = _market_and_run(timestamps, single_bars)
        data = build_mps_data(
            hlcvs[:, 0, 0],
            hlcvs[:, 0, 1],
            hlcvs[:, 0, 2],
            timestamps,
            run,
            market,
        )
        hsl_enabled = name == "tm-single-long-hsl"
        hsl_value_overrides = _single_coin_value_overrides(name)
        if name == "ema-single-long":
            runner = MpsEmaAnchorRunner(
                market,
                run,
                data,
                long_enabled=True,
                short_enabled=False,
                hsl_enabled=False,
            )
            side_matrix = _parameter_matrix(
                EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS, candidates, seed
            )
        else:
            requested_metric_features = mps_requested_metric_features(
                needed_metrics,
                strategy_kind="trailing_martingale",
            )
            runner = MpsTrailingMartingaleRunner(
                market,
                run,
                data,
                long_enabled=True,
                short_enabled=False,
                hsl_enabled=hsl_enabled,
                pnl_lookback_bars=(
                    HSL_PNL_LOOKBACK_BARS if hsl_enabled else 0
                ),
                hsl_diagnostics_enabled=(
                    "hsl_diagnostics" in requested_metric_features
                ),
            )
            side_matrix = _parameter_matrix(
                TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
                candidates,
                seed,
                value_overrides=hsl_value_overrides,
            )
        proxy = MpsSingleCoinProxy.__new__(MpsSingleCoinProxy)
        proxy.batch_size = candidates
        proxy.dispatch_batch_size = dispatch_batch_size
        proxy.interrupt_check = lambda: None
        proxy._torch = __import__("torch")
        proxy.profile_enabled = True
        proxy.last_profile = {}
        proxy.metrics_data = data
        proxy.run = run
        proxy.needed_metrics = needed_metrics
        proxy.strategy_kind = (
            "ema_anchor" if name == "ema-single-long" else "trailing_martingale"
        )
        proxy.entry_interval_enabled = False
        proxy.btc_analysis_enabled = False
        proxy.btc_risk_enabled = False
        proxy.equity_balance_diff_enabled = False
        proxy.param_keys = (
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
            if name == "ema-single-long"
            else TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        )
        proxy.base_params = {
            side: {
                key: hsl_value_overrides.get(key, base_values[key])
                for key in proxy.param_keys
            }
            for side in ("long", "short")
        }
        proxy.static_coin_override_params = {}
        proxy.base_total_wallet_exposure_limits = {"long": 1.0, "short": 0.0}
        proxy.base_n_positions = {"long": 1.0, "short": 0.0}
        proxy.runner = runner
        proxy._compute_objectives = compute_objectives
        candidate_dicts = [
            {
                f"long_{key}": float(row[index])
                for index, key in enumerate(proxy.param_keys)
            }
            for row in side_matrix
        ]
        return (
            proxy,
            candidate_dicts,
            single_bars,
            1,
            1,
            _fixture_sha256(hlcvs, timestamps, side_matrix),
        )

    hlcvs, timestamps = _synthetic_hlcvs(multicoin_bars, coins, seed)
    markets_and_runs = [
        _market_and_run(timestamps, multicoin_bars) for _ in range(coins)
    ]
    markets = [item[0] for item in markets_and_runs]
    runs = [item[1] for item in markets_and_runs]
    data = build_mps_multicoin_data(
        hlcvs,
        timestamps,
        runs,
        markets,
        include_hourly_ranges=True,
    )
    overrides = None
    if name == "ema-multicoin-overrides":
        overrides = np.full(
            (coins, EMA_ANCHOR_COIN_OVERRIDE_COLS),
            np.nan,
            dtype=np.float32,
        )
        overrides[0, EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN] = 0.5
        if coins > 1:
            overrides[1, 0] = 0.04
    runner = MpsEmaAnchorMulticoinRunner(
        runs[0],
        data,
        side="long",
        coin_overrides=overrides,
    )
    matrix = _parameter_matrix(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, candidates, seed)
    proxy = MpsMulticoinProxy.__new__(MpsMulticoinProxy)
    proxy.batch_size = candidates
    proxy.dispatch_batch_size = dispatch_batch_size
    proxy.interrupt_check = lambda: None
    proxy._torch = __import__("torch")
    proxy.profile_enabled = True
    proxy.last_profile = {}
    proxy.metrics_data = data
    proxy.run = runs[0]
    proxy.sides = ["long"]
    proxy.needed_metrics = needed_metrics
    proxy.strategy_kind = "ema_anchor"
    proxy.entry_interval_enabled = False
    proxy.btc_analysis_enabled = False
    proxy.btc_risk_enabled = False
    proxy.equity_balance_diff_enabled = False
    proxy.param_keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    proxy.base_params = {"long": {key: base_values[key] for key in proxy.param_keys}}
    proxy.base_total_wallet_exposure_limits = {"long": 1.0, "short": 0.0}
    proxy.base_n_positions = {"long": 4.0, "short": 0.0}
    proxy.fused_runner = None
    proxy.runners = {"long": runner}
    proxy._compute_objectives = compute_objectives
    candidate_dicts = [
        {f"long_{key}": float(row[index]) for index, key in enumerate(proxy.param_keys)}
        for row in matrix
    ]
    return (
        proxy,
        candidate_dicts,
        multicoin_bars,
        coins,
        1,
        _fixture_sha256(
            hlcvs,
            timestamps,
            matrix,
            overrides if overrides is not None else np.empty((0,), dtype=np.float32),
        ),
    )


def _run_once(proxy, candidates) -> dict:
    started = time.perf_counter()
    proxy.evaluate(candidates)
    wall_seconds = time.perf_counter() - started
    profile = dict(getattr(proxy, "last_profile", {}) or {})
    if profile:
        timings = profile["timings_seconds"]
        compile_seconds = float(timings["cold_compilation"]) + float(
            timings["warm_library_lookup"]
        )
        kernel_seconds = float(timings["kernel_execution"])
        device_to_host_seconds = float(timings["device_to_host"])
        host_overhead_seconds = float(timings["host_overhead"])
        cold = bool(profile["cold_dispatch_count"])
        batch_size = max(profile["actual_dispatch_batch_sizes"], default=0)
        dispatch_count = int(profile["dispatch_count"])
        dispatch_chunk_wall_seconds_max = max(
            (
                float(value)
                for value in profile.get("dispatch_chunk_wall_seconds", ())
            ),
            default=0.0,
        )
    else:
        runners = tuple(getattr(proxy, "runners", {}).values()) or (proxy.runner,)
        runner_profiles = [dict(runner.last_profile) for runner in runners]
        compile_seconds = sum(
            float(item.get("compile_seconds", 0.0)) for item in runner_profiles
        )
        kernel_seconds = sum(
            float(item.get("kernel_seconds", 0.0)) for item in runner_profiles
        )
        device_to_host_seconds = 0.0
        timed = sum(
            float(value)
            for item in runner_profiles
            for key, value in item.items()
            if key.endswith("_seconds")
        )
        host_overhead_seconds = max(0.0, wall_seconds - timed)
        cold = False
        batch_size = len(candidates)
        dispatch_count = len(runner_profiles)
        dispatch_chunk_wall_seconds_max = wall_seconds
    return {
        "batch_size": batch_size,
        "candidates_per_second": len(candidates) / max(wall_seconds, 1.0e-12),
        "cold": cold,
        "compile_seconds": compile_seconds,
        "device_to_host_seconds": device_to_host_seconds,
        "dispatch_count": dispatch_count,
        "dispatch_chunk_wall_seconds_max": dispatch_chunk_wall_seconds_max,
        "host_overhead_seconds": host_overhead_seconds,
        "kernel_seconds": kernel_seconds,
        "proxy_profile": profile,
        "wall_seconds": wall_seconds,
    }


def run_benchmark_case(
    name: str,
    *,
    candidates: int,
    dispatch_batch_size: int,
    warm_runs: int,
    single_bars: int,
    multicoin_bars: int,
    coins: int,
    seed: int,
) -> dict:
    proxy, candidate_dicts, bars, coin_count, side_count, fixture_sha256 = _build_case(
        name,
        candidates=candidates,
        dispatch_batch_size=dispatch_batch_size,
        single_bars=single_bars,
        multicoin_bars=multicoin_bars,
        coins=coins,
        seed=seed,
    )
    cold = _run_once(proxy, candidate_dicts)
    warm = [_run_once(proxy, candidate_dicts) for _ in range(warm_runs)]

    def median(key):
        return statistics.median(float(item[key]) for item in warm)

    cold_profile = cold["proxy_profile"]
    runner = getattr(proxy, "runner", None)
    return {
        "case": name,
        "seed": seed,
        "fixture_sha256": fixture_sha256,
        "candidate_count": candidates,
        "candle_count": bars,
        "coin_count": coin_count,
        "side_count": side_count,
        "candidate_bars": candidates * bars * coin_count * side_count,
        "kernel_candidate_bars": int(
            cold_profile.get(
                "kernel_candidate_bars",
                candidates * bars * coin_count * side_count,
            )
        ),
        "hsl_pnl_lookback_bars": int(
            getattr(runner, "pnl_lookback_bars", 0)
        ),
        "hsl_signal_mode": float(
            candidate_dicts[0].get("long_hsl_signal_mode", 0.0)
        ),
        "recursive_close_ladder_candidate_count": (
            _recursive_close_ladder_candidate_count(candidate_dicts)
        ),
        "recursive_entry_ladder_candidate_count": (
            _recursive_entry_ladder_candidate_count(candidate_dicts)
        ),
        "actual_dispatch_batch_size": int(cold["batch_size"]),
        "dispatch_chunk_count": int(
            cold_profile.get("dispatch_chunk_count", cold["dispatch_count"])
        ),
        "dispatch_count_per_run": int(cold["dispatch_count"]),
        "dispatch_chunk_wall_seconds_max": float(
            cold["dispatch_chunk_wall_seconds_max"]
        ),
        "cold": cold,
        "warm": {
            "runs": warm_runs,
            "wall_seconds_p50": median("wall_seconds"),
            "kernel_seconds_p50": median("kernel_seconds"),
            "device_to_host_seconds_p50": median("device_to_host_seconds"),
            "host_overhead_seconds_p50": median("host_overhead_seconds"),
            "dispatch_chunk_wall_seconds_max_p50": median(
                "dispatch_chunk_wall_seconds_max"
            ),
            "candidates_per_second_p50": median("candidates_per_second"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic, in-memory Apple MPS proxy benchmarks. No "
            "exchange, cache, config, or result files are accessed."
        )
    )
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument(
        "--dispatch-batch-size",
        type=int,
        help="Candidates per MPS dispatch (defaults to --candidates)",
    )
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--single-bars", type=int, default=60_000)
    parser.add_argument("--multicoin-bars", type=int, default=4_320)
    parser.add_argument("--coins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--compact", action="store_true")
    return parser


def _bounded_positive(parser, name: str, value: int, maximum: int) -> int:
    if value <= 0:
        parser.error(f"{name} must be greater than zero")
    if value > maximum:
        parser.error(f"{name} must be at most {maximum}")
    return value


def _require_mps_torch(parser):
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or str(exc.name).startswith("torch."):
            parser.error(
                "Apple MPS benchmarking requires the optional GPU dependencies; "
                'install with python3 -m pip install -e ".[full,gpu-mps]"'
            )
        raise
    if not torch.backends.mps.is_available():
        parser.error("Apple MPS is unavailable in this process")
    return torch


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    candidates = _bounded_positive(
        parser, "--candidates", args.candidates, MAX_CANDIDATES
    )
    dispatch_batch_size = (
        candidates
        if args.dispatch_batch_size is None
        else _bounded_positive(
            parser,
            "--dispatch-batch-size",
            args.dispatch_batch_size,
            candidates,
        )
    )
    warm_runs = _bounded_positive(parser, "--warm-runs", args.warm_runs, MAX_WARM_RUNS)
    single_bars = _bounded_positive(
        parser, "--single-bars", args.single_bars, MAX_SINGLE_BARS
    )
    multicoin_bars = _bounded_positive(
        parser,
        "--multicoin-bars",
        args.multicoin_bars,
        MAX_MULTICOIN_BARS,
    )
    coins = _bounded_positive(parser, "--coins", args.coins, MAX_COINS)
    if coins < 2:
        parser.error("--coins must be at least two")
    selected = CASES if args.case == "all" else (args.case,)
    if any(name in SINGLE_COIN_CASES for name in selected):
        if dispatch_batch_size * single_bars > MAX_DISPATCH_CANDIDATE_BARS:
            parser.error("single-coin candidate-bars exceed the safe benchmark limit")
    if any(name.startswith("ema-multicoin") for name in selected):
        if dispatch_batch_size * multicoin_bars * coins > MAX_DISPATCH_CANDIDATE_BARS:
            parser.error("multicoin candidate-bars exceed the safe benchmark limit")

    torch = _require_mps_torch(parser)
    report = {
        "schema_version": 1,
        "environment": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": True,
        },
        "cases": [
            run_benchmark_case(
                name,
                candidates=candidates,
                dispatch_batch_size=dispatch_batch_size,
                warm_runs=warm_runs,
                single_bars=single_bars,
                multicoin_bars=multicoin_bars,
                coins=coins,
                seed=args.seed,
            )
            for name in selected
        ],
    }
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
