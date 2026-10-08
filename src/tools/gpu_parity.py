"""Offline CPU/GPU metric comparisons for prepared inputs or reproducible fixtures.

This development tool intentionally runs both simulators. It is never called by an
optimizer and makes no market-data requests. Reports omit configs and full histories;
optional diagnostics include bounded fill/state summaries.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack, contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import struct
import sys
import time

from optimization.gpu.parity import MetricTolerance, compare_limits, compare_metrics


DEFAULT_METRICS = ("adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day")
FIXTURE_DEFAULTS = {
    "sides": "long", "coins": 2, "bars": 5760, "seed": 7, "hsl": "disabled",
    "unstuck": False, "market_orders": False, "filter_by_min_effective_cost": False,
    "hsl_red_threshold": 0.05, "hsl_ema_span_minutes": 30.0,
    "hsl_cooldown_minutes": 30.0, "hsl_lookback_days": 1.0, "price_shocks": [],
}
STRESS_OPTIONS = ("hsl_red_threshold", "hsl_ema_span_minutes", "hsl_cooldown_minutes",
                  "hsl_lookback_days", "price_shocks")
# Provisional measurement gates for selected definitions, not a release certificate.
DEFAULT_TOLERANCES = {
    "adg_strategy_eq": MetricTolerance(1e-7, 1e-4),
    "drawdown_worst_strategy_eq": MetricTolerance(1e-5, 1e-4),
    "fills_per_day": MetricTolerance(1e-6, 1e-5),
    "backtest_completion_ratio": MetricTolerance(1e-6, 1e-6),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", choices=("ema_anchor", "trailing_martingale"))
    source.add_argument("--config", help="Config for an already prepared NPZ dataset")
    parser.add_argument("--dataset", help="NPZ: hlcvs, timestamps, btc and ordered coins")
    parser.add_argument("--markets", help="JSON market settings mapping, including __meta__")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--sides", choices=("long", "short", "both"))
    parser.add_argument("--coins", type=int)
    parser.add_argument("--bars", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hsl", choices=("disabled", "coin", "pside", "unified"))
    add_fixture_stress_options(parser)
    parser.add_argument("--unstuck", action="store_true", default=None)
    parser.add_argument("--market-orders", action="store_true", default=None)
    parser.add_argument("--filter-by-min-effective-cost", action="store_true", default=None)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--tolerances", help="JSON per-metric absolute/relative/sentinel policy")
    parser.add_argument("--report", help="Save a standard-JSON report here")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Include bounded fill/state summaries")
    parser.add_argument("--gpu-engine", choices=("legacy", "native"), default="legacy",
                        help="GPU replay path: legacy selection or the native CUDA optimizer service")
    return parser


def add_fixture_stress_options(parser):
    for name in STRESS_OPTIONS[:-1]:
        parser.add_argument("--" + name.replace("_", "-"), type=float,
                            help="Synthetic fixture HSL policy only; does not enable HSL")
    parser.add_argument("--price-shock", dest="price_shocks", nargs=3, action="append",
                        metavar=("COIN_INDEX", "BAR", "FACTOR"),
                        help="Multiply fixture high/low/close from BAR onward; repeat for more shocks")


def resolve_fixture_args(args):
    """Validate and normalize the shared recipe before preparing either simulator."""
    args = argparse.Namespace(**vars(args))
    for name, default in FIXTURE_DEFAULTS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, deepcopy(default))
    if not 1 <= args.coins <= 64 or not 61 <= args.bars <= 100_000:
        raise ValueError("fixtures require 1..64 coins and 61..100000 bars")
    for name, lower, upper in (("hsl_red_threshold", 0, 1),
                              ("hsl_ema_span_minutes", 1, math.inf),
                              ("hsl_cooldown_minutes", 0, math.inf),
                              ("hsl_lookback_days", 1, 90)):
        value = getattr(args, name)
        if not math.isfinite(value) or not lower <= value <= upper or (
                name == "hsl_red_threshold" and value == 0):
            raise ValueError(f"invalid fixture --{name.replace('_', '-')}")
        try:
            encoded = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as error:
            raise ValueError(f"fixture --{name.replace('_', '-')} exceeds GPU float32 range") from error
        if not math.isfinite(encoded):
            raise ValueError(f"fixture --{name.replace('_', '-')} exceeds GPU float32 range")
        if name == "hsl_red_threshold" and encoded <= 0:
            raise ValueError("fixture --hsl-red-threshold must remain positive in GPU float32")
    shocks = []
    # Rust converts this duration to an i64 millisecond timestamp.
    if args.hsl_cooldown_minutes * 60_000 >= float(2**63 - 1):
        raise ValueError("fixture --hsl-cooldown-minutes exceeds Rust's timestamp range")
    for coin, bar, factor in args.price_shocks:
        try:
            coin, bar, factor = int(coin), int(bar), float(factor)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("--price-shock requires integer COIN_INDEX/BAR and a positive finite FACTOR") from error
        if not (0 <= coin < args.coins and 0 <= bar < args.bars
                and math.isfinite(factor) and factor > 0):
            raise ValueError("--price-shock must refer to a fixture coin/bar and a positive finite factor")
        shocks.append([coin, bar, factor])
    args.price_shocks = shocks
    return args


def _fixture_candles(args):
    """Apply the normalized recipe and check the encoding before either backtest."""
    import numpy as np
    from tools.synthetic_backtest_data import synthetic_hlcvs

    hlcvs, timestamps = synthetic_hlcvs(args.bars, args.coins, args.seed)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for coin, bar, factor in args.price_shocks:
                hlcvs[bar:, coin, :3] *= factor
            encoded = hlcvs[:, :, :3].astype(np.float32)
    except FloatingPointError as error:
        raise ValueError("fixture high/low/close must encode as positive finite GPU float32") from error
    if not np.isfinite(encoded).all() or np.any(encoded <= 0):
        raise ValueError("fixture high/low/close must encode as positive finite GPU float32")
    return hlcvs, timestamps


def fixture_inputs(args):
    import numpy as np
    from config import prepare_config
    from config.schema import get_template_config
    from config.hsl import generated_template

    args = resolve_fixture_args(args)
    coins = [f"COIN{i:02d}" for i in range(args.coins)]
    hlcvs, timestamps = _fixture_candles(args)
    # The canonical backtest end date has UTC-day precision. Align the synthetic
    # exclusive endpoint to midnight so both engines see exactly the same span,
    # including small fixtures which do not contain a whole number of days.
    timestamps = timestamps - ((int(timestamps[-1]) + 60_000) % 86_400_000)
    active = ("long", "short") if args.sides == "both" else (args.sides,)
    config = generated_template(
        get_template_config(), "coin" if args.hsl == "disabled" else args.hsl
    )
    # This is a backtest fixture, with no optimizer genes/runtime override policy.
    config["optimize"]["bounds"] = {}
    config["optimize"]["fixed_runtime_overrides"] = {}
    config["live"].update(
        strategy_kind=args.fixture,
        hedge_mode=args.sides == "both",
        max_warmup_minutes=60,
        minimum_coin_age_days=0,
        market_orders_allowed=args.market_orders,
        pnls_max_lookback_days=args.hsl_lookback_days,
        hsl_signal_mode="coin" if args.hsl == "disabled" else args.hsl,
        approved_coins={side: coins if side in active else [] for side in ("long", "short")},
        ignored_coins={"long": [], "short": []},
    )
    config["backtest"].update(
        exchanges=[args.exchange], coins={args.exchange: coins}, starting_balance=1000.0,
        btc_collateral_cap=0.0, filter_by_min_effective_cost=args.filter_by_min_effective_cost,
        start_date=datetime.fromtimestamp(int(timestamps[0]) / 1000, timezone.utc).isoformat(),
        end_date=datetime.fromtimestamp((int(timestamps[-1]) + 60_000) / 1000, timezone.utc).isoformat(),
    )
    for side in ("long", "short"):
        bot = config["bot"][side]
        bot["risk"].update(
            n_positions=args.coins if side in active else 0,
            total_wallet_exposure_limit=1.0 if side in active else 0.0,
            we_excess_allowance_pct=0.0,
            position_exposure_enforcer_enabled=False,
            total_exposure_enforcer_enabled=False,
        )
        bot["hsl"].update(
            enabled=side in active and args.hsl in {"coin", "pside"},
            red_threshold=args.hsl_red_threshold, ema_span_minutes=args.hsl_ema_span_minutes,
            cooldown_minutes_after_red=args.hsl_cooldown_minutes, restart_after_red_policy="always",
        )
        bot["unstuck"]["enabled"] = args.unstuck and side in active
        bot["entry_cooldown"]["base_duration_minutes"] = 0.0
    if args.hsl == "unified":
        config["bot"]["hsl"].update(
            enabled=True, red_threshold=args.hsl_red_threshold,
            ema_span_minutes=args.hsl_ema_span_minutes,
            cooldown_minutes_after_red=args.hsl_cooldown_minutes, restart_after_red_policy="always",
        )
    config = prepare_config(config, verbose=False, target="canonical", runtime=None)
    # Ordered coins are derived preparation metadata, outside the canonical config.
    config["backtest"]["coins"] = {args.exchange: coins}
    btc = np.full(args.bars, 50_000.0, dtype=np.float64)
    markets = {
        coin: dict(
            qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=1.0,
            c_mult=1.0, maker=0.0002, taker=0.0005, exchange=args.exchange,
            first_valid_index=0, last_valid_index=args.bars - 1, warmup_minutes=60,
        )
        for coin in coins
    }
    markets["__meta__"] = {"requested_start_ts": int(timestamps[0])}
    return config, hlcvs, markets, btc, timestamps


def prepared_inputs(args):
    specified = ["--" + name.replace("_", "-") for name in FIXTURE_DEFAULTS
                 if getattr(args, name) is not None]
    if specified:
        raise ValueError("fixture-only options cannot be used with --config: " + ", ".join(specified))
    import numpy as np
    from config import load_input_config, prepare_config
    from config.migrations import detect_flavor
    from hlcv_preparation import _filter_forced_sources_for_coins, _filter_market_settings_sources_for_coins
    from optimization.warmup import _apply_config_overrides
    from utils import to_standard_exchange_name

    if not args.dataset or not args.markets:
        raise ValueError("--config requires --dataset and --markets")
    source, base, snapshot = load_input_config(args.config, log_info=False)
    # Use the same document payload as canonical config normalization. Outer
    # metadata must not hide a wrapped candidate's declared data identities.
    wrapped = detect_flavor(source, {}) == "nested_current"
    source_payload = source["config"] if wrapped else source
    snapshot_payload = snapshot["config"] if wrapped else snapshot
    # Gene bounds are not inputs to this standalone backtest comparison. Removing
    # them also permits effective candidate exports without a reusable opt search.
    source_payload.setdefault("optimize", {})["bounds"] = {}
    snapshot_payload.setdefault("optimize", {})["bounds"] = {}
    config = prepare_config(source, base_config_path=base, raw_snapshot=snapshot, verbose=False)
    if config["optimize"]["enable_overrides"]:
        raise ValueError("prepared comparisons require materialized optimize.enable_overrides")
    fixed = config["optimize"]["fixed_runtime_overrides"]
    if fixed:
        _apply_config_overrides(config, fixed)
        config["optimize"]["bounds"] = {}
        config = prepare_config(
            config, raw_snapshot=deepcopy(config), verbose=False, target="canonical", runtime=None
        )
    exchanges = [to_standard_exchange_name(value) for value in config["backtest"]["exchanges"]]
    expected_exchange = "combined" if len(exchanges) > 1 else exchanges[0]
    exchange = to_standard_exchange_name(args.exchange)
    if exchange != expected_exchange:
        raise ValueError("prepared exchange must match effective config.backtest.exchanges "
                         f"(expected {expected_exchange!r}, got {args.exchange!r})")
    with np.load(args.dataset, allow_pickle=False) as bundle:
        hlcvs, timestamps, btc, coins = (bundle[name] for name in ("hlcvs", "timestamps", "btc", "coins"))
    if coins.ndim != 1 or coins.dtype.kind not in {"U", "S"}:
        raise ValueError("dataset coins must be a one-dimensional string array")
    ordered = [item.decode() if isinstance(item, bytes) else str(item) for item in coins]
    if len(set(ordered)) != len(ordered):
        raise ValueError("dataset coin identities must be unique")
    # prep_backtest_args sorts coin identities while candle columns remain in
    # their input order. Accept only its canonical layout at this tool boundary.
    if ordered != sorted(ordered):
        raise ValueError("dataset requires sorted coin order to match backtest payload construction")
    declared_coins = {}
    for venue, identities in source_payload.get("backtest", {}).get("coins", {}).items():
        venue = to_standard_exchange_name(venue)
        if venue in declared_coins and declared_coins[venue] != identities:
            raise ValueError("config.backtest.coins has conflicting coin lists for exchange aliases")
        declared_coins[venue] = identities
    if declared_coins and exchange not in declared_coins:
        raise ValueError("prepared exchange must match config.backtest.coins")
    declared = declared_coins.get(exchange)
    if declared is not None and declared != ordered:
        raise ValueError("dataset coin order must exactly match config.backtest.coins")
    config["backtest"]["coins"] = {exchange: ordered}
    if hlcvs.ndim != 3 or hlcvs.shape[1:] != (len(ordered), 4):
        raise ValueError("dataset hlcvs must have shape (bars, coins, 4)")
    if timestamps.shape != (len(hlcvs),) or btc.shape != timestamps.shape:
        raise ValueError("dataset timestamps/BTC prices must align with candle rows")
    markets = json.loads(Path(args.markets).read_text())
    forced_sources, settings_sources = {}, {}
    if exchange == "combined":
        forced_sources = {
            coin: to_standard_exchange_name(venue) for coin, venue in
            _filter_forced_sources_for_coins(config["backtest"].get("coin_sources", {}), ordered).items()
        }
        settings_sources = {
            coin: to_standard_exchange_name(venue) for coin, venue in
            _filter_market_settings_sources_for_coins(
                config["backtest"].get("market_settings_sources", {}), ordered
            ).items()
        }
    for coin in ordered:
        market_exchange = markets[coin].get("exchange")
        if not isinstance(market_exchange, str) or not market_exchange:
            raise ValueError(f"prepared market exchange for {coin} is required")
        market_exchange = to_standard_exchange_name(market_exchange)
        candle_exchange = to_standard_exchange_name(markets[coin].get("ohlcv_source", market_exchange))
        allowed_candles = {forced_sources[coin]} if coin in forced_sources else set(exchanges)
        if candle_exchange not in allowed_candles:
            field = "coin_sources" if coin in forced_sources else "exchanges/data sources"
            raise ValueError(f"prepared candle exchange for {coin} must match config.backtest.{field}")
        # Combined preparation may use a separate settings venue, or explicitly
        # fall back to the candle venue when that settings source is unavailable
        # or has a different denomination. Preserve this producer-resolved input.
        allowed_settings = {settings_sources.get(coin, candle_exchange), candle_exchange}
        if market_exchange not in allowed_settings:
            raise ValueError(f"prepared market exchange for {coin} must match config.backtest.market_settings_sources")
    return config, hlcvs, markets, btc, timestamps


def _identity(config, arrays, markets, exchange):
    import numpy as np

    # Transform bookkeeping is not a simulation input. No config contents are emitted.
    def effective(value):
        if isinstance(value, dict):
            return {key: effective(item) for key, item in value.items() if not key.startswith("_")}
        if isinstance(value, list):
            return [effective(item) for item in value]
        return value

    digest = hashlib.sha256(json.dumps(
        {"config": effective(config), "markets": markets, "exchange": exchange},
        sort_keys=True, allow_nan=False,
    ).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _source_fingerprint():
    """Identify Python simulation/preparation/reduction code without emitting paths."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def _native_dataset(inputs, exchange, metrics):
    from optimization.gpu.datasets import PreparedGpuDataset
    from shared_arrays import SharedArrayManager

    config, candles, markets, btc, timestamps = inputs
    arrays = SharedArrayManager()
    resources = ExitStack()
    try:
        specs = []
        for value in (candles, btc, timestamps):
            spec = arrays.create_from(value)[0]
            specs.append(spec)
            resources.callback(arrays.cleanup, [spec])
        yield PreparedGpuDataset(
            config=config, markets=markets, exchange=exchange,
            hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
            candle_coins=config["backtest"]["coins"][exchange], metrics=metrics,
        )
    except BaseException:
        try:
            resources.close()
        except BaseException:
            logging.exception("native parity dataset cleanup failed after an earlier failure")
        raise
    else:
        resources.close()


def run_comparison(inputs, exchange, metrics, policies, checks=(), *, diagnostics=False, gpu_engine="legacy"):
    if gpu_engine not in {"legacy", "native"}:
        raise ValueError("GPU parity engine must be legacy or native")
    from rust_utils import check_and_maybe_compile, verify_loaded_runtime_extension

    if "passivbot_rust" not in sys.modules:
        check_and_maybe_compile(fail_on_stale=True)
        __import__("passivbot_rust")
    runtime = verify_loaded_runtime_extension()
    if runtime.get("skipped") or runtime["runtime_compiled_source_stamp"] != runtime["expected_source_fingerprint"]:
        raise RuntimeError("parity requires a real source-fingerprint-verified Rust extension")
    from backtest import build_backtest_payload, execute_backtest
    from config.metrics import resolve_metric_value
    from optimization.gpu.executor import BacktestRequest, GpuBacktestService
    from optimization.gpu.service import MpsMulticoinProxy, MpsSingleCoinProxy

    config, hlcvs, markets, btc, timestamps = inputs
    identity = _identity(config, (hlcvs, btc, timestamps), markets, exchange)
    payload = build_backtest_payload(
        hlcvs, markets, deepcopy(config), exchange, btc, timestamps,
        metrics_only=not diagnostics, skip_btc_analysis=not any(name.endswith("_btc") for name in metrics),
    )
    started = time.perf_counter()
    fills, equities, analysis = execute_backtest(payload, config)
    cpu_seconds = time.perf_counter() - started
    cpu = {name: value for name in metrics if (value := resolve_metric_value(analysis, name)) is not None}
    replay_cls = MpsSingleCoinProxy if hlcvs.shape[1] == 1 else MpsMulticoinProxy
    preparation_seconds = 0.0
    states = {}

    @contextmanager
    def replay_factory():
        nonlocal preparation_seconds
        started = time.perf_counter()
        replay = replay_cls(
            config=deepcopy(config), hlcvs=hlcvs, mss=markets, btc=btc,
            timestamps=timestamps, exchange=exchange, batch_size=1, needed_metrics=set(metrics),
        )
        preparation_seconds = time.perf_counter() - started
        hooks = []
        try:
            if diagnostics:
                runners = ({"single": replay.runner} if hasattr(replay, "runner") else
                           {"fused": replay.fused_runner} if replay.fused_runner is not None else replay.runners)
                for label, runner in runners.items():
                    original = runner.run

                    def captured(*args, _label=label, _original=original, **kwargs):
                        output = _original(*args, **kwargs)
                        states[_label] = {
                            key: float(output[key].item())
                            for key in ("fill_count", "psize", "short_psize", "pprice", "short_pprice",
                                        "balance", "first_eq_ts", "last_eq_ts")
                            if key in output and output[key].numel() == 1
                        }
                        return output

                    hooks.append((runner, original))
                    runner.run = captured
            yield replay
        finally:
            for runner, original in hooks:
                runner.run = original

    if gpu_engine == "native":
        from optimization.gpu.native import CudaBacktestService

        with _native_dataset(inputs, exchange, metrics) as dataset:
            with CudaBacktestService(batch_size=1, tuning_mode="off") as service:
                service.register_dataset(identity, dataset)
                started = time.perf_counter()
                result = service.submit(BacktestRequest(identity, identity, {})).result()
                gpu_seconds = time.perf_counter() - started
        # Preparation stays behind the native worker boundary. Report its full
        # cold request time rather than inventing a separate preparation timing.
        preparation_seconds = None
        states = {"native_result": {"liquidated": result.liquidated}} if diagnostics else {}
    else:
        with GpuBacktestService(batch_size=1) as service:
            service.register_dataset_factory(identity, replay_factory)
            started = time.perf_counter()
            result = service.submit(BacktestRequest(identity, identity, {})).result()
            gpu_seconds = time.perf_counter() - started - preparation_seconds
    report = compare_metrics(cpu, result.metrics, {name: policies.get(name) for name in metrics})
    report["feasibility"] = compare_limits(cpu, result.metrics, checks)
    report["passed"] = report["passed"] and report["feasibility"]["passed"]
    statuses = {row["status"] for row in report["metrics"].values()}
    status = "passed" if report["passed"] else "mismatch"
    if statuses & {"missing_cpu", "missing_gpu", "missing_both", "unassessed"}:
        status = "comparison_incomplete"
    if not report["feasibility"]["assessed"]:
        status = "comparison_incomplete"
    report.update(
        schema_version=1, status=status,
        evaluation_id=identity, bars=int(hlcvs.shape[0]), coins=int(hlcvs.shape[1]),
        strategy=config["live"]["strategy_kind"],
        gpu_engine=gpu_engine,
        gpu_replay="shared_account" if gpu_engine == "native" or hlcvs.shape[1] > 1 else "single_coin",
        rust_source_fingerprint=runtime["expected_source_fingerprint"],
        python_source_fingerprint=_source_fingerprint(),
        timings_seconds=dict(cpu=cpu_seconds, gpu_prepare=preparation_seconds, gpu_cold=gpu_seconds),
        tolerance_policy={name: vars(policies[name]) if name in policies else None
                          for name in metrics},
    )
    if diagnostics:
        report["diagnostics"] = {
            "cpu": {
                "fill_count": len(fills),
                "first_equity_timestamp": float(equities[0, 0]) if len(equities) else None,
                "last_equity_timestamp": float(equities[-1, 0]) if len(equities) else None,
                "absolute_position_quantity": {
                    side: abs(sum(float(fill[9]) for fill in fills if str(fill[13]).endswith(side)))
                    for side in ("long", "short")
                },
                "last_fills": [
                    {"step": int(fill[0]), "timestamp": int(fill[1]), "coin": str(fill[2]),
                     "quantity": float(fill[9]), "price": float(fill[10]), "order_type": str(fill[13])}
                    for fill in fills[-8:]
                ],
            },
            "gpu": states,
        }
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fixture and (args.dataset or args.markets):
        parser.error("--dataset/--markets are only used with --config")
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    report = None
    fixture_recipe = None
    stage = "metric_contract"
    try:
        from config.metrics import canonicalize_metric_name
        from optimization.gpu.metrics import validate_gpu_metric_names
        from config.scoring import default_scoring_weights, extract_objective_specs
        from limit_utils import expand_limit_checks
        from utils import to_standard_exchange_name

        metrics = list(dict.fromkeys(canonicalize_metric_name(name) for name in args.metrics))
        validate_gpu_metric_names(metrics)
        stage = "inputs"
        args.exchange = to_standard_exchange_name(args.exchange)
        policies = dict(DEFAULT_TOLERANCES)
        if args.tolerances:
            for name, value in json.loads(Path(args.tolerances).read_text()).items():
                policies[canonicalize_metric_name(name)] = MetricTolerance(**value)
        if args.fixture:
            recipe = resolve_fixture_args(args)
            fixture_recipe = {name: getattr(recipe, name) for name in FIXTURE_DEFAULTS}
            fixture_recipe.update(fixture=args.fixture, exchange=args.exchange)
        inputs = fixture_inputs(args) if args.fixture else prepared_inputs(args)
        config = inputs[0]
        weights = default_scoring_weights()
        weights.update({spec.metric: -1.0 if spec.goal == "max" else 1.0
                        for spec in extract_objective_specs(config)})
        checks = expand_limit_checks(
            config["optimize"]["limits"], weights, penalty_weight=1.0,
            reducer_cfg=config["backtest"]["reducer"],
        )
        metrics = list(dict.fromkeys([*metrics, *(check["metric"] for check in checks)]))
        stage = "metric_contract"
        validate_gpu_metric_names(metrics)
        stage = "simulation"
        with redirect_stdout(sys.stderr):
            report = run_comparison(inputs, args.exchange, metrics, policies, checks,
                                    diagnostics=args.diagnostics, gpu_engine=args.gpu_engine)
        code = 0 if report["passed"] else 1
    except Exception as error:
        # Diagnostic boundary: failed execution is never reported as a metric match.
        status = "input_failed" if stage == "inputs" else "execution_failed"
        if stage == "metric_contract" and isinstance(error, ValueError):
            status = "unsupported"
        report = {"schema_version": 1, "status": status, "stage": stage, "passed": False,
                  "gpu_engine": args.gpu_engine,
                  "error": {"type": type(error).__name__, "message": str(error)}}
        code = 2
    if fixture_recipe is not None:
        report["fixture_recipe"] = fixture_recipe
    rendered = json.dumps(report, allow_nan=False, indent=None if args.compact else 2, sort_keys=True)
    print(rendered)
    if args.report:
        try:
            Path(args.report).write_text(rendered + "\n")
        except OSError as error:
            print(f"report_save_failed: {error}", file=sys.stderr)
            return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
