from __future__ import annotations

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config import load_prepared_config
from config.access import require_config_value
from config.overrides import parse_overrides
from logging_setup import configure_logging, resolve_log_level
from rust_utils import collect_runtime_provenance, sha256_file
from suite_runner import extract_suite_config
from utils import format_approved_ignored_coins


class HarnessIndividual(list):
    """List subclass so optimizer evaluators can attach metadata to it."""


def collect_rust_binary_provenance() -> Dict[str, Any]:
    return collect_runtime_provenance()


def extract_metric_means(payload: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not isinstance(payload, dict):
        return metrics
    payload_root = payload
    if isinstance(payload.get("metrics"), dict):
        payload_root = payload["metrics"]
    stats = payload_root.get("stats")
    if isinstance(stats, dict):
        for metric, value in stats.items():
            if isinstance(value, dict) and value.get("mean") is not None:
                metrics[metric] = float(value["mean"])
            elif isinstance(value, (int, float)):
                metrics[metric] = float(value)
    suite_metrics = payload_root.get("suite_metrics")
    if suite_metrics is None:
        suite_metrics = payload.get("suite_metrics")
    if isinstance(suite_metrics, dict):
        for metric, entry in (suite_metrics.get("metrics") or {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("aggregated") is not None:
                metrics[metric] = float(entry["aggregated"])
                continue
            stats_entry = entry.get("stats")
            if isinstance(stats_entry, dict) and stats_entry.get("mean") is not None:
                metrics[metric] = float(stats_entry["mean"])
    return metrics


def select_metrics(
    metrics: Dict[str, float], preferred: Optional[Iterable[str]] = None
) -> Dict[str, float]:
    if preferred is None:
        preferred = (
            "adg_pnl",
            "adg_pnl_w",
            "mdg_pnl_w",
            "peak_recovery_hours_pnl",
            "position_held_hours_max",
            "backtest_completion_ratio",
        )
    out: Dict[str, float] = {}
    for key in preferred:
        if key in metrics:
            out[key] = metrics[key]
    return out


def compare_metric_maps(
    left: Dict[str, float], right: Dict[str, float], *, tolerance: float = 1e-12
) -> Dict[str, Dict[str, float]]:
    deltas: Dict[str, Dict[str, float]] = {}
    for key in sorted(set(left) | set(right)):
        lval = left.get(key)
        rval = right.get(key)
        if lval is None or rval is None:
            deltas[key] = {"left": lval, "right": rval, "abs_diff": float("inf")}
            continue
        diff = abs(float(lval) - float(rval))
        if diff > tolerance:
            deltas[key] = {"left": float(lval), "right": float(rval), "abs_diff": diff}
    return deltas


def _coin_source_map(
    backtest_coins: Dict[str, list[str]], msss: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for exchange, coins in backtest_coins.items():
        mss = msss.get(exchange, {})
        for coin in coins:
            source = mss.get(coin, {}).get("exchange", exchange)
            sources[coin] = str(source)
    return dict(sorted(sources.items()))


async def run_optimizer_replay(config: Dict[str, Any]) -> Dict[str, Any]:
    from backtest import prepare_hlcvs_mss
    from optimize import Evaluator, _maybe_aggregate_backtest_data, config_to_individual

    cfg = deepcopy(config)
    backtest_exchanges = require_config_value(cfg, "backtest.exchanges")
    await format_approved_ignored_coins(cfg, backtest_exchanges, verbose=False)
    cfg["backtest"]["coins"] = {}
    suite_cfg = extract_suite_config(cfg, None)
    suite_enabled = bool(suite_cfg.get("enabled"))
    if suite_enabled:
        raise NotImplementedError(
            "repro_harness currently supports non-suite optimizer replay only. "
            "Pass --suite to backtest.py separately if you need suite replay."
        )

    hlcvs_specs = {}
    btc_usd_specs = {}
    msss = {}
    timestamps = {}
    use_combined = len(backtest_exchanges) > 1
    shared_hlcvs_np = {}
    shared_btc_np = {}
    if use_combined:
        exchange = "combined"
        coins, hlcvs, mss, _results_path, _cache_dir, btc_usd_prices, ts = await prepare_hlcvs_mss(
            cfg, exchange
        )
        hlcvs, ts, btc_usd_prices = _maybe_aggregate_backtest_data(hlcvs, ts, btc_usd_prices, mss, cfg)
        cfg["backtest"]["coins"][exchange] = coins
        timestamps[exchange] = ts
        msss[exchange] = mss
        hlcvs_specs[exchange] = None
        btc_usd_specs[exchange] = None
        shared_hlcvs_np[exchange] = hlcvs
        shared_btc_np[exchange] = btc_usd_prices
    else:
        for exchange in backtest_exchanges:
            coins, hlcvs, mss, _results_path, _cache_dir, btc_usd_prices, ts = await prepare_hlcvs_mss(
                cfg, exchange
            )
            hlcvs, ts, btc_usd_prices = _maybe_aggregate_backtest_data(
                hlcvs, ts, btc_usd_prices, mss, cfg
            )
            cfg["backtest"]["coins"][exchange] = coins
            timestamps[exchange] = ts
            msss[exchange] = mss
            hlcvs_specs[exchange] = None
            btc_usd_specs[exchange] = None
            shared_hlcvs_np[exchange] = hlcvs
            shared_btc_np[exchange] = btc_usd_prices
    evaluator = Evaluator(
        hlcvs_specs=hlcvs_specs,
        btc_usd_specs=btc_usd_specs,
        msss=msss,
        config=cfg,
        timestamps=timestamps,
        shared_array_manager=None,
    )
    evaluator.shared_hlcvs_np.update(shared_hlcvs_np)
    evaluator.shared_btc_np.update(shared_btc_np)
    selected = deepcopy(cfg["backtest"]["coins"])
    source_map = _coin_source_map(cfg["backtest"]["coins"], msss)

    individual = HarnessIndividual(config_to_individual(cfg, evaluator.bounds, evaluator.sig_digits))
    overrides = cfg.get("optimize", {}).get("enable_overrides", [])
    objectives, penalty, metrics_payload = evaluator.evaluate(individual, overrides)
    return {
        "mode": "single",
        "objectives": list(objectives),
        "constraint_violation": float(penalty),
        "metrics": extract_metric_means(metrics_payload or {}),
        "metrics_payload": metrics_payload or {},
        "selected_coins": selected,
        "coin_sources": source_map,
        "aggregate_cfg": None,
    }


async def run_backtest_replay(config: Dict[str, Any]) -> Dict[str, Any]:
    from backtest import prepare_hlcvs_mss, run_backtest
    from suite_runner import run_backtest_suite_async

    cfg = deepcopy(config)
    backtest_exchanges = require_config_value(cfg, "backtest.exchanges")
    await format_approved_ignored_coins(cfg, backtest_exchanges, verbose=False)
    cfg["backtest"]["coins"] = {}
    cfg["backtest"]["cache_dir"] = {}
    suite_cfg = extract_suite_config(cfg, None)
    if suite_cfg.get("enabled"):
        summary = await run_backtest_suite_async(
            cfg,
            suite_cfg,
            disable_plotting=True,
        )
        metrics = extract_metric_means({"suite_metrics": summary.suite_metrics})
        selected = {
            result.scenario.label: deepcopy(result.per_exchange)
            for result in summary.scenarios
        }
        return {
            "mode": "suite",
            "metrics": metrics,
            "suite_metrics": summary.suite_metrics,
            "aggregate": summary.aggregate,
            "selected_coins": selected,
            "output_dir": str(summary.output_dir),
        }

    use_combined = len(backtest_exchanges) > 1
    if use_combined:
        exchange = "combined"
        coins, hlcvs, mss, _results_path, cache_dir, btc_usd_prices, timestamps = await prepare_hlcvs_mss(
            cfg, exchange
        )
        cfg["backtest"]["coins"][exchange] = coins
        cfg["backtest"]["cache_dir"][exchange] = str(cache_dir)
        _fills, _equities, analysis, _payload = run_backtest(
            hlcvs, mss, cfg, exchange, btc_usd_prices, timestamps, return_payload=True
        )
        metrics = {k: float(v) for k, v in analysis.items() if isinstance(v, (int, float))}
        return {
            "mode": "single",
            "exchange": exchange,
            "metrics": metrics,
            "selected_coins": deepcopy(cfg["backtest"]["coins"]),
            "coin_sources": _coin_source_map(cfg["backtest"]["coins"], {exchange: mss}),
        }

    exchange = backtest_exchanges[0]
    coins, hlcvs, mss, _results_path, cache_dir, btc_usd_prices, timestamps = await prepare_hlcvs_mss(
        cfg, exchange
    )
    cfg["backtest"]["coins"][exchange] = coins
    cfg["backtest"]["cache_dir"][exchange] = str(cache_dir)
    _fills, _equities, analysis, _payload = run_backtest(
        hlcvs, mss, cfg, exchange, btc_usd_prices, timestamps, return_payload=True
    )
    return {
        "mode": "single",
        "metrics": {k: float(v) for k, v in analysis.items() if isinstance(v, (int, float))},
        "selected_coins": deepcopy(cfg["backtest"]["coins"]),
        "coin_sources": _coin_source_map({exchange: coins}, {exchange: mss}),
    }


def build_report(
    *,
    config_path: str,
    stored_metrics: Dict[str, float],
    optimizer: Dict[str, Any],
    backtest: Dict[str, Any],
    rust: Dict[str, Any],
    metric_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    optimizer_metrics = optimizer.get("metrics", {})
    backtest_metrics_raw = backtest.get("metrics", {})
    if optimizer.get("mode") == "single" and backtest.get("mode") == "single" and all(
        isinstance(v, (int, float)) for v in backtest_metrics_raw.values()
    ):
        backtest_metrics = backtest_metrics_raw
    else:
        backtest_metrics = backtest_metrics_raw
    if metric_keys is not None:
        selected = tuple(metric_keys)
        optimizer_metrics = select_metrics(optimizer_metrics, selected)
        backtest_metrics = select_metrics(backtest_metrics, selected)
        stored_metrics = select_metrics(stored_metrics, selected)
    else:
        optimizer_metrics = select_metrics(optimizer_metrics)
        backtest_metrics = select_metrics(backtest_metrics)
        stored_metrics = select_metrics(stored_metrics)
    return {
        "config_path": config_path,
        "rust": rust,
        "stored_metrics": stored_metrics,
        "optimizer_replay": {
            "mode": optimizer.get("mode"),
            "objectives": optimizer.get("objectives"),
            "constraint_violation": optimizer.get("constraint_violation"),
            "metrics": optimizer_metrics,
            "selected_coins": optimizer.get("selected_coins"),
            "coin_sources": optimizer.get("coin_sources"),
        },
        "backtest_replay": {
            "mode": backtest.get("mode"),
            "metrics": backtest_metrics,
            "selected_coins": backtest.get("selected_coins"),
            "coin_sources": backtest.get("coin_sources"),
        },
        "diffs": {
            "stored_vs_optimizer": compare_metric_maps(stored_metrics, optimizer_metrics),
            "stored_vs_backtest": compare_metric_maps(stored_metrics, backtest_metrics),
            "optimizer_vs_backtest": compare_metric_maps(optimizer_metrics, backtest_metrics),
        },
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"Config: {report['config_path']}")
    rust = report["rust"]
    print("Rust:")
    print(f"  runtime:   {rust.get('runtime_module_path')}")
    print(f"  preferred: {rust.get('preferred_compiled_path')}")
    print(f"  same_hash: {rust.get('runtime_matches_preferred')}")

    for label in ("stored_metrics", "optimizer_replay", "backtest_replay"):
        section = report[label]
        if label == "stored_metrics":
            metrics = section
            print("\nStored metrics:")
        else:
            metrics = section.get("metrics", {})
            print(f"\n{label.replace('_', ' ').title()}:")
        for key, value in metrics.items():
            print(f"  {key:<28} {value}")

    print("\nDiffs:")
    for label, diff_map in report["diffs"].items():
        print(f"  {label}: {len(diff_map)} differing metrics")
        for key, payload in diff_map.items():
            print(
                f"    {key:<28} left={payload.get('left')} right={payload.get('right')} abs_diff={payload.get('abs_diff')}"
            )


async def async_main(args: argparse.Namespace) -> int:
    configure_logging(resolve_log_level(args.log_level, None))
    raw_path = Path(args.config_path)
    raw_loaded: Dict[str, Any] | None = None
    if raw_path.suffix.lower() == ".json" and raw_path.exists():
        try:
            raw_loaded = json.loads(raw_path.read_text())
        except json.JSONDecodeError:
            raw_loaded = None
    raw_config = load_prepared_config(args.config_path, verbose=False)
    stored_metrics = extract_metric_means(raw_loaded or raw_config)
    config = deepcopy(raw_config)
    config = parse_overrides(config, verbose=False)
    rust = collect_rust_binary_provenance()
    optimizer = await run_optimizer_replay(config)
    backtest = await run_backtest_replay(config)
    metric_keys = None
    if args.metrics:
        metric_keys = [key.strip() for key in args.metrics.split(",") if key.strip()]
    report = build_report(
        config_path=args.config_path,
        stored_metrics=stored_metrics,
        optimizer=optimizer,
        backtest=backtest,
        rust=rust,
        metric_keys=metric_keys,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a config through the optimizer evaluation path and the backtest path in the "
            "same process, then compare metrics and Rust binary provenance."
        )
    )
    parser.add_argument("config_path", help="Config or Pareto JSON to replay.")
    parser.add_argument(
        "--metrics",
        default="",
        help=(
            "Comma-separated metrics to compare. Defaults to adg_pnl, adg_pnl_w, mdg_pnl_w, "
            "peak_recovery_hours_pnl, position_held_hours_max, backtest_completion_ratio."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the harness runtime.",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
