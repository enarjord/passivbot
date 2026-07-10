from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from typing import Any

import passivbot_rust as pbr

from passivbot import Passivbot


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_MINUTES = 240
DEFAULT_SYMBOLS = 8
DEFAULT_ITERATIONS = 1
MAX_MINUTES = 1_440
MAX_SYMBOLS = 32
MAX_ITERATIONS = 20


def _fixture_symbol(index: int) -> str:
    return f"HSLBENCH{index:02d}/USDT:USDT"


def build_coin_hsl_history_fixture(minutes: int, symbols: int) -> dict[str, Any]:
    """Build a bounded, deterministic coin-HSL history with one active long per symbol."""
    if not 1 <= int(minutes) <= MAX_MINUTES:
        raise ValueError(f"minutes must be between 1 and {MAX_MINUTES}")
    if not 1 <= int(symbols) <= MAX_SYMBOLS:
        raise ValueError(f"symbols must be between 1 and {MAX_SYMBOLS}")

    names = [_fixture_symbol(index) for index in range(int(symbols))]
    timeline: list[dict[str, Any]] = []
    fill_events: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(names):
        fill_events.append(
            {
                "timestamp": 1,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
                "id": f"fixture-open-{symbol_index}",
            }
        )
    for minute in range(1, int(minutes) + 1):
        realized_by_symbol: dict[str, dict[str, float]] = {}
        unrealized_by_symbol: dict[str, dict[str, float]] = {}
        total_realized = 0.0
        for symbol_index, symbol in enumerate(names):
            realized = float((minute * (symbol_index + 3)) % 19) / 100.0
            unrealized = -float((minute + symbol_index * 5) % 11) / 100.0
            realized_by_symbol[symbol] = {"long": realized}
            unrealized_by_symbol[symbol] = {"long": unrealized}
            total_realized += realized
        timeline.append(
            {
                "timestamp": minute * 60_000,
                "balance": 1_000.0,
                "realized_pnl": total_realized,
                "realized_pnl_by_coin_pside": realized_by_symbol,
                "unrealized_pnl_by_coin_pside": unrealized_by_symbol,
            }
        )
    return {
        "timeline": timeline,
        "fill_events": fill_events,
        "panic_flatten_events": [],
    }


def _fixture_digest(history: dict[str, Any]) -> str:
    encoded = json.dumps(history, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_offline_replay_bot(history: dict[str, Any], timings: dict[str, dict[str, int]]) -> Passivbot:
    """Create an uninitialized bot that can only replay supplied in-memory history."""
    bot = Passivbot.__new__(Passivbot)
    symbols = sorted(history["timeline"][0]["realized_pnl_by_coin_pside"])
    now_ms = int(history["timeline"][-1]["timestamp"])
    side_effects = {
        "network_calls": 0,
        "cache_reads": 0,
        "cache_writes": 0,
        "latch_writes": 0,
        "latch_removals": 0,
        "monitor_events": 0,
    }

    bot.user = "offline_hsl_replay_benchmark"
    bot.exchange = "offline"
    bot.market_type = "swap"
    bot._equity_hard_stop = {
        "long": bot._equity_hard_stop_make_state(),
        "short": bot._equity_hard_stop_make_state(),
    }
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._runtime_forced_modes = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
        for symbol in symbols
    }
    bot.open_orders = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.c_mults = {}
    bot.qty_steps = {}
    bot.config = {
        "live": {
            "hsl_signal_mode": "coin",
            "hsl_position_during_cooldown_policy": "panic",
            "pnls_max_lookback_days": 30.0,
        }
    }
    bot.hsl = {
        "long": {
            "enabled": True,
            "red_threshold": 0.95,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 15.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 1.0,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
        "short": {
            "enabled": False,
            "red_threshold": 0.95,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 15.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 1.0,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
    }
    bot.get_raw_balance = lambda: 1_000.0
    bot.get_exchange_time = lambda: now_ms
    bot.live_value = lambda key: bot.config["live"][key]
    bot._equity_hard_stop_realized_pnl_now = lambda pside=None: 0.0
    bot.bot_value = lambda pside, key: {
        "n_positions": len(symbols),
        "total_wallet_exposure_limit": 1.0,
    }[key]

    async def history_provider(current_balance=None, **kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return history
        finally:
            timings["history_load"]["calls"] += 1
            timings["history_load"]["elapsed_ns"] += time.perf_counter_ns() - started_ns

    async def no_cache_reuse(*_args, **_kwargs):
        timings["cache_reuse_skipped"]["calls"] += 1
        return None

    async def current_upnl(*_args, **_kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return 0.0
        finally:
            timings["current_upnl"]["calls"] += 1
            timings["current_upnl"]["elapsed_ns"] += time.perf_counter_ns() - started_ns

    def skip_cache_persist(*_args, **_kwargs):
        timings["cache_persist_skipped"]["calls"] += 1
        return 0

    def skip_latch_write(*_args, **_kwargs):
        side_effects["latch_writes"] += 1
        return None

    def skip_latch_remove(*_args, **_kwargs):
        side_effects["latch_removals"] += 1

    original_apply_metrics = bot._equity_hard_stop_apply_coin_metrics_sample

    def profile_coin_metrics_sample(*args, **kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return original_apply_metrics(*args, **kwargs)
        finally:
            timings["coin_metrics_sample"]["calls"] += 1
            timings["coin_metrics_sample"]["elapsed_ns"] += time.perf_counter_ns() - started_ns

    bot.get_balance_equity_history = history_provider
    bot._equity_hard_stop_try_reuse_replay_cache = no_cache_reuse
    bot._calc_upnl_sum_strict = current_upnl
    bot._equity_hard_stop_persist_replay_matrices = skip_cache_persist
    bot._equity_hard_stop_write_latch = skip_latch_write
    bot._equity_hard_stop_remove_latch_file = skip_latch_remove
    bot._equity_hard_stop_apply_coin_metrics_sample = profile_coin_metrics_sample
    bot._offline_hsl_benchmark_side_effects = side_effects
    return bot


def _state_digest(bot: Passivbot) -> str:
    state_projection = []
    for pside, symbols in sorted(bot._equity_hard_stop_coin.items()):
        for symbol, state in sorted(symbols.items()):
            metrics = state.get("last_metrics") or {}
            state_projection.append(
                {
                    "pside": pside,
                    "symbol": symbol,
                    "halted": bool(state["halted"]),
                    "no_restart_latched": bool(state["no_restart_latched"]),
                    "cooldown_until_ms": state["cooldown_until_ms"],
                    "tier": metrics.get("tier"),
                    "drawdown_raw": metrics.get("drawdown_raw"),
                    "drawdown_ema": metrics.get("drawdown_ema"),
                    "drawdown_score": metrics.get("drawdown_score"),
                }
            )
    encoded = json.dumps(state_projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def run_hsl_replay_benchmark(
    *, minutes: int = DEFAULT_MINUTES, symbols: int = DEFAULT_SYMBOLS, iterations: int = DEFAULT_ITERATIONS
) -> dict[str, Any]:
    if getattr(pbr, "__is_stub__", False):
        raise RuntimeError("passivbot_rust extension is required for hsl-replay-benchmark")
    if not 1 <= int(iterations) <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")

    history = build_coin_hsl_history_fixture(minutes=minutes, symbols=symbols)
    timing_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    state_digests: list[str] = []
    side_effect_totals: dict[str, int] = defaultdict(int)
    full_replay_elapsed_ns = 0
    for _ in range(int(iterations)):
        bot = _make_offline_replay_bot(history, timing_totals)
        started_ns = time.perf_counter_ns()
        previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            await bot._equity_hard_stop_initialize_coin_from_history()
        finally:
            logging.disable(previous_logging_disable)
        full_replay_elapsed_ns += time.perf_counter_ns() - started_ns
        state_digests.append(_state_digest(bot))
        for key, value in bot._offline_hsl_benchmark_side_effects.items():
            side_effect_totals[key] += int(value)
    if len(set(state_digests)) != 1:
        raise RuntimeError("offline coin-HSL replay produced non-deterministic final state")

    expected_replay_samples = int(minutes) * int(symbols) * int(iterations)
    expected_current_samples = int(symbols) * int(iterations)
    actual_sample_calls = int(timing_totals["coin_metrics_sample"]["calls"])
    elapsed_seconds = max(full_replay_elapsed_ns, 1) / 1_000_000_000.0
    replay_rows = int(minutes) * int(iterations)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "hsl_replay_benchmark",
        "offline": True,
        "fixture": {
            "minutes": int(minutes),
            "symbols": int(symbols),
            "timeline_rows": len(history["timeline"]),
            "fill_events": len(history["fill_events"]),
            "panic_events": len(history["panic_flatten_events"]),
            "sha256": _fixture_digest(history),
        },
        "counters": {
            "iterations": int(iterations),
            "active_pairs": int(symbols) * int(iterations),
            "expected_replay_samples": expected_replay_samples,
            "expected_current_samples": expected_current_samples,
            "coin_metrics_sample_calls": actual_sample_calls,
            "replay_samples_applied": actual_sample_calls - expected_current_samples,
        },
        "timings": {
            stage: {"calls": int(values["calls"]), "elapsed_ns": int(values["elapsed_ns"])}
            for stage, values in sorted(timing_totals.items())
        }
        | {
            "full_replay": {
                "calls": int(iterations),
                "elapsed_ns": int(full_replay_elapsed_ns),
            }
        },
        "throughput": {
            "timeline_rows_per_second": replay_rows / elapsed_seconds,
            "pair_rows_per_second": expected_replay_samples / elapsed_seconds,
        },
        "determinism": {"final_state_sha256": state_digests[0]},
        "side_effects": dict(sorted(side_effect_totals.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded in-memory benchmark of the coin-HSL replay hot path. "
            "It never contacts exchanges or reads/writes live cache or state."
        )
    )
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES, help=f"Fixture minutes (1-{MAX_MINUTES}).")
    parser.add_argument("--symbols", type=int, default=DEFAULT_SYMBOLS, help=f"Fixture symbols (1-{MAX_SYMBOLS}).")
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS, help=f"Replay iterations (1-{MAX_ITERATIONS})."
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact single-line JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(
            run_hsl_replay_benchmark(
                minutes=int(args.minutes),
                symbols=int(args.symbols),
                iterations=int(args.iterations),
            )
        )
    except ValueError as exc:
        build_parser().error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
