from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
import tracemalloc
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
LOCAL_SCALE_MAX_MINUTES = 43_201
LOCAL_SCALE_MAX_SYMBOLS = 30
BACKGROUND_YIELD_ROWS = 100


def _fixture_symbol(index: int) -> str:
    return f"HSLBENCH{index:02d}/USDT:USDT"


def _validate_fixture_shape(
    minutes: int,
    symbols: int,
    held_symbols: int | None,
    *,
    local_scale: bool,
) -> tuple[int, int, int]:
    max_minutes = LOCAL_SCALE_MAX_MINUTES if local_scale else MAX_MINUTES
    max_symbols = LOCAL_SCALE_MAX_SYMBOLS if local_scale else MAX_SYMBOLS
    minutes = int(minutes)
    symbols = int(symbols)
    held_symbols = symbols if held_symbols is None else int(held_symbols)
    if not 1 <= minutes <= max_minutes:
        raise ValueError(f"minutes must be between 1 and {max_minutes}")
    if not 1 <= symbols <= max_symbols:
        raise ValueError(f"symbols must be between 1 and {max_symbols}")
    if not 0 <= held_symbols <= symbols:
        raise ValueError(f"held_symbols must be between 0 and {symbols}")
    return minutes, symbols, held_symbols


def build_coin_hsl_history_fixture(
    minutes: int,
    symbols: int,
    held_symbols: int | None = None,
    *,
    local_scale: bool = False,
) -> dict[str, Any]:
    """Build deterministic coin-HSL history with optional flat background symbols."""
    minutes, symbols, held_symbols = _validate_fixture_shape(
        minutes, symbols, held_symbols, local_scale=local_scale
    )

    names = [_fixture_symbol(index) for index in range(symbols)]
    timeline: list[dict[str, Any]] = []
    fill_events: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(names[:held_symbols]):
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
    for minute in range(1, minutes + 1):
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


def build_coin_hsl_compact_fixture(
    minutes: int,
    symbols: int,
    held_symbols: int | None = None,
    *,
    local_scale: bool = False,
) -> dict[str, Any]:
    """Build the same deterministic fixture directly in compact replay form."""
    import numpy as np

    minutes, symbols, held_symbols = _validate_fixture_shape(
        minutes, symbols, held_symbols, local_scale=local_scale
    )
    names = [_fixture_symbol(index) for index in range(symbols)]
    minute_numbers = np.arange(1, minutes + 1, dtype=np.int64)
    timestamps = minute_numbers * 60_000
    pair_values: dict[tuple[str, str], dict[str, Any]] = {}
    realized_total = np.zeros(minutes, dtype=np.float64)
    for symbol_index, symbol in enumerate(names):
        realized = ((minute_numbers * (symbol_index + 3)) % 19).astype(
            np.float64
        ) / 100.0
        unrealized = -(
            (minute_numbers + symbol_index * 5) % 11
        ).astype(np.float64) / 100.0
        pair_values[("long", symbol)] = {
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
        }
        realized_total += realized
    fill_events = [
        {
            "timestamp": 1,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "id": f"fixture-open-{symbol_index}",
        }
        for symbol_index, symbol in enumerate(names[:held_symbols])
    ]
    return {
        "hsl_coin_compact_replay": {
            "timestamps": timestamps,
            "balances": np.full(minutes, 1_000.0, dtype=np.float64),
            "realized_pnl": realized_total,
            "pair_values": pair_values,
        },
        "fill_events": fill_events,
        "panic_flatten_events": [],
    }


def _fixture_digest(history: dict[str, Any]) -> str:
    compact = history.get("hsl_coin_compact_replay")
    if compact is not None:
        digest = hashlib.sha256()
        for field in ("timestamps", "balances", "realized_pnl"):
            digest.update(field.encode("utf-8"))
            digest.update(compact[field].tobytes(order="C"))
        for pair in sorted(compact["pair_values"]):
            digest.update(f"{pair[0]}\0{pair[1]}".encode("utf-8"))
            values = compact["pair_values"][pair]
            digest.update(values["realized_pnl"].tobytes(order="C"))
            digest.update(values["unrealized_pnl"].tobytes(order="C"))
        digest.update(
            json.dumps(
                history["fill_events"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        return digest.hexdigest()
    encoded = json.dumps(history, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_offline_replay_bot(
    history: dict[str, Any],
    timings: dict[str, dict[str, int]],
    held_symbols: set[str],
    sample_counts: dict[str, int],
) -> Passivbot:
    """Create an uninitialized bot that can only replay supplied in-memory history."""
    bot = Passivbot.__new__(Passivbot)
    if "hsl_coin_compact_replay" in history:
        now_ms = int(history["hsl_coin_compact_replay"]["timestamps"][-1])
    else:
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
        for symbol in held_symbols
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
        "n_positions": len(held_symbols),
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
            symbol = args[1] if len(args) > 1 else kwargs["symbol"]
            sample_counts[
                "held_replay_samples" if symbol in held_symbols else "background_replay_samples"
            ] += 1

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


async def _run_hsl_replay_benchmark(
    *,
    minutes: int = DEFAULT_MINUTES,
    symbols: int = DEFAULT_SYMBOLS,
    held_symbols: int | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    local_scale: bool = False,
    history_format: str = "timeline",
) -> dict[str, Any]:
    if getattr(pbr, "__is_stub__", False):
        raise RuntimeError("passivbot_rust extension is required for hsl-replay-benchmark")
    if not 1 <= int(iterations) <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")

    minutes, symbols, held_symbols = _validate_fixture_shape(
        minutes, symbols, held_symbols, local_scale=local_scale
    )
    if history_format == "timeline":
        history = build_coin_hsl_history_fixture(
            minutes=minutes,
            symbols=symbols,
            held_symbols=held_symbols,
            local_scale=local_scale,
        )
    elif history_format == "compact":
        history = build_coin_hsl_compact_fixture(
            minutes=minutes,
            symbols=symbols,
            held_symbols=held_symbols,
            local_scale=local_scale,
        )
    else:
        raise ValueError(
            "history_format must be one of timeline or compact, "
            f"got {history_format!r}"
        )
    held_names = {_fixture_symbol(index) for index in range(held_symbols)}
    timing_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_counts: dict[str, int] = defaultdict(int)
    state_digests: list[str] = []
    side_effect_totals: dict[str, int] = defaultdict(int)
    full_replay_elapsed_ns = 0
    for _ in range(int(iterations)):
        bot = _make_offline_replay_bot(history, timing_totals, held_names, sample_counts)
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

    expected_replay_samples = minutes * symbols * int(iterations)
    expected_current_samples = symbols * int(iterations)
    expected_held_samples = minutes * held_symbols * int(iterations)
    background_symbols = symbols - held_symbols
    expected_background_samples = minutes * background_symbols * int(iterations)
    expected_background_yields = (
        (minutes // BACKGROUND_YIELD_ROWS) * background_symbols * int(iterations)
    )
    held_current_samples = held_symbols * int(iterations)
    background_current_samples = background_symbols * int(iterations)
    actual_sample_calls = int(timing_totals["coin_metrics_sample"]["calls"])
    elapsed_seconds = max(full_replay_elapsed_ns, 1) / 1_000_000_000.0
    replay_rows = minutes * int(iterations)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "hsl_replay_benchmark",
        "offline": True,
        "fixture": {
            "minutes": int(minutes),
            "symbols": int(symbols),
            "held_symbols": held_symbols,
            "background_symbols": background_symbols,
            "history_format": history_format,
            "timeline_rows": len(history.get("timeline") or []),
            "compact_rows": (
                len(history["hsl_coin_compact_replay"]["timestamps"])
                if "hsl_coin_compact_replay" in history
                else 0
            ),
            "fill_events": len(history["fill_events"]),
            "panic_events": len(history["panic_flatten_events"]),
            "sha256": _fixture_digest(history),
        },
        "counters": {
            "iterations": int(iterations),
            "active_pairs": symbols * int(iterations),
            "held_pairs": held_symbols * int(iterations),
            "background_pairs": background_symbols * int(iterations),
            "expected_replay_samples": expected_replay_samples,
            "expected_current_samples": expected_current_samples,
            "expected_held_samples": expected_held_samples,
            "expected_background_samples": expected_background_samples,
            "expected_background_yields": expected_background_yields,
            "coin_metrics_sample_calls": actual_sample_calls,
            "replay_samples_applied": actual_sample_calls - expected_current_samples,
            "held_replay_samples": int(sample_counts["held_replay_samples"]) - held_current_samples,
            "background_replay_samples": (
                int(sample_counts["background_replay_samples"]) - background_current_samples
            ),
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


async def run_hsl_replay_benchmark(
    *,
    minutes: int = DEFAULT_MINUTES,
    symbols: int = DEFAULT_SYMBOLS,
    held_symbols: int | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    local_scale: bool = False,
    profile_memory: bool = False,
    history_format: str = "timeline",
) -> dict[str, Any]:
    """Run the benchmark, optionally recording Python allocation peak usage."""
    if getattr(pbr, "__is_stub__", False):
        raise RuntimeError("passivbot_rust extension is required for hsl-replay-benchmark")
    if not 1 <= int(iterations) <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
    _validate_fixture_shape(minutes, symbols, held_symbols, local_scale=local_scale)

    tracing_started_here = False
    if profile_memory:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            tracing_started_here = True
        tracemalloc.reset_peak()
    try:
        report = await _run_hsl_replay_benchmark(
            minutes=minutes,
            symbols=symbols,
            held_symbols=held_symbols,
            iterations=iterations,
            local_scale=local_scale,
            history_format=history_format,
        )
        if profile_memory:
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            report["memory"] = {
                "tracemalloc": True,
                "current_bytes": int(current_bytes),
                "peak_bytes": int(peak_bytes),
            }
        return report
    finally:
        if tracing_started_here:
            tracemalloc.stop()


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
        "--held-symbols",
        type=int,
        default=None,
        help="Current-position symbols (default: all fixture symbols).",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS, help=f"Replay iterations (1-{MAX_ITERATIONS})."
    )
    parser.add_argument(
        "--local-scale",
        action="store_true",
        help=f"Allow local-scale limits ({LOCAL_SCALE_MAX_MINUTES} minutes, {LOCAL_SCALE_MAX_SYMBOLS} symbols).",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record Python allocation current and peak bytes with tracemalloc.",
    )
    parser.add_argument(
        "--history-format",
        choices=("timeline", "compact"),
        default="timeline",
        help="Replay payload representation (default: timeline).",
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
                held_symbols=args.held_symbols,
                iterations=int(args.iterations),
                local_scale=bool(args.local_scale),
                profile_memory=bool(args.profile_memory),
                history_format=str(args.history_format),
            )
        )
    except ValueError as exc:
        build_parser().error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
