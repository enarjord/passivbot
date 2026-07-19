from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
import tracemalloc
from collections import defaultdict
from typing import Any

import passivbot_rust as pbr

from passivbot import Passivbot


BENCHMARK_SCHEMA_VERSION = 3
DEFAULT_MINUTES = 240
DEFAULT_SYMBOLS = 8
DEFAULT_ITERATIONS = 1
MAX_MINUTES = 1_440
MAX_SYMBOLS = 32
MAX_ITERATIONS = 20
LOCAL_SCALE_MAX_MINUTES = 43_201
LOCAL_SCALE_MAX_SYMBOLS = 30
BACKGROUND_YIELD_ROWS = 100
FIXTURE_EMA_SPAN_MINUTES = 7.0
FIXTURE_RED_THRESHOLD = 0.12
FIXTURE_BALANCE = 1_000.0
FIXTURE_ACCOUNT_DRIVER_INDEX = 2
DENSE_REFERENCE_HISTORY_FORMAT = "timeline"
DEFAULT_HISTORY_FORMAT = "compact"
REFERENCE_SAMPLE_COUNT_KEYS = (
    "coin_metrics_sample_calls",
    "replay_samples_applied",
    "held_replay_samples",
    "background_replay_samples",
)
TIMING_STAGE_NAMES = (
    "fixture_construction",
    "history_load",
    "cache_reuse_skipped",
    "coin_metrics_sample",
    "held_coin_metrics_sample",
    "background_coin_metrics_sample",
    "current_upnl",
    "cache_persist_skipped",
    "final_state_projection",
    "full_replay",
)
FULL_REPLAY_LEAF_STAGE_NAMES = (
    "history_load",
    "cache_reuse_skipped",
    "held_coin_metrics_sample",
    "background_coin_metrics_sample",
    "current_upnl",
    "cache_persist_skipped",
)


def _fixture_symbol(index: int) -> str:
    return f"HSLBENCH{index:02d}/USDT:USDT"


def _new_timing_totals() -> dict[str, dict[str, int]]:
    return {
        stage: {"calls": 0, "elapsed_ns": 0} for stage in TIMING_STAGE_NAMES
    }


def _record_elapsed(
    timings: dict[str, dict[str, int]], stage: str, elapsed_ns: int
) -> None:
    values = timings[stage]
    values["calls"] += 1
    values["elapsed_ns"] += elapsed_ns


def _record_timing(
    timings: dict[str, dict[str, int]], stage: str, started_ns: int
) -> None:
    _record_elapsed(timings, stage, time.perf_counter_ns() - started_ns)


def _timing_value(
    timings: dict[str, dict[str, int]], stage: str
) -> dict[str, int]:
    values = timings[stage]
    return {"calls": int(values["calls"]), "elapsed_ns": int(values["elapsed_ns"])}


def _exclusive_elapsed_profile(
    *, total_elapsed_ns: int, stages: dict[str, dict[str, int]], scope: str
) -> dict[str, Any]:
    accounted_elapsed_ns = sum(values["elapsed_ns"] for values in stages.values())
    if accounted_elapsed_ns > total_elapsed_ns:
        raise RuntimeError(
            f"offline HSL benchmark {scope} stages exceed total elapsed time"
        )
    residual_elapsed_ns = total_elapsed_ns - accounted_elapsed_ns

    def percent_of_total(elapsed_ns: int) -> float:
        if total_elapsed_ns <= 0:
            return 0.0
        return min(100.0, max(0.0, elapsed_ns / total_elapsed_ns * 100.0))

    percent_key = f"percent_of_{scope}"
    return {
        "elapsed_ns": total_elapsed_ns,
        "accounted_elapsed_ns": accounted_elapsed_ns,
        f"accounted_{percent_key}": percent_of_total(accounted_elapsed_ns),
        "stages": {
            stage: {**values, percent_key: percent_of_total(values["elapsed_ns"])}
            for stage, values in stages.items()
        },
        "residual_orchestration": {
            "elapsed_ns": residual_elapsed_ns,
            percent_key: percent_of_total(residual_elapsed_ns),
        },
    }


def _stage_profile(timings: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Return exclusive replay timing shares plus uninstrumented orchestration."""
    full_replay = _timing_value(timings, "full_replay")
    stages = {
        stage: _timing_value(timings, stage) for stage in FULL_REPLAY_LEAF_STAGE_NAMES
    }

    return {
        "taxonomy": "exclusive_full_replay_leaf_stages_v1",
        "full_replay": {
            "calls": full_replay["calls"],
            **_exclusive_elapsed_profile(
                total_elapsed_ns=full_replay["elapsed_ns"],
                stages=stages,
                scope="full_replay",
            ),
        },
        "outside_full_replay": {
            stage: _timing_value(timings, stage)
            for stage in ("fixture_construction", "final_state_projection")
        },
    }


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
    held_symbols = 1 if held_symbols is None else int(held_symbols)
    if not 1 <= minutes <= max_minutes:
        raise ValueError(f"minutes must be between 1 and {max_minutes}")
    if not 1 <= symbols <= max_symbols:
        raise ValueError(f"symbols must be between 1 and {max_symbols}")
    if not 0 <= held_symbols <= symbols:
        raise ValueError(f"held_symbols must be between 0 and {symbols}")
    return minutes, symbols, held_symbols


def _fixture_episodes(minutes: int, symbols: int) -> list[dict[str, Any]]:
    """Return fixed historical episodes that fit within the requested horizon."""
    templates = (
        (1, 601, 613, "historical_a", False),
        (1, 12_001, 12_013, "historical_b", False),
        (1, 30_001, 30_014, "historical_c", False),
        (FIXTURE_ACCOUNT_DRIVER_INDEX, 18_001, 18_006, "balance_driver", False),
        (3, 36_001, 36_015, "panic_flatten", True),
    )
    episodes = [
        {
            "symbol_index": symbol_index,
            "start_minute": start_minute,
            "end_minute": end_minute,
            "name": name,
            "panic_flatten": panic_flatten,
        }
        for symbol_index, start_minute, end_minute, name, panic_flatten in templates
        if symbol_index < symbols and end_minute <= minutes
    ]
    for symbol_index in range(4, symbols):
        start_minute = 1_001 + (symbol_index - 4) * 1_200
        end_minute = start_minute + 8
        if end_minute > minutes:
            break
        episodes.append(
            {
                "symbol_index": symbol_index,
                "start_minute": start_minute,
                "end_minute": end_minute,
                "name": f"background_{symbol_index:02d}",
                "panic_flatten": False,
            }
        )
    return episodes


def _build_fixture_contract(minutes: int, symbols: int, held_symbols: int) -> dict[str, Any]:
    """Build the scenario contract shared by dense fixture encodings.

    Pair 0 remains held at the end of the replay. Pair 1 has short historical
    episodes separated by multi-thousand-minute flat gaps. Pair 2 is an account
    balance driver, and pair 3 ends in a historical panic flatten when present.
    """
    names = [_fixture_symbol(index) for index in range(symbols)]
    episodes = _fixture_episodes(minutes, symbols)
    fill_events: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(names[:held_symbols]):
        fill_events.append(
            {
                "timestamp": 1,
                "symbol": symbol,
                "pside": "long",
                "action": "increase",
                "qty": 1.0,
                "id": f"fixture-current-open-{symbol_index}",
            }
        )
    panic_flatten_events: list[dict[str, Any]] = []
    for episode in episodes:
        symbol = names[episode["symbol_index"]]
        start_ts = int(episode["start_minute"]) * 60_000 + 100
        end_ts = int(episode["end_minute"]) * 60_000 + 200
        if episode["name"] == "historical_a":
            # Deliberately preserve source order at one exchange timestamp.
            fill_events.extend(
                (
                    {
                        "timestamp": start_ts,
                        "symbol": symbol,
                        "pside": "long",
                        "action": "increase",
                        "qty": 0.4,
                        "id": "fixture-historical-a-open-first",
                    },
                    {
                        "timestamp": start_ts,
                        "symbol": symbol,
                        "pside": "long",
                        "action": "increase",
                        "qty": 0.6,
                        "id": "fixture-historical-a-open-second",
                    },
                )
            )
        else:
            fill_events.append(
                {
                    "timestamp": start_ts,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "id": f"fixture-{episode['name']}-open",
                }
            )
        fill_events.append(
            {
                "timestamp": end_ts,
                "symbol": symbol,
                "pside": "long",
                "action": "decrease",
                "qty": 1.0,
                "id": f"fixture-{episode['name']}-close",
            }
        )
        if episode["panic_flatten"]:
            panic_flatten_events.append(
                {
                    "timestamp": end_ts,
                    "minute_timestamp": int(episode["end_minute"]) * 60_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            )
    return {
        "kind": "coin_hsl_dense_reference_fixture",
        "minutes": minutes,
        "symbols": names,
        "held_symbols": names[:held_symbols],
        "ema_span_minutes": FIXTURE_EMA_SPAN_MINUTES,
        "red_threshold": FIXTURE_RED_THRESHOLD,
        "account_balance_driver_symbol": (
            names[FIXTURE_ACCOUNT_DRIVER_INDEX]
            if FIXTURE_ACCOUNT_DRIVER_INDEX < symbols
            else None
        ),
        "episodes": episodes,
        "same_timestamp_fill_order": [
            "fixture-historical-a-open-first",
            "fixture-historical-a-open-second",
        ]
        if any(episode["name"] == "historical_a" for episode in episodes)
        else [],
        "fill_events": fill_events,
        "panic_flatten_events": panic_flatten_events,
        "pair_values_are_dense": True,
    }


def _build_dense_series(
    minutes: int, symbols: int, contract: dict[str, Any]
) -> tuple[Any, Any, Any, dict[tuple[str, str], dict[str, Any]]]:
    """Create dense account and per-pair arrays from the scenario contract."""
    import numpy as np

    minute_numbers = np.arange(1, minutes + 1, dtype=np.int64)
    timestamps = minute_numbers * 60_000
    balances = np.full(minutes, FIXTURE_BALANCE, dtype=np.float64)
    pair_values: dict[tuple[str, str], dict[str, Any]] = {}
    episodes_by_symbol: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in contract["episodes"]:
        episodes_by_symbol[int(episode["symbol_index"])].append(episode)
    for symbol_index, symbol in enumerate(contract["symbols"]):
        symbol_episodes = episodes_by_symbol[symbol_index]
        if symbol in contract["held_symbols"]:
            realized = np.zeros(minutes, dtype=np.float64)
            unrealized = -(
                ((minute_numbers + symbol_index) % 17).astype(np.float64) + 1.0
            ) / 100.0
        elif symbol_episodes:
            first_start_minute = min(
                int(episode["start_minute"]) for episode in symbol_episodes
            )
            realized = np.full(minutes, np.nan, dtype=np.float64)
            unrealized = np.full(minutes, np.nan, dtype=np.float64)
            covered = minute_numbers >= first_start_minute
            realized[covered] = 0.0
            unrealized[covered] = 0.0
        else:
            realized = np.full(minutes, np.nan, dtype=np.float64)
            unrealized = np.full(minutes, np.nan, dtype=np.float64)
        for episode_index, episode in enumerate(symbol_episodes, start=1):
            start_minute = int(episode["start_minute"])
            end_minute = int(episode["end_minute"])
            active = (minute_numbers >= start_minute) & (minute_numbers <= end_minute)
            progress = (minute_numbers[active] - start_minute + 1) / (
                end_minute - start_minute + 1
            )
            loss = 0.02 if episode["name"] == "balance_driver" else 0.24
            unrealized[active] = -loss * FIXTURE_BALANCE * progress
            realized[minute_numbers >= end_minute] += (
                24.0 if episode["name"] == "balance_driver" else 3.0 * episode_index
            )
            if episode["name"] == "balance_driver":
                balances[minute_numbers >= end_minute] += 24.0
        pair_values[("long", symbol)] = {
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
        }
    realized_total = np.zeros(minutes, dtype=np.float64)
    for values in pair_values.values():
        realized_total += np.nan_to_num(values["realized_pnl"], nan=0.0)
    return timestamps, balances, realized_total, pair_values


def build_coin_hsl_history_fixture(
    minutes: int,
    symbols: int,
    held_symbols: int | None = None,
    *,
    local_scale: bool = False,
) -> dict[str, Any]:
    """Build the dense-reference timeline fixture for coin-HSL replay."""
    minutes, symbols, held_symbols = _validate_fixture_shape(
        minutes, symbols, held_symbols, local_scale=local_scale
    )
    contract = _build_fixture_contract(minutes, symbols, held_symbols)
    timestamps, balances, realized_total, pair_values = _build_dense_series(
        minutes, symbols, contract
    )
    timeline: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        timeline.append(
            {
                "timestamp": int(timestamp),
                "balance": float(balances[index]),
                "realized_pnl": float(realized_total[index]),
                "realized_pnl_by_coin_pside": {
                    symbol: {"long": float(values["realized_pnl"][index])}
                    for (_pside, symbol), values in pair_values.items()
                    if math.isfinite(float(values["realized_pnl"][index]))
                },
                "unrealized_pnl_by_coin_pside": {
                    symbol: {"long": float(values["unrealized_pnl"][index])}
                    for (_pside, symbol), values in pair_values.items()
                    if math.isfinite(float(values["unrealized_pnl"][index]))
                },
            }
        )
    return {
        "timeline": timeline,
        "fill_events": contract["fill_events"],
        "panic_flatten_events": contract["panic_flatten_events"],
        "fixture_contract": contract,
    }


def build_coin_hsl_compact_fixture(
    minutes: int,
    symbols: int,
    held_symbols: int | None = None,
    *,
    local_scale: bool = False,
) -> dict[str, Any]:
    """Build the dense-reference fixture in compact replay form."""
    import numpy as np

    minutes, symbols, held_symbols = _validate_fixture_shape(
        minutes, symbols, held_symbols, local_scale=local_scale
    )
    contract = _build_fixture_contract(minutes, symbols, held_symbols)
    timestamps, balances, realized_total, pair_values = _build_dense_series(
        minutes, symbols, contract
    )
    pair_values = {
        pair: values
        for pair, values in pair_values.items()
        if bool(np.any(np.isfinite(values["realized_pnl"])))
        or bool(np.any(np.isfinite(values["unrealized_pnl"])))
    }
    return {
        "hsl_coin_compact_replay": {
            "timestamps": timestamps,
            "balances": balances,
            "realized_pnl": realized_total,
            "pair_values": pair_values,
        },
        "fill_events": contract["fill_events"],
        "panic_flatten_events": contract["panic_flatten_events"],
        "fixture_contract": contract,
    }


def _fixture_digest(history: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    contract = history["fixture_contract"]
    digest.update(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    compact = history.get("hsl_coin_compact_replay")
    if compact is not None:
        for field in ("timestamps", "balances", "realized_pnl"):
            digest.update(field.encode("utf-8"))
            digest.update(compact[field].tobytes(order="C"))
        for pair in sorted(compact["pair_values"]):
            digest.update(f"{pair[0]}\0{pair[1]}".encode("utf-8"))
            values = compact["pair_values"][pair]
            digest.update(values["realized_pnl"].tobytes(order="C"))
            digest.update(values["unrealized_pnl"].tobytes(order="C"))
        return digest.hexdigest()
    for row in history["timeline"]:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _fixture_contract_digest(history: dict[str, Any]) -> str:
    encoded = json.dumps(
        history["fixture_contract"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
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
            "red_threshold": FIXTURE_RED_THRESHOLD,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": FIXTURE_EMA_SPAN_MINUTES,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 1.0,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
        "short": {
            "enabled": False,
            "red_threshold": FIXTURE_RED_THRESHOLD,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": FIXTURE_EMA_SPAN_MINUTES,
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
            _record_timing(timings, "history_load", started_ns)

    async def no_cache_reuse(*_args, **_kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return None
        finally:
            _record_timing(timings, "cache_reuse_skipped", started_ns)

    async def current_upnl(*_args, **_kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return 0.0
        finally:
            _record_timing(timings, "current_upnl", started_ns)

    def skip_cache_persist(*_args, **_kwargs):
        started_ns = time.perf_counter_ns()
        try:
            return 0
        finally:
            _record_timing(timings, "cache_persist_skipped", started_ns)

    def skip_latch_write(*_args, **_kwargs):
        side_effects["latch_writes"] += 1
        return None

    def skip_latch_remove(*_args, **_kwargs):
        side_effects["latch_removals"] += 1

    original_apply_metrics = bot._equity_hard_stop_apply_coin_metrics_sample

    def profile_coin_metrics_sample(*args, **kwargs):
        started_ns = time.perf_counter_ns()
        symbol = args[1] if len(args) > 1 else kwargs["symbol"]
        sample_stage = (
            "held_coin_metrics_sample"
            if symbol in held_symbols
            else "background_coin_metrics_sample"
        )
        try:
            return original_apply_metrics(*args, **kwargs)
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            _record_elapsed(timings, "coin_metrics_sample", elapsed_ns)
            _record_elapsed(timings, sample_stage, elapsed_ns)
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


def _state_projection(bot: Passivbot) -> list[dict[str, Any]]:
    def stop_event_projection(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {key: item for key, item in value.items() if key != "triggered_at"}

    state_projection = []
    for pside, symbols in sorted(bot._equity_hard_stop_coin.items()):
        for symbol, state in sorted(symbols.items()):
            metrics = state.get("last_metrics") or {}
            runtime = state["runtime"]
            state_projection.append(
                {
                    "pside": pside,
                    "symbol": symbol,
                    "halted": bool(state["halted"]),
                    "no_restart_latched": bool(state["no_restart_latched"]),
                    "cooldown_until_ms": state["cooldown_until_ms"],
                    "pnl_reset_timestamp_ms": state["pnl_reset_timestamp_ms"],
                    "cooldown_intervention_active": bool(
                        state["cooldown_intervention_active"]
                    ),
                    "cooldown_repanic_reset_pending": bool(
                        state["cooldown_repanic_reset_pending"]
                    ),
                    "cooldown_unresolved_residue": bool(
                        state["cooldown_unresolved_residue"]
                    ),
                    "pending_red_since_ms": state["pending_red_since_ms"],
                    "red_flat_confirmations": int(state["red_flat_confirmations"]),
                    "red_trigger_event_emitted": bool(
                        state["red_trigger_event_emitted"]
                    ),
                    "no_restart_peak_strategy_equity": float(
                        state["no_restart_peak_strategy_equity"]
                    ),
                    "pending_stop_event": stop_event_projection(
                        state["pending_stop_event"]
                    ),
                    "last_stop_event": stop_event_projection(state["last_stop_event"]),
                    "runtime": {
                        "initialized": bool(runtime.initialized()),
                        "red_latched": bool(runtime.red_latched()),
                        "red_seen_in_episode": bool(runtime.red_seen_in_episode()),
                        "tier": str(runtime.tier()),
                        "drawdown_ema": float(runtime.drawdown_ema()),
                        "peak_strategy_equity": float(runtime.peak_strategy_equity()),
                        "rolling_peak_strategy_equity": float(
                            runtime.rolling_peak_strategy_equity()
                        ),
                    },
                    "last_metrics": {
                        key: metrics.get(key)
                        for key in (
                            "timestamp_ms",
                            "tier",
                            "red_active_now",
                            "red_seen_in_episode",
                            "balance",
                            "slot_budget",
                            "peak_realized_pnl",
                            "realized_pnl",
                            "unrealized_pnl",
                            "drawdown_raw",
                            "drawdown_ema",
                            "drawdown_score",
                        )
                    },
                }
            )
    return state_projection


def _state_digest_from_projection(state_projection: list[dict[str, Any]]) -> str:
    encoded = json.dumps(state_projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_digest(bot: Passivbot) -> str:
    return _state_digest_from_projection(_state_projection(bot))


def _sample_count_projection(report: dict[str, Any]) -> dict[str, int]:
    return {key: int(report["counters"][key]) for key in REFERENCE_SAMPLE_COUNT_KEYS}


def compare_dense_reference_output(
    dense_reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare a replay candidate with the dense fixture reference output.

    A future sparse per-pair consumer can call this with its own report. Timing is
    deliberately excluded: fixture identity, non-increasing sample counts, and
    final reconstructed per-pair state define equivalence.
    """
    reference_samples = _sample_count_projection(dense_reference)
    candidate_samples = _sample_count_projection(candidate)
    reference_state = dense_reference["output_state"]
    candidate_state = candidate["output_state"]
    fixture_matches = (
        dense_reference["fixture"]["scenario_sha256"]
        == candidate["fixture"]["scenario_sha256"]
    )
    sample_reduction = {
        key: reference_samples[key] - candidate_samples[key]
        for key in REFERENCE_SAMPLE_COUNT_KEYS
    }
    samples_match = all(reduction >= 0 for reduction in sample_reduction.values())
    state_matches = reference_state == candidate_state
    return {
        "fixture_scenario_matches": fixture_matches,
        "sample_counts": {
            "matches": samples_match,
            "dense_reference": reference_samples,
            "candidate": candidate_samples,
            "reduction": sample_reduction,
        },
        "output_state": {
            "matches": state_matches,
            "dense_reference_sha256": reference_state["sha256"],
            "candidate_sha256": candidate_state["sha256"],
            "dense_reference": reference_state["pairs"],
            "candidate": candidate_state["pairs"],
        },
        "matches": fixture_matches and samples_match and state_matches,
    }


async def _run_hsl_replay_benchmark(
    *,
    minutes: int = DEFAULT_MINUTES,
    symbols: int = DEFAULT_SYMBOLS,
    held_symbols: int | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    local_scale: bool = False,
    history_format: str = DEFAULT_HISTORY_FORMAT,
) -> dict[str, Any]:
    if getattr(pbr, "__is_stub__", False):
        raise RuntimeError("passivbot_rust extension is required for hsl-replay-benchmark")
    if not 1 <= int(iterations) <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")

    run_started_ns = time.perf_counter_ns()
    minutes, symbols, held_symbols = _validate_fixture_shape(
        minutes, symbols, held_symbols, local_scale=local_scale
    )
    timing_totals = _new_timing_totals()
    fixture_started_ns = time.perf_counter_ns()
    try:
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
    finally:
        _record_timing(timing_totals, "fixture_construction", fixture_started_ns)
    held_names = {_fixture_symbol(index) for index in range(held_symbols)}
    active_names = set(held_names)
    active_names.update(
        str(event["symbol"])
        for event in history["fill_events"]
        if event.get("symbol")
    )
    active_symbols = len(active_names)
    sample_counts: dict[str, int] = defaultdict(int)
    state_digests: list[str] = []
    state_projections: list[list[dict[str, Any]]] = []
    side_effect_totals: dict[str, int] = defaultdict(int)
    for _ in range(int(iterations)):
        bot = _make_offline_replay_bot(history, timing_totals, held_names, sample_counts)
        started_ns = time.perf_counter_ns()
        previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            await bot._equity_hard_stop_initialize_coin_from_history()
        finally:
            logging.disable(previous_logging_disable)
            _record_timing(timing_totals, "full_replay", started_ns)
        projection_started_ns = time.perf_counter_ns()
        try:
            state_projection = _state_projection(bot)
        finally:
            _record_timing(timing_totals, "final_state_projection", projection_started_ns)
        state_projections.append(state_projection)
        state_digests.append(_state_digest_from_projection(state_projection))
        for key, value in bot._offline_hsl_benchmark_side_effects.items():
            side_effect_totals[key] += int(value)
    if len(set(state_digests)) != 1 or len(
        {
            json.dumps(projection, sort_keys=True, separators=(",", ":"))
            for projection in state_projections
        }
    ) != 1:
        raise RuntimeError("offline coin-HSL replay produced non-deterministic final state")

    expected_replay_samples = minutes * active_symbols * int(iterations)
    expected_current_samples = active_symbols * int(iterations)
    expected_held_samples = minutes * held_symbols * int(iterations)
    background_symbols = active_symbols - held_symbols
    expected_background_samples = minutes * background_symbols * int(iterations)
    expected_background_yields = (
        (minutes // BACKGROUND_YIELD_ROWS) * background_symbols * int(iterations)
    )
    held_current_samples = held_symbols * int(iterations)
    background_current_samples = background_symbols * int(iterations)
    actual_sample_calls = int(timing_totals["coin_metrics_sample"]["calls"])
    full_replay_elapsed_ns = int(timing_totals["full_replay"]["elapsed_ns"])
    elapsed_seconds = max(full_replay_elapsed_ns, 1) / 1_000_000_000.0
    replay_rows = minutes * int(iterations)
    stage_profile = _stage_profile(timing_totals)
    report = {
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
            "scenario_sha256": _fixture_contract_digest(history),
            "sha256": _fixture_digest(history),
            "contract": {
                "ema_span_minutes": history["fixture_contract"]["ema_span_minutes"],
                "red_threshold": history["fixture_contract"]["red_threshold"],
                "account_balance_driver_symbol": history["fixture_contract"][
                    "account_balance_driver_symbol"
                ],
                "historical_episodes": len(history["fixture_contract"]["episodes"]),
                "same_timestamp_fill_order": history["fixture_contract"][
                    "same_timestamp_fill_order"
                ],
                "pair_values_are_dense": history["fixture_contract"]["pair_values_are_dense"],
            },
        },
        "counters": {
            "iterations": int(iterations),
            "active_pairs": active_symbols * int(iterations),
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
            stage: _timing_value(timing_totals, stage) for stage in TIMING_STAGE_NAMES
        },
        "stage_profile": stage_profile,
        "throughput": {
            "timeline_rows_per_second": replay_rows / elapsed_seconds,
            "pair_rows_per_second": expected_replay_samples / elapsed_seconds,
            "applied_pair_samples_per_second": actual_sample_calls / elapsed_seconds,
        },
        "determinism": {"final_state_sha256": state_digests[0]},
        "output_state": {
            "sha256": state_digests[0],
            "pairs": state_projections[0],
        },
        "side_effects": dict(sorted(side_effect_totals.items())),
    }
    run_elapsed_ns = time.perf_counter_ns() - run_started_ns
    run_stages = {
        stage: _timing_value(timing_totals, stage)
        for stage in ("fixture_construction", "full_replay", "final_state_projection")
    }
    stage_profile["run"] = {
        "calls": 1,
        **_exclusive_elapsed_profile(
            total_elapsed_ns=run_elapsed_ns,
            stages=run_stages,
            scope="run",
        ),
    }
    return report


async def run_hsl_replay_benchmark(
    *,
    minutes: int = DEFAULT_MINUTES,
    symbols: int = DEFAULT_SYMBOLS,
    held_symbols: int | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    local_scale: bool = False,
    profile_memory: bool = False,
    history_format: str = DEFAULT_HISTORY_FORMAT,
) -> dict[str, Any]:
    """Run the benchmark, optionally recording Python allocation peak usage."""
    if getattr(pbr, "__is_stub__", False):
        raise RuntimeError("passivbot_rust extension is required for hsl-replay-benchmark")
    if not 1 <= int(iterations) <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
    _validate_fixture_shape(minutes, symbols, held_symbols, local_scale=local_scale)

    pipeline_started_ns = time.perf_counter_ns()
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
        reference_is_candidate = history_format == DENSE_REFERENCE_HISTORY_FORMAT
        reference_report = report
        if not reference_is_candidate:
            reference_report = await _run_hsl_replay_benchmark(
                minutes=minutes,
                symbols=symbols,
                held_symbols=held_symbols,
                iterations=iterations,
                local_scale=local_scale,
                history_format=DENSE_REFERENCE_HISTORY_FORMAT,
            )
        dense_reference = {
            "history_format": DENSE_REFERENCE_HISTORY_FORMAT,
            "same_run_as_candidate": reference_is_candidate,
            "fixture_scenario_sha256": reference_report["fixture"]["scenario_sha256"],
            "sample_counts": _sample_count_projection(reference_report),
            "output_state": reference_report["output_state"],
        }
        if not reference_is_candidate:
            dense_reference.update(
                {
                    "fixture": reference_report["fixture"],
                    "timings": reference_report["timings"],
                    "stage_profile": reference_report["stage_profile"],
                    "throughput": reference_report["throughput"],
                }
            )
        report["dense_reference"] = dense_reference
        comparison_started_ns = time.perf_counter_ns()
        report["equivalence"] = compare_dense_reference_output(reference_report, report)
        comparison_elapsed_ns = time.perf_counter_ns() - comparison_started_ns
        if profile_memory:
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            report["memory"] = {
                "tracemalloc": True,
                "current_bytes": int(current_bytes),
                "peak_bytes": int(peak_bytes),
            }
        pipeline_stages = {
            "candidate_run": {
                "calls": 1,
                "elapsed_ns": int(report["stage_profile"]["run"]["elapsed_ns"]),
            }
        }
        if not reference_is_candidate:
            pipeline_stages["dense_reference_run"] = {
                "calls": 1,
                "elapsed_ns": int(reference_report["stage_profile"]["run"]["elapsed_ns"]),
            }
        pipeline_stages["equivalence_comparison"] = {
            "calls": 1,
            "elapsed_ns": comparison_elapsed_ns,
        }
        pipeline_elapsed_ns = time.perf_counter_ns() - pipeline_started_ns
        report["pipeline_profile"] = {
            "taxonomy": "exclusive_benchmark_pipeline_stages_v1",
            "calls": 1,
            **_exclusive_elapsed_profile(
                total_elapsed_ns=pipeline_elapsed_ns,
                stages=pipeline_stages,
                scope="pipeline",
            ),
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
        help="Current-position symbols (default: one fixture symbol).",
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
        default=DEFAULT_HISTORY_FORMAT,
        help="Replay payload representation (default: compact dense reference).",
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
