from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from candlestick_manager import CANDLE_DTYPE
from config import load_prepared_config
from exchanges.fake import FakeCCXTClient, load_fake_scenario
from fill_events_manager import FillEvent, FillEventCache
from logging_setup import configure_logging
import passivbot as passivbot_mod
from passivbot import setup_bot, shutdown_bot
from procedures import ensure_parent_directory


def _build_output_dir(root: str | None, scenario: dict) -> Path:
    base = Path(root) if root else Path("artifacts") / "fake_live"
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenario_name = str(
        scenario.get("name")
        or Path(str(scenario.get("_scenario_path", "scenario"))).stem
    )
    return base / f"{stamp}_{scenario_name}"


def _mode_run_user(user: str | None, mode: str) -> str | None:
    if user is None:
        return None
    return f"{user}_{mode}"


def _dump_json(path: Path, data: Any) -> None:
    ensure_parent_directory(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _summarize_remote_calls(call_log: List[dict]) -> dict:
    by_method: Dict[str, int] = {}
    by_step: Dict[str, Dict[str, int]] = {}
    ohlcv_calls: List[dict] = []
    for entry in call_log:
        method = str(entry.get("method") or "unknown")
        step_key = str(entry.get("step_index") if entry.get("step_index") is not None else "unknown")
        by_method[method] = by_method.get(method, 0) + 1
        step_bucket = by_step.setdefault(step_key, {})
        step_bucket[method] = step_bucket.get(method, 0) + 1
        if method == "fetch_ohlcv":
            ohlcv_calls.append(
                {
                    "step_index": entry.get("step_index"),
                    "symbol": entry.get("symbol"),
                    "timeframe": entry.get("timeframe"),
                    "since": entry.get("since"),
                    "until": entry.get("until"),
                    "limit": entry.get("limit"),
                    "rows": entry.get("rows"),
                }
            )
    return {
        "total_calls": len(call_log),
        "by_method": dict(sorted(by_method.items())),
        "by_step": {key: dict(sorted(value.items())) for key, value in sorted(by_step.items())},
        "ohlcv_calls": ohlcv_calls,
    }


def _install_candle_remote_fetch_trace(bot) -> tuple[List[dict], callable]:
    if not hasattr(bot, "cm"):
        return [], lambda: None
    existing_cb = getattr(bot.cm, "_remote_fetch_callback", None)
    events: List[dict] = []

    def traced(payload: Dict[str, Any]) -> None:
        item = dict(payload)
        item["event_index"] = len(events)
        events.append(item)
        if existing_cb is not None:
            existing_cb(payload)

    bot.cm._remote_fetch_callback = traced
    return events, lambda: setattr(bot.cm, "_remote_fetch_callback", existing_cb)


def _attach_file_logging(path: Path) -> logging.Handler:
    ensure_parent_directory(path)
    root = logging.getLogger()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    if root.handlers and root.handlers[0].formatter is not None:
        handler.setFormatter(root.handlers[0].formatter)
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    return handler


def _extract_hsl_trace(bot) -> Dict[str, dict]:
    trace: Dict[str, dict] = {}
    for pside in ("long", "short"):
        if not hasattr(bot, "_hsl_state"):
            break
        state = bot._hsl_state(pside)
        trace[pside] = {
            "halted": bool(state.get("halted", False)),
            "no_restart_latched": bool(state.get("no_restart_latched", False)),
            "cooldown_until_ms": state.get("cooldown_until_ms"),
            "pending_red_since_ms": state.get("pending_red_since_ms"),
            "red_flat_confirmations": state.get("red_flat_confirmations"),
            "cooldown_intervention_active": bool(state.get("cooldown_intervention_active", False)),
            "cooldown_repanic_reset_pending": bool(
                state.get("cooldown_repanic_reset_pending", False)
            ),
            "last_metrics": state.get("last_metrics"),
            "last_stop_event": state.get("last_stop_event"),
        }
    return trace


def _coerce_numeric_assertion(spec: Any) -> Dict[str, float]:
    if isinstance(spec, (int, float)):
        return {"eq": float(spec)}
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported numeric assertion spec: {spec!r}")
    result: Dict[str, float] = {}
    for key in ("eq", "min", "max", "approx", "tolerance"):
        if key in spec:
            result[key] = float(spec[key])
    return result


def _assert_numeric(name: str, actual: float, spec: Any) -> None:
    parsed = _coerce_numeric_assertion(spec)
    if "eq" in parsed and actual != parsed["eq"]:
        raise AssertionError(f"{name}: expected {parsed['eq']} got {actual}")
    if "min" in parsed and actual < parsed["min"]:
        raise AssertionError(f"{name}: expected >= {parsed['min']} got {actual}")
    if "max" in parsed and actual > parsed["max"]:
        raise AssertionError(f"{name}: expected <= {parsed['max']} got {actual}")
    if "approx" in parsed:
        tolerance = parsed.get("tolerance", 1e-9)
        if abs(actual - parsed["approx"]) > tolerance:
            raise AssertionError(
                f"{name}: expected {parsed['approx']} +/- {tolerance} got {actual}"
            )


def _assert_value(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, dict) and any(
        key in expected for key in ("eq", "min", "max", "approx", "tolerance")
    ):
        _assert_numeric(name, float(actual), expected)
        return
    if isinstance(expected, dict) and "contains" in expected:
        needle = str(expected["contains"])
        if needle not in str(actual):
            raise AssertionError(f"{name}: expected to contain {needle!r}, got {actual!r}")
        return
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r} got {actual!r}")


def _get_path_value(root: Any, path: str) -> Any:
    current = root
    for segment in [part for part in str(path).split(".") if part]:
        if isinstance(current, list):
            current = current[int(segment)]
        elif isinstance(current, dict):
            current = current[segment]
        else:
            raise KeyError(f"Cannot descend into {segment!r} on non-container value {current!r}")
    return current


def _apply_path_assertions(group: str, root: Any, specs: Dict[str, Any]) -> None:
    for path, expected in specs.items():
        actual = _get_path_value(root, path)
        _assert_value(f"{group}[{path}]", actual, expected)


def _positions_map(fake_client: FakeCCXTClient) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in fake_client.export_positions():
        result[f"{row['symbol']}|{row['position_side']}"] = float(row["size"])
    return result


def _apply_assertions(
    bot,
    fake_client: FakeCCXTClient,
    scenario: dict,
    *,
    step_summaries: List[dict] | None = None,
    log_text: str = "",
) -> None:
    assertions = scenario.get("assertions") or {}
    if not assertions:
        return
    state = fake_client.export_state()
    hsl_trace = _extract_hsl_trace(bot)

    if "fill_count" in assertions:
        _assert_numeric("fill_count", float(len(fake_client.fills)), assertions["fill_count"])
    if "final_balance" in assertions:
        _assert_numeric(
            "final_balance",
            float(fake_client.balance_total),
            assertions["final_balance"],
        )
    if "last_prices" in assertions:
        current_prices = fake_client.get_current_step()["prices"]
        for symbol, expected in assertions["last_prices"].items():
            if symbol not in current_prices:
                raise AssertionError(f"last_prices: missing symbol {symbol}")
            _assert_numeric(f"last_price[{symbol}]", float(current_prices[symbol]), expected)
    if "final_positions" in assertions:
        actual_positions = _positions_map(fake_client)
        for key, expected in assertions["final_positions"].items():
            actual = float(actual_positions.get(key, 0.0))
            _assert_numeric(f"final_position[{key}]", actual, expected)
    if "halted_psides" in assertions:
        for pside, expected in assertions["halted_psides"].items():
            actual = bool(bot._hsl_state(pside)["halted"])
            if actual != bool(expected):
                raise AssertionError(f"halted_psides[{pside}]: expected {expected} got {actual}")
    if "state_paths" in assertions:
        _apply_path_assertions("state_paths", state, assertions["state_paths"])
    if "hsl_paths" in assertions:
        _apply_path_assertions("hsl_paths", hsl_trace, assertions["hsl_paths"])
    if "summary_paths" in assertions:
        summary_root = {
            "step_count": len(step_summaries or []),
            "last": (step_summaries or [None])[-1],
            "steps": step_summaries or [],
        }
        _apply_path_assertions("summary_paths", summary_root, assertions["summary_paths"])
    if "log_contains" in assertions:
        for fragment in assertions["log_contains"]:
            if str(fragment) not in log_text:
                raise AssertionError(f"log_contains missing fragment: {fragment!r}")


def _install_fake_user_override(config: dict, scenario_path: str, user: str | None) -> tuple[str, callable]:
    config.setdefault("live", {})
    fake_user = user or str(config["live"].get("user") or "fake_runner")
    config["live"]["user"] = fake_user
    fake_user_info = {
        "exchange": "fake",
        "quote": str(config["live"].get("quote") or "USDT"),
        "fake_scenario_path": scenario_path,
    }
    original = passivbot_mod.load_user_info

    def patched(requested_user: str):
        if requested_user == fake_user:
            return dict(fake_user_info)
        return original(requested_user)

    passivbot_mod.load_user_info = patched
    return fake_user, lambda: setattr(passivbot_mod, "load_user_info", original)


def _prime_fake_fill_cache(bot, fake_client: FakeCCXTClient, cache_root: Path | None = None) -> Path:
    root = cache_root or Path("caches") / "fill_events"
    cache_path = root / str(bot.exchange) / str(bot.user)
    cache_path.mkdir(parents=True, exist_ok=True)
    for path in cache_path.glob("*.json"):
        path.unlink()
    metadata_path = cache_path / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    cache = FillEventCache(cache_path)
    events = [FillEvent.from_dict(event) for event in fake_client.get_fill_events(None, None)]
    cache.save(events)
    return cache_path


def _prime_fake_candles(bot, fake_client: FakeCCXTClient) -> None:
    if not hasattr(bot, "cm"):
        return
    for symbol in fake_client.symbols:
        rows = fake_client._candles_by_symbol.get(symbol, [])[: fake_client.current_index + 1]
        arr = np.zeros(len(rows), dtype=CANDLE_DTYPE)
        for idx, row in enumerate(rows):
            arr[idx]["ts"] = int(row[0])
            arr[idx]["o"] = float(row[1])
            arr[idx]["h"] = float(row[2])
            arr[idx]["l"] = float(row[3])
            arr[idx]["c"] = float(row[4])
            arr[idx]["bv"] = float(row[5])
        bot.cm._cache[symbol] = arr
        bot.cm._ema_cache.pop(symbol, None)
        bot.cm._current_close_cache.pop(symbol, None)
        bot.cm._tf_range_cache.pop(symbol, None)


def _install_runtime_overrides(bot, scenario: dict) -> None:
    if hasattr(bot, "cca") and isinstance(bot.cca, FakeCCXTClient):
        bot.get_exchange_time = lambda: int(bot.cca.now_ms)


def _fake_active_red_psides(bot) -> List[str]:
    return [
        pside
        for pside in bot._hsl_psides()
        if bot._equity_hard_stop_enabled(pside)
        and bot._equity_hard_stop_runtime_red_latched(pside)
        and not bot._hsl_state(pside)["halted"]
    ]


async def _run_fake_red_supervisor_step(bot) -> dict:
    active_red_psides = _fake_active_red_psides(bot)
    if not active_red_psides:
        return {"red_supervisor": False}

    for pside in list(active_red_psides):
        state = bot._hsl_state(pside)
        n_positions = bot._equity_hard_stop_count_open_positions(pside)
        entry_orders, nonpanic_close_orders = bot._equity_hard_stop_count_blocking_open_orders(pside)
        if n_positions == 0 and entry_orders == 0 and nonpanic_close_orders == 0:
            if state["red_flat_confirmations"] == 0:
                state["pending_stop_event"] = await bot._equity_hard_stop_compute_stop_event(
                    pside, int(bot.get_exchange_time())
                )
            state["red_flat_confirmations"] += 1
        else:
            state["red_flat_confirmations"] = 0
            state["pending_stop_event"] = None
        bot._equity_hard_stop_log_red_progress(
            pside,
            n_positions,
            entry_orders,
            nonpanic_close_orders,
            state["red_flat_confirmations"],
        )
        if state["red_flat_confirmations"] >= 2:
            await bot._equity_hard_stop_finalize_red_stop(pside, state["pending_stop_event"])

    active_red_psides = _fake_active_red_psides(bot)
    if not active_red_psides:
        return {"red_supervisor": True, "finalized": True}

    for pside in active_red_psides:
        bot._equity_hard_stop_set_red_runtime_forced_modes(pside)
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    await bot.execute_to_exchange()
    return {"red_supervisor": True, "finalized": False}


async def _run_fake_bot(
    bot,
    fake_client: FakeCCXTClient,
    max_steps: int | None,
    *,
    snapshot_dir: Path | None = None,
    run_initial_cycle: bool = True,
) -> List[dict]:
    summaries: List[dict] = []
    steps_run = 0

    if run_initial_cycle:
        _prime_fake_candles(bot, fake_client)
        result = await _run_fake_cycle(bot)
        summaries.append(
            {
                "step_index": int(fake_client.current_index),
                "timestamp": int(fake_client.now_ms),
                "result": str(result),
                "fills": len(fake_client.fills),
                "open_orders": len(fake_client.open_orders),
                "positions": fake_client.export_positions(),
            }
        )
        if snapshot_dir is not None:
            _dump_json(
                snapshot_dir / f"step_{int(fake_client.current_index):04d}.json",
                {
                    "summary": summaries[-1],
                    "state": fake_client.export_state(),
                    "hsl_trace": _extract_hsl_trace(bot),
                },
            )
        steps_run += 1

    while fake_client.has_next_step():
        if max_steps is not None and steps_run >= max_steps:
            break
        fake_client.advance_time()
        _prime_fake_candles(bot, fake_client)
        result = await _run_fake_cycle(bot)
        summaries.append(
            {
                "step_index": int(fake_client.current_index),
                "timestamp": int(fake_client.now_ms),
                "result": str(result),
                "fills": len(fake_client.fills),
                "open_orders": len(fake_client.open_orders),
                "positions": fake_client.export_positions(),
            }
        )
        if snapshot_dir is not None:
            _dump_json(
                snapshot_dir / f"step_{int(fake_client.current_index):04d}.json",
                {
                    "summary": summaries[-1],
                    "state": fake_client.export_state(),
                    "hsl_trace": _extract_hsl_trace(bot),
                },
            )
        steps_run += 1

    return summaries


async def _run_fake_cycle(bot):
    if not await bot.update_pos_oos_pnls_ohlcvs():
        return {"updated": False}
    if bot._equity_hard_stop_enabled():
        if any(
            bot._equity_hard_stop_runtime_red_latched(pside) and not bot._hsl_state(pside)["halted"]
            for pside in bot._hsl_psides()
            if bot._equity_hard_stop_enabled(pside)
        ):
            if getattr(bot, "exchange", "").lower() == "fake":
                return await _run_fake_red_supervisor_step(bot)
            await bot._equity_hard_stop_run_red_supervisor()
            return {"red_supervisor": True}
        await bot._equity_hard_stop_check()
        if any(
            bot._equity_hard_stop_runtime_red_latched(pside) and not bot._hsl_state(pside)["halted"]
            for pside in bot._hsl_psides()
            if bot._equity_hard_stop_enabled(pside)
        ):
            if getattr(bot, "exchange", "").lower() == "fake":
                return await _run_fake_red_supervisor_step(bot)
            await bot._equity_hard_stop_run_red_supervisor()
            return {"red_supervisor": True}
    return await bot.execute_to_exchange()

def _load_run_artifacts(output_dir: Path) -> dict[str, Any]:
    def _load(name: str, default):
        path = output_dir / name
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    log_path = output_dir / "fake_live.log"
    return {
        "step_summaries": _load("step_summaries.json", []),
        "fake_exchange_state": _load("fake_exchange_state.json", {}),
        "fills": _load("fills.json", []),
        "positions": _load("positions.json", []),
        "hsl_trace": _load("hsl_trace.json", {}),
        "run_metadata": _load("run_metadata.json", {}),
        "log_text": log_path.read_text(encoding="utf-8") if log_path.exists() else "",
    }


def _canonical_fill(fill: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in fill.items():
        if key in {"clientOrderId", "id", "order"}:
            continue
        if key == "info" and isinstance(value, dict):
            normalized[key] = {
                info_key: info_value
                for info_key, info_value in value.items()
                if info_key not in {"clientOrderId"}
            }
            continue
        normalized[key] = value
    return normalized


def _canonical_hsl_trace(trace: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(trace))
    for pside_state in normalized.values():
        if not isinstance(pside_state, dict):
            continue
        stop_event = pside_state.get("last_stop_event")
        if isinstance(stop_event, dict):
            stop_event.pop("triggered_at", None)
            stop_event.pop("user", None)
    return normalized


def _canonicalize_artifact(name: str, payload: Any) -> Any:
    if name == "step_summaries":
        return payload
    if name == "fills":
        return sorted(
            [_canonical_fill(fill) for fill in payload],
            key=lambda x: (
                x.get("timestamp"),
                x.get("symbol"),
                x.get("position_side"),
                x.get("side"),
                x.get("price"),
                x.get("amount"),
                x.get("pnl"),
                x.get("reduceOnly"),
            ),
        )
    if name == "positions":
        return sorted(
            payload,
            key=lambda x: (
                x.get("symbol"),
                x.get("position_side"),
                x.get("entry_price"),
                x.get("size"),
            ),
        )
    if name == "fake_exchange_state":
        normalized = dict(payload)
        if isinstance(normalized.get("fills"), list):
            normalized["fills"] = _canonicalize_artifact("fills", normalized["fills"])
        if isinstance(normalized.get("positions"), list):
            normalized["positions"] = _canonicalize_artifact("positions", normalized["positions"])
        if isinstance(normalized.get("open_orders"), list):
            normalized["open_orders"] = sorted(
                [
                    {
                        key: value
                        for key, value in order.items()
                        if key not in {"clientOrderId", "id"}
                    }
                    for order in normalized["open_orders"]
                ],
                key=lambda x: (
                    x.get("symbol"),
                    x.get("position_side"),
                    x.get("side"),
                    x.get("price"),
                    x.get("amount"),
                    x.get("qty"),
                ),
            )
        return normalized
    if name == "hsl_trace":
        return _canonical_hsl_trace(payload)
    return payload


def _compare_run_artifacts(legacy: dict[str, Any], staged: dict[str, Any]) -> dict[str, Any]:
    keys = ("step_summaries", "fake_exchange_state", "fills", "positions", "hsl_trace")
    diffs = []
    for key in keys:
        legacy_normalized = _canonicalize_artifact(key, legacy.get(key))
        staged_normalized = _canonicalize_artifact(key, staged.get(key))
        if legacy_normalized != staged_normalized:
            diffs.append(
                {
                    "artifact": key,
                    "legacy": legacy_normalized,
                    "staged": staged_normalized,
                }
            )
    return {
        "match": not diffs,
        "diff_count": len(diffs),
        "diffs": diffs,
        "legacy_mode": legacy.get("run_metadata", {}).get("authoritative_refresh_mode"),
        "staged_mode": staged.get("run_metadata", {}).get("authoritative_refresh_mode"),
    }


async def _run_fake_case(
    *,
    config_path: str,
    scenario_path: str,
    user: str | None,
    max_steps: int | None,
    output_dir: Path,
    log_level: int,
    snapshot_each_step: bool,
    authoritative_refresh_mode: str,
    enforce_assertions: bool = True,
) -> Path:
    config = load_prepared_config(
        config_path,
        verbose=False,
        target="live",
        runtime="live",
    )
    config.setdefault("live", {})
    config["live"]["fake_scenario_path"] = scenario_path
    config["live"]["authoritative_refresh_mode"] = str(authoritative_refresh_mode or "legacy")

    scenario = load_fake_scenario(scenario_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "fake_live.log"
    log_handler = _attach_file_logging(log_path)
    _, restore_user_override = _install_fake_user_override(config, scenario_path, user)
    bot = None
    restore_candle_trace = lambda: None

    try:
        bot = setup_bot(config)
        if bot.exchange != "fake":
            raise ValueError(
                f"Config user resolved to exchange '{bot.exchange}', expected 'fake' for fake harness"
            )
        bot.debug_mode = True
        if not isinstance(bot.cca, FakeCCXTClient):
            raise TypeError("Fake harness expected bot.cca to be FakeCCXTClient")
        candle_remote_fetches, restore_candle_trace = _install_candle_remote_fetch_trace(bot)
        _prime_fake_fill_cache(bot, bot.cca)
        _prime_fake_candles(bot, bot.cca)
        _install_runtime_overrides(bot, scenario)
        await bot.start_bot()
        bot.debug_mode = False
        snapshot_dir = (output_dir / "snapshots") if snapshot_each_step else None
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
        step_summaries = await _run_fake_bot(
            bot,
            bot.cca,
            max_steps,
            snapshot_dir=snapshot_dir,
            run_initial_cycle=bool(scenario.get("run_initial_cycle", True)),
        )
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if enforce_assertions:
            _apply_assertions(
                bot,
                bot.cca,
                scenario,
                step_summaries=step_summaries,
                log_text=log_text,
            )

        _dump_json(output_dir / "step_summaries.json", step_summaries)
        _dump_json(output_dir / "fake_exchange_state.json", bot.cca.export_state())
        _dump_json(output_dir / "fills.json", bot.cca.fills)
        _dump_json(output_dir / "positions.json", bot.cca.export_positions())
        _dump_json(output_dir / "hsl_trace.json", _extract_hsl_trace(bot))
        _dump_json(
            output_dir / "run_metadata.json",
            {
                "authoritative_refresh_mode": str(
                    config.get("live", {}).get("authoritative_refresh_mode", "legacy")
                ),
                "assertions_enforced": bool(enforce_assertions),
                "user": str(config.get("live", {}).get("user") or ""),
                "scenario_path": str(scenario_path),
            },
        )
        remote_calls = bot.cca.export_request_log()
        _dump_json(output_dir / "remote_calls.json", remote_calls)
        _dump_json(output_dir / "remote_call_summary.json", _summarize_remote_calls(remote_calls))
        _dump_json(output_dir / "candle_remote_fetches.json", candle_remote_fetches)
        return output_dir
    finally:
        try:
            if bot is not None:
                await shutdown_bot(bot)
        finally:
            try:
                restore_candle_trace()
            except Exception:
                pass
            restore_user_override()
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()


async def _async_main(args: argparse.Namespace) -> int:
    configure_logging(debug=args.log_level)
    scenario = load_fake_scenario(args.scenario)
    output_dir = _build_output_dir(args.output_dir, scenario)

    if getattr(args, "compare_authoritative_refresh_modes", False):
        legacy_dir = output_dir / "legacy"
        staged_dir = output_dir / "staged"
        legacy_run = await _run_fake_case(
            config_path=args.config,
            scenario_path=args.scenario,
            user=_mode_run_user(args.user, "legacy"),
            max_steps=args.max_steps,
            output_dir=legacy_dir,
            log_level=args.log_level,
            snapshot_each_step=args.snapshot_each_step,
            authoritative_refresh_mode="legacy",
            enforce_assertions=False,
        )
        staged_run = await _run_fake_case(
            config_path=args.config,
            scenario_path=args.scenario,
            user=_mode_run_user(args.user, "staged"),
            max_steps=args.max_steps,
            output_dir=staged_dir,
            log_level=args.log_level,
            snapshot_each_step=args.snapshot_each_step,
            authoritative_refresh_mode="staged",
            enforce_assertions=False,
        )
        report = _compare_run_artifacts(
            _load_run_artifacts(legacy_run),
            _load_run_artifacts(staged_run),
        )
        _dump_json(output_dir / "comparison.json", report)
        print(str(output_dir))
        return 0

    run_dir = await _run_fake_case(
        config_path=args.config,
        scenario_path=args.scenario,
        user=args.user,
        max_steps=args.max_steps,
        output_dir=output_dir,
        log_level=args.log_level,
        snapshot_each_step=args.snapshot_each_step,
        authoritative_refresh_mode=str(
            getattr(args, "authoritative_refresh_mode", "legacy") or "legacy"
        ),
    )
    print(str(run_dir))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run passivbot against the fake exchange harness")
    parser.add_argument("config", help="Passivbot config path")
    parser.add_argument("scenario", help="Fake scenario path (HJSON or JSON)")
    parser.add_argument("--user", default=None, help="Override live.user from the config")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum execution cycles to run, including the initial boot cycle",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory root (default: artifacts/fake_live)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=1,
        help="Logging level 0-3 (warning/info/debug/trace)",
    )
    parser.add_argument(
        "--snapshot-each-step",
        action="store_true",
        help="Write a JSON snapshot after each execution cycle",
    )
    parser.add_argument(
        "--authoritative-refresh-mode",
        choices=("legacy", "staged"),
        default="legacy",
        help="Select which authoritative refresh path to use during fake-live runs",
    )
    parser.add_argument(
        "--compare-authoritative-refresh-modes",
        action="store_true",
        help="Run both legacy and staged authoritative refresh modes and write a structured comparison report",
    )
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
