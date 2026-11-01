"""Read-only preparation plans and exact simulation-owner preparation reports.

Plans describe requirements, not verified candle coverage or reusable fitness.
Only the existing backtest/suite owners acquire and validate data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

from config.access import require_config_value
from config.universe import require_side_universe_sources
from simulation_data import simulation_data_scope


PREPARATION_REPORT_SCHEMA_VERSION = 1


def optimizer_data_config(config: dict) -> dict:
    """Share the optimizer's fixed-source and reachable-bound preparation view."""
    from optimization.warmup import (
        apply_optimizer_universe_sources,
        build_optimizer_data_config,
        refresh_optimizer_source_bindings,
    )

    result = deepcopy(config)
    apply_optimizer_universe_sources(result)
    refresh_optimizer_source_bindings(result)
    return build_optimizer_data_config(result)


def validate_optimizer_preparation(
    config: dict, suite_cfg: dict
) -> None:
    """Validate effective GPU contracts before historical preparation or device imports."""

    if str(config.get("optimize", {}).get("backend", "")).strip().lower() != "gpu":
        return
    from suite_runner import build_scenarios
    from config.gpu import resolve_gpu_screening, resolve_gpu_sizing, validate_gpu_backtest_config
    from optimization.warmup import _apply_config_overrides

    effective_config = deepcopy(config)
    _apply_config_overrides(effective_config, config["optimize"].get("fixed_runtime_overrides", {}))
    if effective_config['bot']["portfolio"]['strategy_kind'] != config['bot']["portfolio"]['strategy_kind']:
        raise ValueError("fixed_runtime_overrides may not change bot.portfolio.strategy_kind")
    validate_gpu_backtest_config(effective_config)
    for field in ("batch_size", "max_dispatch_candidate_bars"):
        resolve_gpu_sizing(config["optimize"]["gpu"].get(field), field=field)
    screening = resolve_gpu_screening(config["optimize"]["gpu"].get("screening"))
    if suite_cfg.get("enabled"):
        scenarios, _reducer_cfg = build_scenarios(
            dict(suite_cfg),
            base_exchanges=effective_config.get("backtest", {}).get("exchanges"),
        )
        unknown = set(screening["scenarios"]) - {scenario.label for scenario in scenarios}
        if unknown:
            raise ValueError(f"GPU screening.scenarios contains unknown labels: {sorted(unknown)}")
        for scenario in scenarios:
            scenario_config = deepcopy(effective_config)
            _apply_config_overrides(scenario_config, dict(scenario.overrides or {}))
            validate_gpu_backtest_config(scenario_config)
    elif screening["scenarios"]:
        raise ValueError("GPU screening.scenarios requires a scenario suite")


def _file_fact(path: Path) -> dict:
    if not path.is_file():
        return {"path": path.as_posix(), "present": False}
    # Small metadata snapshots only. No locks, database opens, or array reads.
    raw = path.read_bytes()
    stat = path.stat()
    return {"path": path.as_posix(), "present": True, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "mtime_ns": stat.st_mtime_ns}


def _literal_memberships(config: dict) -> tuple[dict, list[str] | None]:
    memberships = {
        field: {side: deepcopy(value) for side, value in
                require_side_universe_sources(config, field).items()}
        for field in ("approved_coins", "ignored_coins")
    }
    approved = memberships["approved_coins"]
    if any(not isinstance(value, list) or any(not isinstance(coin, str) or coin == "all"
                                             for coin in value)
           for value in approved.values()):
        return memberships, None
    # This is an upper bound on declared identifiers, not a factual market-column
    # identity. Ignore alias-based exclusions until the universe owner resolves them.
    return memberships, list(dict.fromkeys(coin for values in approved.values() for coin in values))


def build_preparation_plan(config: dict, *, for_optimize: bool = False,
                           target_cuda_free_bytes: int | None = None) -> dict:
    """Inspect declared requirements without network, cache writes or GPU imports."""
    from suite_runner import (
        build_scenarios, extract_suite_config, apply_scenario_overrides,
        _normalize_date_to_ts, resolve_coin_sources,
    )
    from simulation_validation import validate_simulation_config
    from config.age_policy import minimum_coin_age_days
    from backtest_universe import backtest_side_enabled
    from warmup_utils import compute_backtest_warmup_minutes
    from optimization.warmup import compute_optimizer_backtest_warmup_minutes
    from optimization.warmup import refresh_optimizer_source_bindings
    from optimization.gpu.residency import multicoin_invariant_bytes
    from config.overrides import parse_overrides

    if target_cuda_free_bytes is not None and (
        isinstance(target_cuda_free_bytes, bool) or not isinstance(target_cuda_free_bytes, int)
        or target_cuda_free_bytes <= 0
    ):
        raise ValueError("target CUDA free bytes must be a positive integer")
    # Reading explicit override files is allowed; market-symbol discovery is
    # deferred to preparation so planning cannot create alias-map caches.
    source = parse_overrides(config, verbose=False, symbol_normalizer=lambda value: value)
    prepared = optimizer_data_config(source) if for_optimize else source
    suite = extract_suite_config(prepared, None)
    if for_optimize:
        validate_optimizer_preparation(prepared, suite)
    scenarios = build_scenarios(suite, base_exchanges=prepared["backtest"]["exchanges"])[0] if suite["enabled"] else [None]
    cases = []
    exchanges = list(prepared["backtest"]["exchanges"])
    unresolved = ["Actual market aliases, listing evidence, ordered candle columns and candle coverage require owner preparation."]
    for scenario in scenarios:
        case = deepcopy(prepared)
        label = "base" if scenario is None else scenario.label
        if scenario is not None:
            patch = {}
            for field, value in (("approved_coins", scenario.coins), ("ignored_coins", scenario.ignored_coins)):
                if value is not None:
                    patch.update({f"bot.{side}.universe.{field}": value for side in ("long", "short")})
            patch.update(scenario.overrides or {})
            if scenario.start_date is not None:
                case["backtest"]["start_date"] = scenario.start_date
            if scenario.end_date is not None:
                case["backtest"]["end_date"] = scenario.end_date
            if scenario.exchanges:
                case["backtest"]["exchanges"] = list(scenario.exchanges)
            case["backtest"]["coin_sources"] = resolve_coin_sources(
                case["backtest"].get("coin_sources", {}), scenario.coin_sources)
            apply_scenario_overrides(case, patch)
            if for_optimize:
                refresh_optimizer_source_bindings(case)
        validate_simulation_config(case, context=label)
        start = require_config_value(case, "backtest.start_date")
        end = require_config_value(case, "backtest.end_date")
        start_ms, end_ms = _normalize_date_to_ts(start), _normalize_date_to_ts(end)
        if end_ms <= start_ms:
            raise ValueError(f"Scenario {label} end date must follow its start date")
        warmup = (compute_optimizer_backtest_warmup_minutes(case) if for_optimize
                  else compute_backtest_warmup_minutes(case))
        sources, identifiers = _literal_memberships(case)
        interval = int(require_config_value(case, "backtest.candle_interval_minutes"))
        # Count the inclusive source clock and round the history prefix upward.
        rows = math.ceil((end_ms - start_ms) / (60_000 * interval)) + math.ceil(warmup / interval) + 1
        estimate = None if identifiers is None else multicoin_invariant_bytes(rows, len(identifiers), device="cuda")
        note = "Conservative declared-identifier estimate; actual prepared shape and live device admission remain authoritative."
        if identifiers is None:
            unresolved.append(f"Scenario {label} has a dynamic/file universe; its column count is unknown until resolution.")
        case_exchanges = list(case["backtest"]["exchanges"])
        exchanges.extend(case_exchanges)
        cases.append({"label": label, "start_date_literal": start, "end_date_literal": end,
                      "start_timestamp_ms": start_ms, "end_timestamp_ms": end_ms,
                      "dates_are_fixed": str(start).strip() not in ("", "now", "today") and str(end).strip() not in ("", "now", "today"),
                      "exchanges": case_exchanges, "universe_sources": sources,
                      "declared_identifiers": identifiers, "warmup_minutes": warmup,
                      "minimum_coin_age_days": {side: minimum_coin_age_days(case, side) for side in ("long", "short")},
                      "listing_facts_required_for_sides": [side for side in ("long", "short")
                          if backtest_side_enabled(case, side) and minimum_coin_age_days(case, side) > 0],
                      "estimated_rows_with_warmup": rows, "cuda_invariant_bytes_estimate": estimate,
                      "target_cuda_45_percent_bytes": None if target_cuda_free_bytes is None else int(target_cuda_free_bytes * .45),
                      "estimate_note": note,
                      "coin_sources": deepcopy(case["backtest"].get("coin_sources", {})),
                      "market_settings_sources": deepcopy(case["backtest"].get("market_settings_sources", {}))})
        exchanges.extend(case["backtest"].get("coin_sources", {}).values())
        exchanges.extend(case["backtest"].get("market_settings_sources", {}).values())
    paths = [Path("caches") / exchange / name for exchange in dict.fromkeys(exchanges)
             for name in ("markets.json", "first_timestamps.json")]
    paths += [Path("caches") / name for name in (
        "first_ohlcv_timestamps_unified.json", "first_ohlcv_timestamps_unified.version",
        "first_ohlcv_timestamps_unified_exchange_specific.json",
        "first_ohlcv_timestamps_unified_exchange_specific_symbols.json",
    )]
    facts = [_file_fact(path) for path in paths]
    return {"schema_version": PREPARATION_REPORT_SCHEMA_VERSION, "kind": "preparation-plan",
            "consumer": "optimize" if for_optimize else "backtest",
            "suite": {"enabled": bool(suite["enabled"]), "reducer": deepcopy(suite["reducer"])},
            "acquisition_mode": prepared["backtest"].get("acquisition_mode", "auto"),
            "cases": cases, "metadata": facts,
            "raw_catalog_present": Path("caches/ohlcvs/catalog.sqlite").is_file(),
            "missing_metadata": [fact["path"] for fact in facts if not fact["present"]],
            "coverage": "not_verified", "unresolved": unresolved,
            "scope": "Read-only requirements estimate; no candles acquired, no strategy replay, no fitness reuse or device admission."}


@simulation_data_scope
async def prepare_optimizer_inputs(config: dict, *, force_refetch_gaps: bool = False) -> dict:
    """Materialize exactly the optimizer's owner data view, without any replay."""
    from config.overrides import parse_overrides
    from utils import format_approved_ignored_coins
    from optimization.warmup import apply_optimizer_universe_sources
    from suite_runner import extract_suite_config
    from optimize_suite import prepare_suite_contexts
    from shared_arrays import SharedArrayManager
    from hlcvs_manifest import numpy_artifact_loader
    from backtest import prepare_hlcvs_mss

    prepared = parse_overrides(config, verbose=False)
    apply_optimizer_universe_sources(prepared)
    validate_optimizer_preparation(prepared, extract_suite_config(prepared, None))
    await format_approved_ignored_coins(prepared, prepared["backtest"]["exchanges"],
                                       prefer_backtest_coin_source_keys=True)
    data_config = optimizer_data_config(prepared)
    suite = extract_suite_config(prepared, None)
    allow_nan = prepared["optimize"]["backend"] == "gpu"
    report = describe_prepared_inputs(prepared)
    manager = SharedArrayManager(file_backed=True)
    contexts = []
    try:
        with numpy_artifact_loader(manager.load_numpy_artifact):
            if suite["enabled"]:
                contexts, _ = await prepare_suite_contexts(data_config, suite, shared_array_manager=manager,
                                                         allow_internal_nan_gaps=allow_nan,
                                                         force_refetch_gaps=force_refetch_gaps,
                                                         preparation_report=report)
                for context in contexts:
                    report["cases"].append(case_summary(context))
                    manager.discard_file_pages()
            else:
                exchanges = prepared["backtest"]["exchanges"]
                for exchange in (["combined"] if len(exchanges) > 1 else exchanges):
                    coins, _hlcvs, _mss, _results, cache, _btc, timestamps = await prepare_hlcvs_mss(
                        data_config, exchange, allow_internal_nan_gaps=allow_nan,
                        force_refetch_gaps=force_refetch_gaps)
                    report["datasets"].append(dataset_summary(exchange, cache, coins, len(timestamps)))
                    manager.discard_file_pages()
    except BaseException as error:
        # Failed owner frames can retain views into the mappings. Release their
        # locals before closing our owner; keep the exception and traceback.
        import traceback
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        for context in contexts:
            for attachments in context.attachments.values():
                for attachment in attachments.values():
                    attachment.close()
            context.timestamps.clear()
            context.shared_hlcvs_np.clear()
            context.shared_btc_np.clear()
        contexts.clear()
        _hlcvs = _btc = timestamps = None
        manager.cleanup()
    if not suite["enabled"]:
        return describe_prepared_inputs(prepared, datasets=report["datasets"])
    return report


def dataset_summary(exchange: str, cache_dir, coins, rows: int) -> dict:
    manifest = Path(cache_dir) / "manifest.json"
    fact = _file_fact(manifest)
    return {"exchange": exchange, "cache_dir": str(cache_dir), "ordered_coins": list(coins),
            "rows": int(rows), "manifest_present": fact["present"],
            "manifest_sha256": fact.get("sha256")}


def case_summary(context) -> dict:
    """Describe actual owner-selected case clocks and ordered membership."""
    return {"label": context.label,
            "start_date_literal": context.config["backtest"]["start_date"],
            "end_date_literal": context.config["backtest"]["end_date"],
            "exchanges": list(context.exchanges),
            "ordered_coins": deepcopy(context.config["backtest"]["coins"]),
            "rows": {exchange: len(context.timestamps[exchange]) for exchange in context.exchanges}}


def describe_prepared_inputs(config: dict, *, datasets=(), scenario_contexts=()) -> dict:
    """Shared report schema for download and existing optimizer session provenance."""
    from suite_runner import extract_suite_config
    suite = extract_suite_config(config, None)
    datasets = deepcopy(list(datasets))
    cases = [case_summary(context) for context in scenario_contexts]
    if not suite["enabled"] and datasets:
        cases = [{"label": "base", "start_date_literal": config["backtest"]["start_date"],
                  "end_date_literal": config["backtest"]["end_date"],
                  "exchanges": [dataset["exchange"] for dataset in datasets],
                  "ordered_coins": {dataset["exchange"]: dataset["ordered_coins"] for dataset in datasets},
                  "rows": {dataset["exchange"]: dataset["rows"] for dataset in datasets}}]
    return {"schema_version": PREPARATION_REPORT_SCHEMA_VERSION, "kind": "prepared-inputs",
            "consumer": "optimize", "suite": {"enabled": bool(suite["enabled"]), "reducer": deepcopy(suite["reducer"])},
            "acquisition_mode": config["backtest"].get("acquisition_mode", "auto"),
            "datasets": datasets, "cases": cases,
            "scope": "Owner-validated prepared inputs only; no strategy replay or fitness."}
