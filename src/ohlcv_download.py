import argparse
import asyncio
import logging
import json
import math
import sys
from pathlib import Path

# Backtest's ordinary import can rebuild the extension. Planning must only
# inspect an already-built runtime and must not prune/build/stamp anything.
if "--plan" in sys.argv[1:]:
    sys.dont_write_bytecode = True
    from rust_utils import source_fingerprint, verify_loaded_runtime_extension
    expected = source_fingerprint(Path(__file__).resolve().parents[1] / "passivbot-rust")
    try:
        import passivbot_rust

        runtime = verify_loaded_runtime_extension(fingerprint=expected)
        if expected is not None and runtime.get("runtime_compiled_source_stamp") != expected:
            raise RuntimeError("missing or mismatched native source stamp")
    except (ImportError, RuntimeError, OSError) as error:
        raise SystemExit(
            "Read-only planning requires a current stamped Rust extension; "
            "rebuild it explicitly first."
        ) from error

    async def prepare_hlcvs_mss(*args, **kwargs):
        from backtest import prepare_hlcvs_mss as owner
        return await owner(*args, **kwargs)
else:
    from backtest import prepare_hlcvs_mss
from cli_utils import get_cli_prog
from config import load_input_config, prepare_config
from config.access import require_config_value, require_live_value
from config_utils import comma_separated_values, update_config_with_args
from pure_funcs import str2bool
from utils import format_approved_ignored_coins
from simulation_data import simulation_data_scope


@simulation_data_scope
async def warm_ohlcv_caches(config: dict, *, force_refetch_gaps: bool = False,
                           for_optimize: bool = False) -> dict:
    if for_optimize:
        from simulation_preparation import prepare_optimizer_inputs
        return await prepare_optimizer_inputs(config, force_refetch_gaps=force_refetch_gaps)
    exchanges = require_config_value(config, "backtest.exchanges")
    await format_approved_ignored_coins(
        config, exchanges, prefer_backtest_coin_source_keys=True
    )
    config.setdefault("backtest", {})
    config["backtest"]["cache_dir"] = {}
    config["backtest"]["coins"] = {}
    if len(exchanges) > 1:
        exchange = "combined"
        coins, _hlcvs, _mss, _results_path, cache_dir, _btc, _timestamps = await prepare_hlcvs_mss(
            config,
            exchange,
            force_refetch_gaps=force_refetch_gaps,
        )
        config["backtest"]["coins"][exchange] = coins
        config["backtest"]["cache_dir"][exchange] = str(cache_dir)
        logging.info("download materialized %s HLCV dataset at %s", exchange, cache_dir)
        return {"kind": "prepared-inputs", "consumer": "backtest-base",
                "cache_dirs": dict(config["backtest"]["cache_dir"])}
    for exchange in exchanges:
        coins, _hlcvs, _mss, _results_path, cache_dir, _btc, _timestamps = await prepare_hlcvs_mss(
            config,
            exchange,
            force_refetch_gaps=force_refetch_gaps,
        )
        config["backtest"]["coins"][exchange] = coins
        config["backtest"]["cache_dir"][exchange] = str(cache_dir)
        logging.info("download materialized %s HLCV dataset at %s", exchange, cache_dir)
    return {"kind": "prepared-inputs", "consumer": "backtest-base",
            "cache_dirs": dict(config["backtest"]["cache_dir"])}


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog=get_cli_prog("download"), description="download ohlcv data"
    )
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    parser.add_argument("--for-optimize", action="store_true",
                        help="Prepare the exact optimizer suite and maximum searched warmup, without replay.")
    parser.add_argument("--plan", action="store_true",
                        help="Print read-only requirements and missing metadata; do not acquire or write cache data.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Write the plan or completed preparation report to this explicit path.")
    parser.add_argument("--target-cuda-free-gib", type=float, default=None,
                        help="Optional target free VRAM for the conservative --plan estimate; no device probe.")
    parser.add_argument("--offline", type=str2bool, default=None, dest="backtest.offline",
                        help="Use only verified local data and metadata (y/n).")
    parser.add_argument("--acquisition-mode", choices=("auto", "rest"), default=None,
                        dest="backtest.acquisition_mode",
                        help="Public acquisition route; validation and offline requirements are unchanged.")
    parser.add_argument(
        "--symbols",
        "-s",
        dest="approved_coins",
        type=str,
        default=None,
        metavar="CSV_OR_PATH",
        help=(
            "Approved coins. Use CSV like BTC,ETH,XRP, the literal 'all', a path to a JSON "
            "coin list file, or a JSON/HJSON per-side object like "
            '{"long":["BTC"],"short":"all"}. An empty value approves nothing; null in a per-side object inherits the shared approval source. Undirected null is invalid.'
        ),
    )
    parser.add_argument(
        "--ignored-coins",
        "-ic",
        dest="ignored_coins",
        type=str,
        default=None,
        metavar="CSV_OR_PATH",
        help="Ignored coins. CSV, a JSON/HJSON per-side object or a JSON coin list file. An empty value or null clears ignores, including null in a per-side object.",
    )
    for side in ("long", "short"):
        parser.add_argument(
            f"--bot.{side}.universe.minimum_coin_age_days", type=float,
            default=None, metavar="FLOAT", help=f"Minimum {side} initial-entry age in days.",
        )
    side_source_keys = {
        f"bot.{side}.universe.{field}"
        for side in ("long", "short") for field in ("approved_coins", "ignored_coins")
    }
    side_source_keys.add("bot.global.universe.approved_coins")
    for key in sorted(side_source_keys):
        parser.add_argument(f"--{key}", dest=key, type=str, default=argparse.SUPPRESS, metavar="CSV_OR_PATH")
    parser.add_argument(
        "--exchanges",
        "-e",
        dest="backtest.exchanges",
        type=comma_separated_values,
        default=None,
        metavar="CSV",
        help="Backtest exchanges to use, for example bybit or binance,bybit.",
    )
    parser.add_argument(
        "--start-date",
        "-sd",
        dest="backtest.start_date",
        type=str,
        default=None,
        metavar="DATE",
        help="Backtest start date. Examples: 2025, 2025-01, 2025-01-15.",
    )
    parser.add_argument(
        "--end-date",
        "-ed",
        dest="backtest.end_date",
        type=str,
        default=None,
        metavar="DATE",
        help='Backtest end date. Use "-ed now" for the latest available candles.',
    )
    args = parser.parse_args()
    source_config, base_config_path, raw_snapshot = load_input_config(args.config_path)
    update_config_with_args(
        source_config,
        args,
        allowed_keys={
            *side_source_keys,
            "approved_coins",
            "ignored_coins",
            'bot.long.universe.minimum_coin_age_days',
            'bot.short.universe.minimum_coin_age_days',
            "backtest.exchanges",
            "backtest.start_date",
            "backtest.end_date",
            "backtest.offline",
            "backtest.acquisition_mode",
        },
    )
    config = prepare_config(
        source_config,
        base_config_path=base_config_path,
        verbose=False,
        raw_snapshot=raw_snapshot,
    )
    if args.plan:
        from simulation_preparation import build_preparation_plan
        target = args.target_cuda_free_gib
        if target is not None and (not math.isfinite(target) or target <= 0):
            parser.error("--target-cuda-free-gib must be finite and positive")
        # The legacy download path warms the base request. Suite-complete,
        # bound-aware preparation is selected explicitly with --for-optimize.
        plan_config = config
        if not args.for_optimize:
            from copy import deepcopy
            plan_config = deepcopy(config)
            plan_config["backtest"]["suite_enabled"] = False
        report = build_preparation_plan(plan_config, for_optimize=args.for_optimize,
            target_cuda_free_bytes=None if target is None else int(target * 2**30))
        print(json.dumps(report, indent=2))
    else:
        if args.target_cuda_free_gib is not None:
            parser.error("--target-cuda-free-gib requires --plan")
        report = await warm_ohlcv_caches(config, for_optimize=args.for_optimize)
    if args.manifest is not None:
        from utils import _atomic_write_json
        _atomic_write_json(str(args.manifest), report, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
