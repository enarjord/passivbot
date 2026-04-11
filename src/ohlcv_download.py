import argparse
import asyncio
import logging

from cli_utils import get_cli_prog
from config import load_input_config, prepare_config
from config.access import require_config_value, require_live_value
from config_utils import comma_separated_values, update_config_with_args
from hlcv_preparation import HLCVManager
from utils import format_approved_ignored_coins


async def warm_ohlcv_caches(config: dict, *, force_refetch_gaps: bool = False) -> None:
    exchanges = require_config_value(config, "backtest.exchanges")
    await format_approved_ignored_coins(config, exchanges)

    managers = {}
    try:
        for exchange in exchanges:
            managers[exchange] = HLCVManager(
                exchange,
                require_config_value(config, "backtest.start_date"),
                require_config_value(config, "backtest.end_date"),
                gap_tolerance_ohlcvs_minutes=require_config_value(
                    config, "backtest.gap_tolerance_ohlcvs_minutes"
                ),
                cm_debug_level=int(config.get("backtest", {}).get("cm_debug_level", 0) or 0),
                cm_progress_log_interval_seconds=float(
                    config.get("backtest", {}).get("cm_progress_log_interval_seconds", 10.0) or 10.0
                ),
                force_refetch_gaps=force_refetch_gaps,
                ohlcv_source_dir=config.get("backtest", {}).get("ohlcv_source_dir"),
            )
        logging.info("loading markets for %s", exchanges)
        await asyncio.gather(*[managers[exchange].load_markets() for exchange in managers])

        approved = require_live_value(config, "approved_coins")
        coins = sorted({coin for side_coins in approved.values() for coin in side_coins})
        for coin in coins:
            tasks = [asyncio.create_task(managers[exchange].get_ohlcvs(coin)) for exchange in managers]
            await asyncio.gather(*tasks)
    finally:
        for manager in managers.values():
            await manager.aclose()
            if manager.cc:
                await manager.cc.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog=get_cli_prog("download"), description="download ohlcv data"
    )
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    parser.add_argument(
        "--symbols",
        "-s",
        dest="live.approved_coins",
        type=str,
        default=None,
        metavar="CSV_OR_PATH",
        help=(
            "Approved coins. Use CSV like BTC,ETH,XRP, the literal 'all', a path to a JSON "
            "coin list file, or a JSON/HJSON per-side object like "
            '{"long":["BTC"],"short":"all"}. Use coin tickers, not exchange symbols.'
        ),
    )
    parser.add_argument(
        "--ignored-coins",
        "-ic",
        dest="live.ignored_coins",
        type=str,
        default=None,
        metavar="CSV_OR_PATH",
        help="Ignored coins. Comma-separated coins or path to a JSON coin list file.",
    )
    parser.add_argument(
        "--minimum-coin-age-days",
        "-mcad",
        dest="live.minimum_coin_age_days",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Minimum coin age in days required before a coin is eligible to trade.",
    )
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
            "live.approved_coins",
            "live.ignored_coins",
            "live.minimum_coin_age_days",
            "backtest.exchanges",
            "backtest.start_date",
            "backtest.end_date",
        },
    )
    config = prepare_config(
        source_config,
        base_config_path=base_config_path,
        verbose=False,
        raw_snapshot=raw_snapshot,
    )
    await warm_ohlcv_caches(config)


if __name__ == "__main__":
    asyncio.run(main())
