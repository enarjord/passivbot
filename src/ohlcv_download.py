import argparse
import asyncio
import logging

from backtest import prepare_hlcvs_mss
from cli_utils import get_cli_prog
from config import load_input_config, prepare_config
from config.access import require_config_value, require_live_value
from config_utils import comma_separated_values, update_config_with_args
from pure_funcs import str2bool
from utils import format_approved_ignored_coins


async def warm_ohlcv_caches(config: dict, *, force_refetch_gaps: bool = False) -> None:
    exchanges = require_config_value(config, "backtest.exchanges")
    await format_approved_ignored_coins(config, exchanges)
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
        return
    for exchange in exchanges:
        coins, _hlcvs, _mss, _results_path, cache_dir, _btc, _timestamps = await prepare_hlcvs_mss(
            config,
            exchange,
            force_refetch_gaps=force_refetch_gaps,
        )
        config["backtest"]["coins"][exchange] = coins
        config["backtest"]["cache_dir"][exchange] = str(cache_dir)
        logging.info("download materialized %s HLCV dataset at %s", exchange, cache_dir)


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
    parser.add_argument(
        "--hlcvs-cache-permissive",
        dest="backtest.hlcvs_cache_permissive",
        type=str2bool,
        default=None,
        metavar="Y/N",
        help="Allow legacy final HLCV caches without manifests to load with warning-only compatibility behavior.",
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
            "backtest.hlcvs_cache_permissive",
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
