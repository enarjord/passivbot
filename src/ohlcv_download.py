import argparse
import asyncio
import logging

from cli_utils import get_cli_prog
from config import load_input_config, prepare_config
from config.access import require_config_value, require_live_value
from config.schema import get_template_config
from config_utils import add_config_arguments, update_config_with_args
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
    template_config = get_template_config()
    del template_config["optimize"]
    del template_config["bot"]
    template_config["live"] = {
        k: v
        for k, v in template_config["live"].items()
        if k in {"approved_coins", "ignored_coins"}
    }
    template_config["backtest"] = {
        k: v
        for k, v in template_config["backtest"].items()
        if k in {"combine_ohlcvs", "end_date", "start_date", "exchanges"}
    }
    add_config_arguments(parser, template_config)
    args = parser.parse_args()
    source_config, base_config_path, raw_snapshot = load_input_config(args.config_path)
    update_config_with_args(source_config, args)
    config = prepare_config(
        source_config,
        base_config_path=base_config_path,
        verbose=False,
        raw_snapshot=raw_snapshot,
    )
    await warm_ohlcv_caches(config)


if __name__ == "__main__":
    asyncio.run(main())
