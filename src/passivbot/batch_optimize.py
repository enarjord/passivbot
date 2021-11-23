from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess

from passivbot.utils.procedures import make_get_filepath

log = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tokens = [
        "BTS",
        "LTC",
        "STORJ",
        "BAT",
        "DASH",
        "SOL",
        "AVAX",
        "LUNA",
        "DYDX",
        "COMP",
        "FIL",
        "LINK",
        "MATIC",
        "LIT",
        "NEO",
        "OMG",
        "XRP",
        "HBAR",
        "MANA",
        "IOTA",
        "ADA",
        "QTUM",
        "SXP",
        "XEM",
        "EOS",
        "XMR",
        "ETC",
        "XLM",
        "MKR",
        "BNB",
        "AAVE",
        "ALGO",
        "TRX",
        "ZEC",
        "XTZ",
        "BCH",
    ]
    start_from = "BTS"
    symbols = tokens[tokens.index(start_from) :] + tokens[: tokens.index(start_from)]

    quote = "USDT"
    cfgs_dir = make_get_filepath("cfgs_batch_optimize/")
    exchange = "binance"

    symbols = [e + quote for e in symbols]
    kwargs_list = [
        {
            "start": cfgs_dir,
            "symbol": symbol,
            # 'starting_balance': 10000.0,
            # 'end_date': '2021-09-20T15:00',
            # 'start_date': '2021-03-01',
        }
        for symbol in symbols
    ]
    passivbot_cli_path = shutil.which("passivbot")
    for kwargs in kwargs_list:
        cmd_args = [passivbot_cli_path, "optimize"]
        for key in kwargs:
            cmd_args.extend([f"--{key}", f"{kwargs[key]}"])
        log.info("command: %s", cmd_args)
        subprocess.run(cmd_args, shell=False, check=True)
        try:
            d = f'backtests/{exchange}/{kwargs["symbol"]}/plots/'
            ds = sorted(f for f in os.listdir(d) if "20" in f)
            for d1 in ds:
                log.info(f"copying resulting config to {cfgs_dir}: %s", d + d1)
                shutil.copy(d + d1 + "/live_config.json", f'{cfgs_dir}{kwargs["symbol"]}_{d1}.json')
        except Exception as e:
            log.error("Error: %s %s", kwargs["symbol"], e)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(func=main)
