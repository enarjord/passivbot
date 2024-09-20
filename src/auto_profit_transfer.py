import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import traceback
import json
import argparse
import asyncio
from procedures import (
    create_binance_bot,
    create_bybit_bot,
    make_get_filepath,
    load_exchange_key_secret_passphrase,
    load_user_info,
    utc_ms,
)
from passivbot import setup_bot
from pure_funcs import get_template_live_config, flatten
from njit_funcs import round_dynamic
from time import sleep
import pprint
import logging
import logging.config


async def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        prog="auto profit transfer",
        description="automatically transfer percentage of profits from futures wallet to spot wallet",
    )
    parser.add_argument("user", type=str, help="user/account_name defined in api-keys.json")
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        required=False,
        default=0.5,
        dest="percentage",
        help="per uno, i.e. 0.02==2 per cent.  default=0.5",
    )
    parser.add_argument(
        "-q",
        "--quote",
        type=str,
        required=False,
        default="USDT",
        dest="quote",
        help="quote coin, default USDT",
    )
    args = parser.parse_args()
    user_info = load_user_info(args.user)
    exchange = user_info["exchange"]
    pnls_fname = os.path.join("caches", exchange, args.user + "_pnls.json")
    transfer_log_fpath = make_get_filepath(
        os.path.join("logs", f"automatic_profit_transfer_log_{exchange}_{args.user}.json")
    )
    try:
        already_transferred_ids = set(json.load(open(transfer_log_fpath)))
        logging.info(f"loaded already transferred IDs: {transfer_log_fpath}")
    except:
        already_transferred_ids = set()
        logging.info(f"no previous transfers to load")
    if exchange == "bybit":
        config = get_template_live_config("v7")
        config["user"] = args.user
        bot = setup_bot(config)
        await bot.determine_utc_offset()
    else:
        raise Exception(f"unsupported exchange {exchange}")
    day_ms = 1000 * 60 * 60 * 24
    sleep_time_error = 10
    while True:
        try:
            if os.path.exists(pnls_fname):
                pnls = json.load(open(f"caches/{user_info['exchange']}/{args.user}_pnls.json"))
            else:
                logging.info(f"pnls file does not exist {pnls_fname}")
                pnls = []
            now = bot.get_exchange_time()
            pnls_last_24_h = [x for x in pnls if x["timestamp"] >= now - day_ms]
            pnls_last_24_h = [x for x in pnls_last_24_h if x["id"] not in already_transferred_ids]
            profit = sum([e["pnl"] for e in pnls_last_24_h])
            to_transfer = round_dynamic(profit * args.percentage, 4)
            if args.quote in ["USDT", "BUSD", "USDC"]:
                to_transfer = round(to_transfer, 4)
            if to_transfer > 0:
                try:
                    transferred = await bot.cca.transfer(args.quote, to_transfer, "CONTRACT", "SPOT")
                    logging.info(f"pnl: {profit} transferred {to_transfer} {args.quote}")
                    logging.info(f"{transferred}")
                    already_transferred_ids.update([e["id"] for e in pnls_last_24_h])
                    json.dump(list(already_transferred_ids), open(transfer_log_fpath, "w"))
                except Exception as e:
                    logging.error(f"failed transferring {e}")
                    traceback.print_exc()
            else:
                logging.info("nothing to transfer")
            sleep(60 * 60)
        except Exception as e:
            logging.info(f"error with profit transfer {e}")
            logging.info(f"trying again in {sleep_time_error} minutes")
            sleep(60 * sleep_time_error)


if __name__ == "__main__":
    asyncio.run(main())
