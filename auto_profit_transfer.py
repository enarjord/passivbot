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
)
from pure_funcs import get_template_live_config, flatten
from njit_funcs import round_dynamic
from time import sleep
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
        help="per uno, i.e. 0.02==2%.  default=0.5",
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
    config = get_template_live_config()
    config["user"] = args.user
    config["symbol"] = "BTCUSDT"  # dummy symbol
    config["market_type"] = "futures"
    exchange = load_exchange_key_secret_passphrase(args.user)[0]
    if exchange == "binance":
        bot = await create_binance_bot(config)
    elif exchange == "bybit":
        bot = await create_bybit_bot(config)
    else:
        raise Exception(f"unknown exchange {exchange}")
    transfer_log_fpath = make_get_filepath(
        os.path.join("logs", f"automatic_profit_transfer_log_{exchange}_{config['user']}.json")
    )
    try:
        already_transferred_ids = set(json.load(open(transfer_log_fpath)))
        logging.info(f"loaded already transferred IDs: {transfer_log_fpath}")
    except:
        already_transferred_ids = set()
        logging.info(f"no previous transfers to load")
    while True:
        now = await bot.get_server_time()
        try:
            income = await bot.get_all_income(start_time=now - 1000 * 60 * 60 * 24)
        except Exception as e:
            logging.error(f"failed fetching income {e}")
            traceback.print_exc()
            income = []
        income = [e for e in income if e["transaction_id"] not in already_transferred_ids]
        income = [e for e in income if e["token"] == args.quote]
        profit = sum([e["income"] for e in income])
        to_transfer = round_dynamic(profit * args.percentage, 4)
        if args.quote in ["USDT", "BUSD", "USDC"]:
            to_transfer = round(to_transfer, 4)
        if to_transfer > 0:
            try:
                transferred = await bot.transfer_from_derivatives_to_spot(args.quote, to_transfer)
                logging.info(f"income: {profit} transferred {to_transfer} {args.quote}")
                if exchange == "bybit":
                    if "ret_msg" not in transferred or transferred["ret_msg"] != "OK":
                        print(f"error with transfer {transferred}")
                        continue
                logging.info(f"{transferred}")
                already_transferred_ids.update([e["transaction_id"] for e in income])
                json.dump(list(already_transferred_ids), open(transfer_log_fpath, "w"))
            except Exception as e:
                logging.error(f"failed transferring {e}")
                traceback.print_exc()
        else:
            logging.info("nothing to transfer")
        sleep(60 * 60)


if __name__ == "__main__":
    asyncio.run(main())
