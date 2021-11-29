from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import pathlib
import signal
import time
from typing import cast

import dateutil.parser
import numpy as np
import pandas as pd

from passivbot.datastructures.config import BacktestNamedConfig
from passivbot.datastructures.config import DownloaderNamedConfig
from passivbot.downloader import Downloader
from passivbot.exceptions import PassivBotSystemExit
from passivbot.utils.funcs.njit import BacktestFill
from passivbot.utils.funcs.njit import BacktestStat
from passivbot.utils.funcs.njit import njit_backtest
from passivbot.utils.funcs.njit import round_
from passivbot.utils.funcs.pure import analyze_fills
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.plotting import dump_plots
from passivbot.utils.procedures import add_backtesting_argparse_args
from passivbot.utils.procedures import dump_live_config
from passivbot.utils.procedures import load_hjson_config
from passivbot.utils.procedures import post_process_backtesting_argparse_parsed_args

log = logging.getLogger(__name__)


class Backtester:
    def __init__(self, config: BacktestNamedConfig):
        self.config = config
        self.downloader = Downloader(cast(DownloaderNamedConfig, config), download_only=False)

    async def _on_signal(self, signum, loop):
        if signum == signal.SIGINT:
            signame = "SIGINT"
        else:
            signame = "SIGTERM"
        log.info("Caught %s signal", signame)
        # Ignore the signal, since we've handled it already
        signal.signal(signum, signal.SIG_IGN)
        await self.await_closed()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks)
        loop.stop()

    async def _init(self):
        await self.downloader._init()
        if self.config.api_key.exchange == "binance":
            if self.downloader.rtc.spot:
                self.maker_fee = 0.001
                self.taker_fee = 0.001
            else:
                self.maker_fee = 0.0002
                self.taker_fee = 0.0004
        elif self.config.api_key.exchange == "bybit":
            if self.downloader.rtc.spot:
                raise PassivBotSystemExit("spot not implemented on bybit")
            self.maker_fee = -0.00025
            self.taker_fee = 0.00075
        assert self.config.parent.backtests_dir
        self.plots_dirpath = self.config.parent.backtests_dir.joinpath(
            self.config.api_key.exchange,
            self.config.symbol.name,
            "plots",
            f"{ts_to_date(time.time())[:19].replace(':', '')}",
        )
        self.plots_dirpath.mkdir(parents=True, exist_ok=True)

    async def await_closed(self):
        await self.downloader.await_closed()

    async def run(self):
        loop = asyncio.get_event_loop()

        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                signum, lambda signum=signum: asyncio.create_task(self._on_signal(signum, loop))
            )
        try:
            await self._init()
            data: np.ndarray = await self.downloader.get_sampled_ticks()
            self.n_days = round_((data[-1][0] - data[0][0]) / (1000 * 60 * 60 * 24), 0.1)
            log.info("n_days: %s", self.n_days)
            log.info("starting_balance: %s", self.config.parent.starting_balance)
            self.plot_wrap(data)
        finally:
            await self.await_closed()

    def backtest(
        self, data: np.ndarray, do_print=False
    ) -> tuple[list[BacktestFill], list[BacktestStat]]:
        log.info("backtesting...")
        sts = time.time()
        try:
            return njit_backtest(
                ticks=data,
                starting_balance=self.config.parent.starting_balance,
                latency_simulation_ms=self.config.parent.latency_simulation_ms,
                maker_fee=self.maker_fee,
                spot=self.downloader.rtc.spot,
                hedge_mode=self.downloader.rtc.hedge_mode,
                inverse=self.downloader.rtc.inverse,
                do_long=self.downloader.rtc.do_long,
                do_short=self.downloader.rtc.do_short,
                qty_step=self.downloader.rtc.qty_step,
                price_step=self.downloader.rtc.price_step,
                min_qty=self.downloader.rtc.min_qty,
                min_cost=self.downloader.rtc.min_cost,
                c_mult=self.downloader.rtc.c_mult,
                grid_span=(self.config.long.grid_span, self.config.short.grid_span),
                wallet_exposure_limit=(
                    self.config.long.wallet_exposure_limit,
                    self.config.short.wallet_exposure_limit,
                ),
                max_n_entry_orders=(
                    self.config.long.max_n_entry_orders,
                    self.config.short.max_n_entry_orders,
                ),
                initial_qty_pct=(
                    self.config.long.initial_qty_pct,
                    self.config.short.initial_qty_pct,
                ),
                eprice_pprice_diff=(
                    self.config.long.eprice_pprice_diff,
                    self.config.short.eprice_pprice_diff,
                ),
                secondary_allocation=(
                    self.config.long.secondary_allocation,
                    self.config.short.secondary_allocation,
                ),
                secondary_pprice_diff=(
                    self.config.long.secondary_pprice_diff,
                    self.config.short.secondary_pprice_diff,
                ),
                eprice_exp_base=(
                    self.config.long.eprice_exp_base,
                    self.config.short.eprice_exp_base,
                ),
                min_markup=(self.config.long.min_markup, self.config.short.min_markup),
                markup_range=(self.config.long.markup_range, self.config.short.markup_range),
                n_close_orders=(self.config.long.n_close_orders, self.config.short.n_close_orders),
            )
        finally:
            log.info(f"{time.time() - sts:.2f} seconds elapsed")

    def plot_wrap(self, data: np.array) -> None:
        fills, stats = self.backtest(data, do_print=True)
        if not fills:
            log.info("no fills")
            return
        fdf, sdf, result = analyze_fills(
            fills=fills,
            stats=stats,
            inverse=self.downloader.rtc.inverse,
            c_mult=self.downloader.rtc.c_mult,
            exchange=self.config.api_key.exchange,
            symbol=self.config.symbol.name,
        )
        results_path = self.plots_dirpath / "result.json"
        try:
            rel_results_path = results_path.relative_to(self.config.parent.basedir)
        except ValueError:
            rel_results_path = results_path
        log.info("Writing %s", rel_results_path)
        results_path.write_text(json.dumps(result, indent=4))

        fills_path = self.plots_dirpath / "fills.csv"
        try:
            rel_fills_path = fills_path.relative_to(self.config.parent.basedir)
        except ValueError:
            rel_fills_path = fills_path
        log.info("Writing %s", rel_fills_path)
        fdf.to_csv(fills_path)

        stats_path = self.plots_dirpath / "stats.csv"
        try:
            rel_stats_path = stats_path.relative_to(self.config.parent.basedir)
        except ValueError:
            rel_stats_path = stats_path
        log.info("Writing %s", rel_stats_path)
        sdf.to_csv(stats_path)

        df = pd.DataFrame(
            {**{"timestamp": data[:, 0], "qty": data[:, 1], "price": data[:, 2]}, **{}}
        )
        log.info("dumping plots...")
        dump_plots(
            self.plots_dirpath,
            result,
            fdf,
            sdf,
            df,
            exchange=self.config.api_key.exchange,
            symbol=self.config.symbol.name,
            market_type=self.config.market_type,
            starting_balance=self.config.parent.starting_balance,
            n_days=self.n_days,
            do_long=self.config.long.enabled,
            do_short=self.config.short.enabled,
        )

        config = self.config.dict()
        config.update(result)
        live_config_path = self.plots_dirpath / "live_config.json"
        try:
            rel_live_config_path = live_config_path.relative_to(self.config.parent.basedir)
        except ValueError:
            rel_live_config_path = live_config_path
        log.info("Writing %s", rel_live_config_path)
        dump_live_config(config, live_config_path)


async def _main(config: BacktestNamedConfig) -> None:
    backtester = Backtester(config)
    await backtester.run()


def main(config: BacktestNamedConfig) -> None:
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(_main(config))
    except asyncio.CancelledError:
        pass
    except PassivBotSystemExit:
        raise
    except Exception as e:
        log.error("There was an error starting the bot: %s", e, exc_info=True)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "backtest_config_path", type=pathlib.Path, default=None, help="Path to backtest config file"
    )
    add_backtesting_argparse_args(parser)
    parser.add_argument(
        "--sb",
        "--starting-balance",
        dest="starting_balance",
        type=float,
        help="Starting balance for the backtest, overriding value from backtest config file",
    )
    parser.set_defaults(func=main)


def process_argparse_parsed_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    pass


def post_process_argparse_parsed_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: BacktestNamedConfig
) -> None:
    if args.backtest_config_path:
        backtest_config = load_hjson_config(args.backtest_config_path)
        for key, value in backtest_config.items():
            if key == "market_type" and not args.market_type:
                log.info("Overriding market type with: %s", value)
                args.market_type = value
            if key == "start_date" and not args.start_date:
                value = dateutil.parser.parse(value).replace(tzinfo=datetime.timezone.utc)
                log.info("Overriding start date with: %s", value)
                config.parent.start_date = value
            if key == "end_date" and not args.end_date:
                value = dateutil.parser.parse(value).replace(tzinfo=datetime.timezone.utc)
                log.info("Overriding end date with: %s", value)
                config.parent.end_date = value
            if key == "user" and not args.key_name:
                log.warning("'user' is not longer supported. Using it's value as 'key_name")
                key = "key_name"
            if key == "key_name" and not args.key_name:
                log.info("Overriding key name with: %s", value)
                args.key_name = value
            if key == "symbol" and not args.symbol:
                log.info("Overriding symbol with: %s", value)
                args.symbol = value
            if key == "starting_balance" and not args.starting_balance:
                log.info("Overriding starting balance with: %s", value)
                args.starting_balance = value
            if key == "latency_simulation_ms":
                log.info("Overriding latency simulation with: %sms", value)
                config.parent.latency_simulation_ms = value

    if args.starting_balance:
        config.parent.starting_balance = args.starting_balance

    post_process_backtesting_argparse_parsed_args(parser, args, config)

    log.info("Selected market type: %s", config.market_type)
    log.info(
        "Selected configuration for symbol %r:\n%s",
        config.symbol,
        config.json(indent=2),
    )
    log.info(
        "Backtest configuration:\n%s",
        config.parent.json(
            indent=2,
            exclude={
                "long": ...,
                "short": ...,
                "api_keys": ...,
                "logging": ...,
                "configs": ...,
                "symbols": ...,
            },
        ),
    )
