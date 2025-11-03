import json
import re
import os
from typing import Callable, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import time
from colorama import init, Fore
from prettytable import PrettyTable
from config_utils import dump_config
from utils import make_get_filepath
from pure_funcs import denumpyize, ts_to_date
import passivbot_rust as pbr


plt.rcParams["figure.figsize"] = [21, 13]

try:  # pragma: no cover
    from IPython.display import display as _ipy_display
except Exception:  # pragma: no cover
    _ipy_display = None


def plot_two_series_shared_x(
    df: pd.DataFrame, upper_column: str, lower_column: str, *, title: str = ""
):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    df[upper_column].plot(ax=axes[0])
    axes[0].set_ylabel(upper_column)
    df[lower_column].plot(ax=axes[1])
    axes[1].set_ylabel(lower_column)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_two_series(series1: pd.Series, series2: pd.Series, *, title: str = ""):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    series1.plot(ax=axes[0])
    axes[0].set_ylabel(series1.name)
    series2.plot(ax=axes[1])
    axes[1].set_ylabel(series2.name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_table(result_):
    result = result_.copy()
    if "result" not in result:
        result["result"] = result
    table = PrettyTable(["Metric", "Value"])
    table.align["Metric"] = "l"
    table.align["Value"] = "l"
    table.title = "Summary"

    table.add_row(["Exchange", result["exchange"] if "exchange" in result else "unknown"])
    table.add_row(["Market type", result["market_type"] if "market_type" in result else "unknown"])
    table.add_row(["Symbol", result["symbol"] if "symbol" in result else "unknown"])
    table.add_row(
        ["Passivbot mode", result["passivbot_mode"] if "passivbot_mode" in result else "unknown"]
    )
    table.add_row(
        [
            "ADG n subdivisions",
            result["adg_n_subdivisions"] if "adg_n_subdivisions" in result else "unknown",
        ]
    )
    table.add_row(["No. days", pbr.round_dynamic(result["result"]["n_days"], 2)])
    table.add_row(["Starting balance", pbr.round_dynamic(result["result"]["starting_balance"], 6)])
    for side in ["long", "short"]:
        if side not in result:
            result[side] = {"enabled": result[f"do_{side}"]}
        if result[side]["enabled"]:
            table.add_row([" ", " "])
            table.add_row([side.capitalize(), True])
            profit_color = (
                Fore.RED
                if f"final_balance_{side}" in result["result"]
                and result["result"][f"final_balance_{side}"] < result["result"]["starting_balance"]
                else Fore.RESET
            )
            for title, key, precision, mul, suffix in [
                ("ADG per exposure", f"adg_per_exposure_{side}", 3, 100, "%"),
                (
                    "ADG weighted per exposure",
                    f"adg_weighted_per_exposure_{side}",
                    3,
                    100,
                    "%",
                ),
                ("Max drawdown", f"drawdown_max_{side}", 5, 100, "%"),
                (
                    "Drawdown mean of 1% worst (hourly)",
                    f"drawdown_1pct_worst_mean_{side}",
                    5,
                    100,
                    "%",
                ),
                ("Sharpe Ratio (daily)", f"sharpe_ratio_{side}", 6, 1, ""),
                ("Loss to profit ratio", f"loss_profit_ratio_{side}", 4, 1, ""),
                (
                    "P.A. distance mean of 1% worst (hourly)",
                    f"pa_distance_1pct_worst_mean_{side}",
                    6,
                    1,
                    "",
                ),
                ("#newline", "", 0, 0, ""),
                ("Final balance", f"final_balance_{side}", 6, 1, ""),
                ("Final equity", f"final_equity_{side}", 6, 1, ""),
                ("Net PNL + fees", f"net_pnl_plus_fees_{side}", 6, 1, ""),
                ("Net Total gain", f"gain_{side}", 4, 100, "%"),
                ("Average daily gain", f"adg_{side}", 3, 100, "%"),
                ("Average daily gain weighted", f"adg_weighted_{side}", 3, 100, "%"),
                ("Exposure ratios mean", f"exposure_ratios_mean_{side}", 5, 1, ""),
                ("Price action distance mean", f"pa_distance_mean_{side}", 6, 1, ""),
                ("Price action distance std", f"pa_distance_std_{side}", 6, 1, ""),
                ("Price action distance max", f"pa_distance_max_{side}", 6, 1, ""),
                ("Closest bankruptcy", f"closest_bkr_{side}", 4, 100, "%"),
                ("Lowest equity/balance ratio", f"eqbal_ratio_min_{side}", 4, 1, ""),
                (
                    "Mean of 10 worst eq/bal ratios (hourly)",
                    f"eqbal_ratio_mean_of_10_worst_{side}",
                    4,
                    1,
                    "",
                ),
                ("Equity/balance ratio std", f"equity_balance_ratio_std_{side}", 4, 1, ""),
                ("Ratio of time spent at max exposure", f"time_at_max_exposure_{side}", 4, 1, ""),
            ]:
                if key in result["result"]:
                    val = pbr.round_dynamic(result["result"][key] * mul, precision)
                    table.add_row(
                        [
                            title,
                            f"{profit_color}{val}{suffix}{Fore.RESET}",
                        ]
                    )
                elif title == "#newline":
                    table.add_row([" ", " "])
            for title, key in [
                ("No. fills", f"n_fills_{side}"),
                ("No. entries", f"n_entries_{side}"),
                ("No. closes", f"n_closes_{side}"),
                ("No. initial entries", f"n_ientries_{side}"),
                ("No. reentries", f"n_rentries_{side}"),
                ("No. unstuck/EMA entries", f"n_unstuck_entries_{side}"),
                ("No. unstuck/EMA closes", f"n_unstuck_closes_{side}"),
                ("No. normal closes", f"n_normal_closes_{side}"),
            ]:
                if key in result["result"]:
                    table.add_row([title, result["result"][key]])
            for title, key, precision in [
                ("Average n fills per day", f"avg_fills_per_day_{side}", 3),
                ("Mean hours stuck", f"hrs_stuck_avg_{side}", 6),
                ("Max hours stuck", f"hrs_stuck_max_{side}", 6),
            ]:
                if key in result["result"]:
                    table.add_row([title, pbr.round_dynamic(result["result"][key], precision)])

            if f"pnl_sum_{side}" in result["result"]:
                profit_color = Fore.RED if result["result"][f"pnl_sum_{side}"] < 0 else Fore.RESET

                table.add_row(
                    [
                        "PNL sum",
                        f"{profit_color}{pbr.round_dynamic(result['result'][f'pnl_sum_{side}'], 4)}{Fore.RESET}",
                    ]
                )
            for title, key, precision in [
                ("Profit sum", f"profit_sum_{side}", 4),
                ("Loss sum", f"loss_sum_{side}", 4),
                ("Fee sum", f"fee_sum_{side}", 4),
                ("Biggest pos cost", f"biggest_psize_quote_{side}", 4),
                ("Volume quote", f"volume_quote_{side}", 6),
                ("Biggest pos size", f"biggest_psize_{side}", 3),
            ]:
                if key in result["result"]:
                    table.add_row([title, pbr.round_dynamic(result["result"][key], precision)])
    return table


def dump_plots(
    result: dict,
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    sdf: pd.DataFrame,
    df: pd.DataFrame,
    n_parts: int = None,
    disable_plotting: bool = False,
):
    init(autoreset=True)
    plt.rcParams["figure.figsize"] = [21, 13]
    try:
        pd.set_option("display.precision", 10)
    except Exception as e:
        print("error setting pandas precision", e)

    result["plots_dirpath"] = make_get_filepath(
        os.path.join(result["plots_dirpath"], f"{ts_to_date(time.time())[:19].replace(':', '')}", "")
    )
    # sdf = sdf.set_index(pd.to_datetime(pd.to_datetime(sdf.timestamp * 1000 * 1000)))
    # longs = longs.set_index(pd.to_datetime(pd.to_datetime(sdf.timestamp * 1000 * 1000)))
    # shorts = shorts.set_index(pd.to_datetime(pd.to_datetime(sdf.timestamp * 1000 * 1000)))
    longs.to_csv(result["plots_dirpath"] + "fills_long.csv")
    shorts.to_csv(result["plots_dirpath"] + "fills_short.csv")
    sdf.to_csv(result["plots_dirpath"] + "stats.csv")
    table = make_table(result)

    dump_config(result, result["plots_dirpath"] + "live_config.json")
    json.dump(denumpyize(result), open(result["plots_dirpath"] + "result.json", "w"), indent=4)

    print("writing backtest_result.txt...\n")
    with open(f"{result['plots_dirpath']}backtest_result.txt", "w") as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output))

    if disable_plotting:
        return
    n_parts = (
        n_parts
        if n_parts is not None
        else min(12, max(3, int(pbr.round_up(result["n_days"] / 14, 1.0))))
    )
    for side, fdf in [("long", longs), ("short", shorts)]:
        if result[side]["enabled"]:
            plt.clf()
            fig = plot_fills(df, fdf, plot_whole_df=True, title=f"Overview Fills {side.capitalize()}")
            if not fig:
                continue
            fig.savefig(f"{result['plots_dirpath']}whole_backtest_{side}.png")
            print(f"\nplotting balance and equity {side} {result['plots_dirpath']}...")
            plt.clf()
            sdf[f"balance_{side}"].plot()
            sdf[f"equity_{side}"].plot(
                title=f"Balance and equity {side.capitalize()}", xlabel="Time", ylabel="Balance"
            )
            plt.savefig(f"{result['plots_dirpath']}balance_and_equity_sampled_{side}.png")

            if result["passivbot_mode"] == "clock":
                spans = sorted(
                    [
                        result[side]["ema_span_0"],
                        (result[side]["ema_span_0"] * result[side]["ema_span_1"]) ** 0.5,
                        result[side]["ema_span_1"],
                    ]
                )
                emas = pd.DataFrame(
                    {f"ema_{span}": df.price.ewm(span=span, adjust=False).mean() for span in spans},
                    index=df.index,
                )
                ema_dist_lower = result[side][
                    "ema_dist_entry" if side == "long" else "ema_dist_close"
                ]
                ema_dist_upper = result[side][
                    "ema_dist_entry" if side == "short" else "ema_dist_close"
                ]
                if abs(ema_dist_lower) < 0.1:
                    df = df.join(
                        pd.DataFrame(
                            {"ema_band_lower": emas.min(axis=1) * (1 - ema_dist_lower)},
                            index=df.index,
                        )
                    )
                if abs(ema_dist_upper) < 0.1:
                    df = df.join(
                        pd.DataFrame(
                            {"ema_band_upper": emas.max(axis=1) * (1 + ema_dist_upper)},
                            index=df.index,
                        )
                    )
            for z in range(n_parts):
                start_ = z / n_parts
                end_ = (z + 1) / n_parts
                print(f"{side} {z} of {n_parts} {start_ * 100:.2f}% to {end_ * 100:.2f}%")
                fig = plot_fills(
                    df,
                    fdf.iloc[int(len(fdf) * start_) : int(len(fdf) * end_)],
                    title=f"Fills {side} {z+1} of {n_parts}",
                )
                if fig is not None:
                    fig.savefig(f"{result['plots_dirpath']}backtest_{side}{z + 1}of{n_parts}.png")
                else:
                    print(f"no {side} fills...")
            if result["passivbot_mode"] == "clock":
                if "ema_band_lower" in df.columns:
                    df = df.drop(["ema_band_lower"], axis=1)
                if "ema_band_upper" in df.columns:
                    df = df.drop(["ema_band_upper"], axis=1)

    print("plotting wallet exposures...")
    plt.clf()
    sdf.wallet_exposure_short = sdf.wallet_exposure_short.abs() * -1
    sdf[["wallet_exposure_long", "wallet_exposure_short"]].plot(
        title="Wallet exposures: +long, -short",
        xlabel="Time",
        ylabel="Wallet Exposure",
    )
    plt.savefig(f"{result['plots_dirpath']}wallet_exposures_plot.png")


def plot_fills(df, fdf_, side: int = 0, plot_whole_df: bool = False, title=""):
    if fdf_.empty:
        return None

    if df.index.name != "timestamp":
        dfc = df.set_index("timestamp")
    else:
        dfc = df

    if fdf_.index.name != "timestamp":
        fdf = fdf_.set_index("timestamp")
    else:
        fdf = fdf_

    if not plot_whole_df:
        start_ts = fdf.index[0]
        end_ts = fdf.index[-1]
        dfc = dfc.loc[start_ts:end_ts]

    fig, ax = plt.subplots()
    ax.plot(dfc.index, dfc["price"].values, "y-", label="price")
    if "ema_band_lower" in dfc.columns and "ema_band_upper" in dfc.columns:
        ax.plot(dfc.index, dfc["ema_band_lower"].values, "b--", label="ema_band_lower")
        ax.plot(dfc.index, dfc["ema_band_upper"].values, "r--", label="ema_band_upper")

    type_series = fdf["type"].astype(str)

    def _mask(series, substring):
        return series.str.contains(substring, regex=False)

    if side >= 0:
        longs = fdf[_mask(type_series, "long")]
        long_types = longs["type"].astype(str)

        mask_entry = _mask(long_types, "rentry") | _mask(long_types, "ientry")
        ax.scatter(longs.index[mask_entry], longs.loc[mask_entry, "price"], c="b", marker="o")

        mask_secondary = _mask(long_types, "secondary")
        ax.scatter(longs.index[mask_secondary], longs.loc[mask_secondary, "price"], c="g", marker="o")

        mask_nclose = long_types == "long_nclose"
        ax.scatter(longs.index[mask_nclose], longs.loc[mask_nclose, "price"], c="r", marker="o")

        mask_unstuck_entry = _mask(long_types, "unstuck_entry") | (long_types == "clock_entry_long")
        ax.scatter(
            longs.index[mask_unstuck_entry], longs.loc[mask_unstuck_entry, "price"], c="b", marker="x"
        )

        mask_unstuck_close = _mask(long_types, "unstuck_close") | (long_types == "clock_close_long")
        ax.scatter(
            longs.index[mask_unstuck_close], longs.loc[mask_unstuck_close, "price"], c="r", marker="x"
        )

        lppu = longs[(longs.pprice != longs.pprice.shift(1)) & (longs.pprice != 0.0)]
        if len(lppu) > 1:
            for idx_start, idx_end, price_val in zip(
                lppu.index[:-1],
                lppu.index[1:],
                lppu.pprice.iloc[:-1],
            ):
                ax.plot([idx_start, idx_end], [price_val, price_val], "b--")
    if side <= 0:
        shorts = fdf[_mask(type_series, "short")]
        short_types = shorts["type"].astype(str)

        mask_entry = _mask(short_types, "rentry") | _mask(short_types, "ientry")
        ax.scatter(shorts.index[mask_entry], shorts.loc[mask_entry, "price"], c="r", marker="o")

        mask_secondary = _mask(short_types, "secondary")
        ax.scatter(
            shorts.index[mask_secondary], shorts.loc[mask_secondary, "price"], c="g", marker="o"
        )

        mask_nclose = short_types == "short_nclose"
        ax.scatter(shorts.index[mask_nclose], shorts.loc[mask_nclose, "price"], c="b", marker="o")

        mask_unstuck_entry = _mask(short_types, "unstuck_entry") | (
            short_types == "clock_entry_short"
        )
        ax.scatter(
            shorts.index[mask_unstuck_entry],
            shorts.loc[mask_unstuck_entry, "price"],
            c="r",
            marker="x",
        )

        mask_unstuck_close = _mask(short_types, "unstuck_close") | (
            short_types == "clock_close_short"
        )
        ax.scatter(
            shorts.index[mask_unstuck_close],
            shorts.loc[mask_unstuck_close, "price"],
            c="b",
            marker="x",
        )

        sppu = shorts[(shorts.pprice != shorts.pprice.shift(1)) & (shorts.pprice != 0.0)]
        if len(sppu) > 1:
            for idx_start, idx_end, price_val in zip(
                sppu.index[:-1],
                sppu.index[1:],
                sppu.pprice.iloc[:-1],
            ):
                ax.plot([idx_start, idx_end], [price_val, price_val], "r--")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price + Fills")
    fig.tight_layout()

    return fig


def scale_array(xs, bottom, top):
    # Calculate the midpoint
    midpoint = (bottom + top) / 2

    # Scale the array
    scaled_xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))  # Scale between 0 and 1
    scaled_xs = (
        scaled_xs * (top - bottom) + midpoint - (top - bottom) / 2
    )  # Scale to the desired range and shift to the midpoint

    return scaled_xs


def plot_fills_long(hlcvs, fdf, coins, coin, start_pct=0.0, end_pct=1.0):
    start_idx = int(round(len(hlcvs) * start_pct))
    end_idx = int(round(len(hlcvs) * end_pct))
    coin_idx = coins.index(coin)
    cdf = pd.DataFrame(hlcvs[:, coin_idx, 2]).loc[start_idx:end_idx]
    fdfc = fdf[fdf.coin == coin].set_index("index").loc[start_idx:end_idx]
    longs = fdfc[fdfc.type.str.contains("long")]
    long_entries = longs[longs.qty > 0.0]
    long_closes = longs[longs.qty < 0.0]
    cdf.plot(style="y-")
    long_entries.price.plot(style="b.")
    long_closes.price.plot(style="r.")
    long_positions = longs[["pprice", "psize"]].copy()
    long_positions.loc[long_positions.psize == 0.0, "pprice"] = np.nan
    long_positions.pprice.plot(style="b--")
    return cdf, fdfc


def plot_fills_multi(symbol, sdf, fdf, start_pct=0.0, end_pct=1.0):
    plt.clf()
    start_minute = int(sdf.index[-1] * start_pct)
    end_minute = int(sdf.index[-1] * end_pct)
    sdfc = sdf.loc[start_minute:end_minute]
    fdfc = fdf.loc[start_minute:end_minute]
    fdfc = fdfc[fdfc.symbol == symbol]
    longs = fdfc[fdfc.type.str.contains("long")]
    shorts = fdfc[fdfc.type.str.contains("short")]

    ax = sdfc[f"{symbol}_price"].plot(style="y-")
    longs[longs.type.str.contains("entry")].price.plot(style="b.")
    longs[longs.type.str.contains("close")].price.plot(style="r.")
    sdfc[f"{symbol}_pprice_l"].plot(style="b--")

    shorts[shorts.type.str.contains("entry")].price.plot(style="mx")
    shorts[shorts.type.str.contains("close")].price.plot(style="cx")
    sdfc[f"{symbol}_pprice_s"].plot(style="r--")

    ax.legend(
        [
            "price",
            "entries_long",
            "closes_long",
            "pprices_long",
            "entries_short",
            "closes_short",
            "pprices_short",
        ]
    )

    return plt


def plot_pnls_long_short(sdf, fdf, start_pct=0.0, end_pct=1.0, symbol=None):
    plt.clf()
    start_minute = int(sdf.index[-1] * start_pct)
    end_minute = int(sdf.index[-1] * end_pct)
    fdfc = fdf.loc[start_minute:end_minute]
    if symbol is not None:
        fdfc = fdfc[fdfc.symbol == symbol]
    longs = fdfc[fdfc.type.str.contains("long")]
    shorts = fdfc[fdfc.type.str.contains("short")]
    ax = fdfc.pnl.cumsum().plot()
    longs.pnl.cumsum().plot()
    shorts.pnl.cumsum().plot()
    ax.legend(["pnl_sum", "pnl_long", "pnl_short"])
    return plt


def plot_pnls_separate(sdf, fdf, start_pct=0.0, end_pct=1.0, symbols=None):
    plt.clf()
    if symbols is None:
        symbols = [c[: c.find("_price")] for c in sdf.columns if "_price" in c]
    elif isinstance(symbols, str):
        symbols = [symbols]
    plt.clf()
    start_minute = int(sdf.index[-1] * start_pct)
    end_minute = int(sdf.index[-1] * end_pct)
    fdfc = fdf.loc[start_minute:end_minute]
    for symbol in symbols:
        ax = fdfc[fdfc.symbol == symbol].pnl.cumsum().plot()
    ax.legend(symbols)
    return plt


def plot_pnls_stuck(sdf, fdf, symbol=None, start_pct=0.0, end_pct=1.0, unstuck_threshold=0.9):
    plt.clf()
    symbols = [c[: c.find("_price")] for c in sdf.columns if "_price" in c]
    start_minute = int(sdf.index[-1] * start_pct)
    end_minute = int(sdf.index[-1] * end_pct)
    sdfc = sdf.loc[start_minute:end_minute]
    fdfc = fdf.loc[start_minute:end_minute]
    any_stuck = np.zeros(len(sdfc))
    for symbol in fdfc.symbol.unique():
        fdfcc = fdfc[(fdfc.symbol == symbol) & (fdfc.pnl < 0.0)]
        stuck_threshold_long = fdfcc[(fdfcc.type.str.contains("long"))].WE.mean() * 0.99
        stuck_threshold_short = fdfcc[(fdfcc.type.str.contains("short"))].WE.mean() * 0.99
        is_stuck_long = sdfc.loc[:, f"{symbol}_WE_l"] / stuck_threshold_long > unstuck_threshold
        is_stuck_short = sdfc.loc[:, f"{symbol}_WE_s"] / stuck_threshold_short > unstuck_threshold
        any_stuck = (
            pd.DataFrame({"0": any_stuck, "1": is_stuck_long.values, "2": is_stuck_short.values})
            .any(axis=1)
            .values
        )
    ax = sdfc.equity.plot()
    sdfc[any_stuck].balance.plot(style="r.")
    sdfc[~any_stuck].balance.plot(style="b.")
    ax.legend(["equity", "balance_with_any_stuck", "balance_with_none_stuck"])
    return plt


def plot_fills_forager(
    fdf: pd.DataFrame,
    hlcvs_df: pd.DataFrame,
    start_pct=0.0,
    end_pct=1.0,
    whole=False,
    clear: bool = True,
    *,
    stride: int = 1,
    fast: bool = False,
):
    if clear:
        plt.clf()
    if len(fdf) == 0:
        return
    if whole:
        hlcc = hlcvs_df[["high", "low", "close"]]
    else:
        hlcc = hlcvs_df[["high", "low", "close"]].loc[fdf.iloc[0].minute : fdf.iloc[-1].minute]
    fdfc = fdf.set_index(fdf.minute.astype(int))

    start_minute = int(hlcc.index[0] + hlcc.index[-1] * start_pct)
    end_minute = int(hlcc.index[0] + hlcc.index[-1] * end_pct)
    hlcc = hlcc[(hlcc.index >= start_minute) & (hlcc.index <= end_minute)]
    fdfc = fdfc[(fdfc.index >= start_minute) & (fdfc.index <= end_minute)]

    stride = max(1, int(stride)) if stride else 1
    if stride > 1:
        hlcc = hlcc.iloc[::stride]

    ax = plt.gca()
    hlcc_close = hlcc["close"].to_numpy()
    hlcc_low = hlcc["low"].to_numpy()
    hlcc_high = hlcc["high"].to_numpy()

    ax.plot(hlcc.index, hlcc_close, "y--", label="close", zorder=1.0)
    ax.plot(hlcc.index, hlcc_low, "g--", label="low", zorder=0.9)
    ax.plot(hlcc.index, hlcc_high, "g-.", alpha=0.6, label="high", zorder=0.8)

    type_series = fdfc["type"].astype(str)
    longs = fdfc[type_series.str.contains("long", regex=False)]
    shorts = fdfc[type_series.str.contains("short", regex=False)]
    if len(longs) == 0 and len(shorts) == 0:
        return plt
    legend = ["close", "low", "high"]
    if len(longs) > 0:
        longs_types = longs["type"].astype(str)
        longs_price_series = longs["price"]
        if fast and np.issubdtype(longs_price_series.dtype, np.number):
            longs_price = longs_price_series.to_numpy(copy=False)
        else:
            longs_price = pd.to_numeric(longs_price_series, errors="coerce").to_numpy()
        mask_entry = longs_types.str.contains("entry", regex=False).to_numpy()
        mask_close = longs_types.str.contains("close", regex=False).to_numpy()
        long_index_vals = longs.index.to_numpy()
        ax.scatter(
            long_index_vals[mask_entry],
            longs_price[mask_entry],
            c="b",
            marker=".",
            zorder=3.0,
        )
        ax.scatter(
            long_index_vals[mask_close],
            longs_price[mask_close],
            c="r",
            marker=".",
            zorder=3.0,
        )

        lp = longs[["pprice", "psize"]]
        if fast and all(np.issubdtype(lp[col].dtype, np.number) for col in lp.columns):
            lp = lp.astype(float, copy=False)
        else:
            lp = lp.apply(pd.to_numeric, errors="coerce")
        lp = lp.groupby(level=0).last()
        pprices_long = lp.reindex(hlcc.index).ffill()
        pct_change = pprices_long["pprice"].pct_change().fillna(0.0)
        pprices_long.loc[pct_change != 0.0, "pprice"] = np.nan
        pprices_filtered = pprices_long.loc[pprices_long["psize"] != 0.0, "pprice"]
        if not pprices_filtered.empty:
            ax.scatter(
                pprices_filtered.index.to_numpy(),
                pprices_filtered.to_numpy(),
                c="b",
                marker="|",
                zorder=2.8,
            )
        legend.extend(
            [
                "entries_long",
                "closes_long",
                "pprices_long",
            ]
        )
    if len(shorts) > 0:
        shorts_types = shorts["type"].astype(str)
        shorts_price_series = shorts["price"]
        if fast and np.issubdtype(shorts_price_series.dtype, np.number):
            shorts_price = shorts_price_series.to_numpy(copy=False)
        else:
            shorts_price = pd.to_numeric(shorts_price_series, errors="coerce").to_numpy()
        mask_entry = shorts_types.str.contains("entry", regex=False).to_numpy()
        mask_close = shorts_types.str.contains("close", regex=False).to_numpy()
        short_index_vals = shorts.index.to_numpy()
        ax.scatter(
            short_index_vals[mask_entry],
            shorts_price[mask_entry],
            c="m",
            marker="x",
            zorder=3.0,
        )
        ax.scatter(
            short_index_vals[mask_close],
            shorts_price[mask_close],
            c="c",
            marker="x",
            zorder=3.0,
        )

        sp = shorts[["pprice", "psize"]]
        if fast and all(np.issubdtype(sp[col].dtype, np.number) for col in sp.columns):
            sp = sp.astype(float, copy=False)
        else:
            sp = sp.apply(pd.to_numeric, errors="coerce")
        sp = sp.groupby(level=0).last()
        pprices_short = sp.reindex(hlcc.index).ffill()
        pct_change = pprices_short["pprice"].pct_change().fillna(0.0)
        pprices_short.loc[pct_change != 0.0, "pprice"] = np.nan
        pprices_filtered = pprices_short.loc[pprices_short["psize"] != 0.0, "pprice"]
        if not pprices_filtered.empty:
            ax.scatter(
                pprices_filtered.index.to_numpy(),
                pprices_filtered.to_numpy(),
                c="r",
                marker="|",
                zorder=2.8,
            )
        legend.extend(
            [
                "entries_short",
                "closes_short",
                "pprices_short",
            ]
        )
    ax.legend(legend)
    return plt


def create_forager_balance_figures(
    bal_eq: pd.DataFrame,
    figsize=(21, 13),
    *,
    logy: bool = False,
    include_logy: bool = False,
    autoplot: bool | None = None,
    return_figures: bool | None = None,
    stride: int = 1,
    fast: bool = False,
) -> dict:
    stride = max(1, int(stride)) if stride else 1
    df = bal_eq.iloc[::stride]

    def _extract_columns(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
        n_rows = len(df)
        if n_rows == 0:
            return np.empty((0, len(keys)))
        columns: list[np.ndarray] = []
        for key in keys:
            if key in df.columns:
                series = df[key]
                if fast and np.issubdtype(series.dtype, np.number):
                    values = series.to_numpy(copy=False)
                else:
                    values = pd.to_numeric(series, errors="coerce").to_numpy()
            else:
                values = np.full(n_rows, np.nan)
            columns.append(np.asarray(values, dtype=float))
        if not columns:
            return np.empty((n_rows, 0))
        return np.column_stack(columns)

    figures = {}
    panel_configs = [
        (
            "USD Cash / Balance / Equity",
            [
                ("USD Cash Wallet", "usd_cash_wallet"),
                ("USD Total Balance", "usd_total_balance"),
                ("USD Total Equity", "usd_total_equity"),
            ],
        ),
        (
            "BTC Cash / Balance / Equity",
            [
                ("BTC Cash Wallet", "btc_cash_wallet"),
                ("BTC Total Balance", "btc_total_balance"),
                ("BTC Total Equity", "btc_total_equity"),
            ],
        ),
    ]
    panel_data = [_extract_columns(df, [key for _, key in specs]) for _, specs in panel_configs]
    x = df.index.to_numpy()

    autoplot = (_ipy_display is not None) if autoplot is None else autoplot
    if return_figures is None:
        return_figures = not autoplot

    modes = [logy]
    if include_logy:
        modes = [False, True]

    for mode in modes:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
        y_transform = (lambda arr: np.where(arr > 0.0, arr, np.nan)) if mode else (lambda arr: arr)

        for ax, (title, series_specs), data in zip(axes, panel_configs, panel_data):
            ax.set_yscale("log" if mode else "linear")
            y_values = y_transform(data)
            for col_idx, (label, _) in enumerate(series_specs):
                ax.plot(
                    x,
                    y_values[:, col_idx],
                    label=label,
                    linewidth=1.0,
                )
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("Time")
        fig.tight_layout()

        key = "balance_and_equity_logy" if mode else "balance_and_equity"
        if return_figures:
            figures[key] = fig
        if autoplot:
            if _ipy_display is not None:
                _ipy_display(fig)
            else:  # pragma: no cover
                try:
                    fig.show()
                except Exception:
                    pass
        if not return_figures:
            plt.close(fig)

    return figures if return_figures else {}


def create_forager_coin_figures(
    coins: list,
    fdf: pd.DataFrame,
    hlcvs: np.ndarray,
    figsize=(21, 13),
    start_pct=0.0,
    end_pct=1.0,
    coin=None,
    on_figure: Optional[Callable[[str, Figure], None]] = None,
    close_after_callback: bool = True,
) -> dict:
    figures: Dict[str, Figure] = {}
    if hlcvs is None:
        return figures
    for idx, coin_ in enumerate(coins):
        if coin is not None and coin_ != coin:
            continue
        fdfc = fdf[fdf.coin == coin_]
        if fdfc.empty:
            continue
        hlcvs_df = pd.DataFrame(hlcvs[:, idx, :3], columns=["high", "low", "close"])
        plt.figure(figsize=figsize)
        plot_fills_forager(fdfc, hlcvs_df, clear=False, start_pct=start_pct, end_pct=end_pct)
        fig = plt.gcf()
        ax = fig.axes[0] if fig.axes else fig.add_subplot(111)
        ax.set_title(f"Fills {coin_}")
        ax.set_xlabel("Minute")
        ax.set_ylabel("Price")
        if on_figure is not None:
            on_figure(coin_, fig)
            if close_after_callback:
                plt.close(fig)
        else:
            figures[coin_] = fig
    return figures


def save_figures(figures: dict, output_dir: str, suffix: str = ".png", close: bool = True) -> dict:
    if not figures:
        return {}
    output_dir = make_get_filepath(output_dir if output_dir.endswith("/") else f"{output_dir}/")
    saved_paths = {}
    for name, fig in figures.items():
        filepath = os.path.join(output_dir, f"{name}{suffix}")
        fig.savefig(filepath)
        if close:
            plt.close(fig)
        saved_paths[name] = filepath
    return saved_paths


def plot_pareto_front(df, metrics, minimize=(True, True)):
    """
    Plot optimization results with Pareto front highlighted.

    Parameters:
    df (pandas.DataFrame): DataFrame containing optimization results
    metrics (tuple): Tuple of two column names to plot (metric1, metric2)
    minimize (tuple): Tuple of booleans indicating whether each metric should be minimized (default: (True, True))

    Returns:
    matplotlib.figure.Figure: The generated plot
    """
    if len(metrics) != 2:
        raise ValueError("Exactly two metrics must be provided")

    metric1, metric2 = metrics

    # Extract the metrics data
    x = df[metric1].values
    y = df[metric2].values

    # Function to identify Pareto optimal points
    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep any point with at least one better coordinate than this one
                if minimize[0] and minimize[1]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                elif not minimize[0] and minimize[1]:
                    costs_comp = costs.copy()
                    costs_comp[:, 0] = -costs_comp[:, 0]
                    is_efficient[is_efficient] = np.any(
                        costs_comp[is_efficient] < costs_comp[i], axis=1
                    )
                elif minimize[0] and not minimize[1]:
                    costs_comp = costs.copy()
                    costs_comp[:, 1] = -costs_comp[:, 1]
                    is_efficient[is_efficient] = np.any(
                        costs_comp[is_efficient] < costs_comp[i], axis=1
                    )
                else:  # not minimize[0] and not minimize[1]
                    is_efficient[is_efficient] = np.any(-costs[is_efficient] < -c, axis=1)
                is_efficient[i] = True
        return is_efficient

    # Find Pareto optimal points
    costs = np.column_stack((x, y))
    pareto_mask = is_pareto_efficient(costs)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all points
    ax.scatter(x[~pareto_mask], y[~pareto_mask], c="gray", alpha=0.5, label="Non-Pareto optimal")

    # Plot Pareto optimal points
    ax.scatter(x[pareto_mask], y[pareto_mask], c="red", label="Pareto optimal")

    # Connect Pareto points with a line
    pareto_points = costs[pareto_mask]
    # Sort points for proper line connection
    if minimize[0]:
        pareto_points = pareto_points[pareto_points[:, 0].argsort()]
    else:
        pareto_points = pareto_points[(-pareto_points[:, 0]).argsort()]
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], "r--", alpha=0.5)

    # Labels and title
    ax.set_xlabel(metric1)
    ax.set_ylabel(metric2)
    ax.set_title("Optimization Results with Pareto Front")
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig


def add_metrics_to_fdf(fdf):
    fdf.loc[:, "wallet_exposure"] = fdf.psize.abs() * fdf.pprice / fdf.balance
    fdf.loc[:, "pprice_dist"] = fdf.apply(
        lambda x: pbr.calc_pprice_diff_int(0 if "long" in x.type else 1, x.pprice, x.price), axis=1
    )
    return fdf
