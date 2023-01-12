import json
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from colorama import init, Fore
from prettytable import PrettyTable

from njit_funcs import round_up
from procedures import dump_live_config, make_get_filepath
from pure_funcs import round_dynamic, denumpyize, ts_to_date


def dump_plots_emas(
    table,
    result: dict,
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    sdf: pd.DataFrame,
    df: pd.DataFrame,
    n_parts: int,
    disable_plotting: bool = False,
):
    # (text, mul, precision, suffix)
    formatting = {
        "adg_realized_per_exposure_long": ("ADG realized per exposure long", 100, 3, "%"),
        "adg_realized": ("ADG realized", 100, 3, "%"),
        "adg_realized_per_exposure": ("ADG realized per exposure", 100, 3, "%"),
        "adg_realized_per_exposure_short": ("ADG realized per exposure short", 100, 3, "%"),
        "eqbal_ratio_min": ("Equity to Balance Ratio min", 1, 4, ""),
    }
    exclude = {}
    for key in result["result"]:
        if key in exclude:
            continue
        try:
            if key in formatting:
                val = round_dynamic(result["result"][key] * formatting[key][1], formatting[key][2])
                table.add_row(
                    [
                        formatting[key][0],
                        f"{val}{formatting[key][3]}",
                    ]
                )
            else:
                table.add_row([key, round_dynamic(result["result"][key], 4)])
        except:
            pass
    print("writing backtest_result.txt...\n")
    with open(f"{result['plots_dirpath']}backtest_result.txt", "w") as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output))
    dump_live_config(result, result["plots_dirpath"] + "live_config.json")
    if disable_plotting:
        return
    print(f"\nplotting balance and equity...")
    sdf = sdf.set_index("timestamp")
    longs = longs.set_index("timestamp")
    shorts = shorts.set_index("timestamp")
    plt.clf()
    sdf.balance.plot()
    sdf.equity.plot(title=f"Balance and equity", xlabel="Time", ylabel="Balance")
    plt.savefig(f"{result['plots_dirpath']}balance_and_equity_sampled.png")
    print("plotting whole backtest")
    plt.clf()
    eqnorm = sdf.equity
    eqnorm = (eqnorm - eqnorm.min()) / (eqnorm.max() - eqnorm.min())
    eqnorm = eqnorm * sdf.price.max() + sdf.price.min()
    eqnorm.plot()
    buys = fdf[fdf.qty > 0.0]
    sells = fdf[fdf.qty < 0.0]
    sdf.price.plot(style="y-")
    buys.price.plot(style="bo")
    sells.price.plot(style="ro", title="Whole backtest with eq", xlabel="Time", ylabel="Fills")
    plt.savefig(f"{result['plots_dirpath']}whole_backtest.png")
    spans = sorted(
        [
            result["ema_span_0"],
            (result["ema_span_0"] * result["ema_span_1"]) ** 0.5,
            result["ema_span_1"],
        ]
    )
    price_1m = df.set_index("timestamp").close
    emas = pd.DataFrame(
        {f"ema_{span}": price_1m.ewm(span=span, adjust=False).mean() for span in spans},
        index=price_1m.index,
    )
    eb_lower = emas.min(axis=1) * (1 - result["ema_dist_lower"])
    eb_upper = emas.max(axis=1) * (1 + result["ema_dist_upper"])
    print("plotting backtest slices")
    if n_parts is None:
        n_parts = 10
    for i in range(n_parts):
        start_idx = int(sdf.index[0] + (sdf.index[-1] - sdf.index[0]) * (i / n_parts))
        end_idx = int(sdf.index[0] + (sdf.index[-1] - sdf.index[0]) * ((i + 1) / n_parts))
        plt.clf()
        fdf_slice = fdf[(fdf.index >= start_idx) & (fdf.index <= end_idx)]
        sdf_slice = sdf[(sdf.index >= start_idx) & (sdf.index <= end_idx)]
        buys = fdf_slice[fdf_slice.qty > 0.0]
        sells = fdf_slice[fdf_slice.qty < 0.0]
        """
        eqnorm = sdf_slice.equity
        eqnorm = (eqnorm - eqnorm.min()) / (eqnorm.max() - eqnorm.min())
        eqnorm = eqnorm * sdf_slice.price.max() + sdf_slice.price.min()
        eqnorm.plot(style="k-")
        """
        sdf_slice.price.plot(style="y-")
        eb_lower[(eb_lower.index >= start_idx) & (eb_lower.index <= end_idx)].plot(style="b--")
        eb_upper[(eb_upper.index >= start_idx) & (eb_upper.index <= end_idx)].plot(style="r--")
        buys.price.plot(style="bo")
        title = f"backtest {i + 1}/{n_parts}"
        sells.price.plot(style="ro", title=title, xlabel="Time", ylabel="Fills")
        plt.savefig(f"{result['plots_dirpath']}backtest_{i + 1}_of_{n_parts}.png")


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
    plt.rcParams["figure.figsize"] = [29, 18]
    try:
        pd.set_option("display.precision", 10)
    except Exception as e:
        print("error setting pandas precision", e)

    result["plots_dirpath"] = make_get_filepath(
        os.path.join(result["plots_dirpath"], f"{ts_to_date(time.time())[:19].replace(':', '')}", "")
    )
    longs.to_csv(result["plots_dirpath"] + "fills_long.csv")
    shorts.to_csv(result["plots_dirpath"] + "fills_short.csv")
    sdf.to_csv(result["plots_dirpath"] + "stats.csv")

    table = PrettyTable(["Metric", "Value"])
    table.align["Metric"] = "l"
    table.align["Value"] = "l"
    table.title = "Summary"

    table.add_row(["Exchange", result["exchange"] if "exchange" in result else "unknown"])
    table.add_row(["Market type", result["market_type"] if "market_type" in result else "unknown"])
    table.add_row(["Symbol", result["symbol"] if "symbol" in result else "unknown"])
    table.add_row(["No. days", round_dynamic(result["result"]["n_days"], 2)])
    table.add_row(["Starting balance", round_dynamic(result["result"]["starting_balance"], 6)])
    for side in ["long", "short"]:
        if side not in result:
            result[side] = {"enabled": result[f"do_{side}"]}
        if result[side]["enabled"]:
            table.add_row([" ", " "])
            table.add_row([side.capitalize(), True])
            adg_realized_per_exp = result["result"][f"adg_realized_per_exposure_{side}"]
            table.add_row(
                ["ADG realized per exposure", f"{round_dynamic(adg_realized_per_exp * 100, 3)}%"]
            )
            profit_color = (
                Fore.RED
                if result["result"][f"final_balance_{side}"] < result["result"]["starting_balance"]
                else Fore.RESET
            )
            for title, key, precision, mul in [
                ("Final balance", f"final_balance_{side}", 6, 1),
                ("Final equity", f"final_equity_{side}", 6, 1),
                ("Net PNL + fees", f"net_pnl_plus_fees_{side}", 6, 1),
                ("Total gain", f"gain_{side}", 4, 100),
                ("Average daily gain", f"adg_{side}", 3, 100),
                ("Net PNL + fees", f"net_pnl_plus_fees_{side}", 6, 1),
                ("Loss to profit ratio", f"loss_profit_ratio_{side}", 4, 1),
                (f"Price action distance mean", f"pa_distance_mean_{side}", 6, 1),
                (f"Price action distance std", f"pa_distance_std_{side}", 6, 1),
                (f"Price action distance max", f"pa_distance_max_{side}", 6, 1),
                ("Closest bankruptcy", f"closest_bkr_{side}", 4, 100),
                ("Lowest equity/balance ratio", f"eqbal_ratio_min_{side}", 4, 1),
                ("Equity/balance ratio std", f"equity_balance_ratio_std_{side}", 4, 1),
            ]:
                if key in result["result"]:
                    table.add_row(
                        [
                            title,
                            f"{profit_color}{round_dynamic(result['result'][key] * mul, precision)}{Fore.RESET}",
                        ]
                    )
            for title, key in [
                ("No. fills", f"n_fills_{side}"),
                ("No. entries", f"n_entries_{side}"),
                ("No. closes", f"n_closes_{side}"),
                ("No. initial entries", f"n_ientries_{side}"),
                ("No. reentries", f"n_rentries_{side}"),
                ("No. unstuck entries", f"n_unstuck_entries_{side}"),
                ("No. unstuck closes", f"n_unstuck_closes_{side}"),
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
                    table.add_row([title, round_dynamic(result["result"][key], precision)])

            if f"pnl_sum_{side}" in result["result"]:
                profit_color = Fore.RED if result["result"][f"pnl_sum_{side}"] < 0 else Fore.RESET

                table.add_row(
                    [
                        "PNL sum",
                        f"{profit_color}{round_dynamic(result['result'][f'pnl_sum_{side}'], 4)}{Fore.RESET}",
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
                    table.add_row([title, round_dynamic(result["result"][key], precision)])

    dump_live_config(result, result["plots_dirpath"] + "live_config.json")
    json.dump(denumpyize(result), open(result["plots_dirpath"] + "result.json", "w"), indent=4)

    print("writing backtest_result.txt...\n")
    with open(f"{result['plots_dirpath']}backtest_result.txt", "w") as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output))

    if disable_plotting:
        return
    n_parts = (
        n_parts if n_parts is not None else min(12, max(3, int(round_up(result["n_days"] / 14, 1.0))))
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

    print("plotting pos sizes...")
    plt.clf()

    sdf[["psize_long", "psize_short"]].plot(
        title="Position size in terms of contracts",
        xlabel="Time",
        ylabel="Position size",
    )
    plt.savefig(f"{result['plots_dirpath']}psizes_plot.png")


def plot_fills(df, fdf_, side: int = 0, plot_whole_df: bool = False, title=""):
    if fdf_.empty:
        return
    plt.clf()
    fdf = fdf_.set_index("timestamp") if fdf_.index.name != "timestamp" else fdf_
    dfc = df  # .iloc[::max(1, int(len(df) * 0.00001))]
    if dfc.index.name != "timestamp":
        dfc = dfc.set_index("timestamp")
    if not plot_whole_df:
        dfc = dfc[(dfc.index > fdf.index[0]) & (dfc.index < fdf.index[-1])]
        dfc = dfc.loc[fdf.index[0] : fdf.index[-1]]
    dfc.price.plot(style="y-", title=title, xlabel="Time", ylabel="Price + Fills")
    if side >= 0:
        longs = fdf[fdf.type.str.contains("long")]
        types = longs.type.unique()
        if any(x in types for x in ["entry_ema_long", "close_ema_long", "close_markup_long"]):
            # emas mode
            longs[longs.type == "entry_ema_long"].price.plot(style="bo")
            longs[longs.type == "close_ema_long"].price.plot(style="ro")
            longs[longs.type == "close_markup_long"].price.plot(style="rx")
        else:
            lentry = longs[
                longs.type.str.contains("rentry")
                | longs.type.str.contains("ientry")
                | (longs.type == "entry_long")
                | (longs.type == "entry_ema_long")
            ]
            lnclose = longs[longs.type.str.contains("nclose") | (longs.type == "close_long")]
            luentry = longs[longs.type.str.contains("unstuck_entry")]
            luclose = longs[longs.type.str.contains("unstuck_close")]
            ldca = longs[longs.type.str.contains("secondary")]
            lentry.price.plot(style="b.")
            lnclose.price.plot(style="r.")
            ldca.price.plot(style="go")
            luentry.price.plot(style="bx")
            luclose.price.plot(style="rx")

        # longs.where(longs.pprice != 0.0).pprice.fillna(method="ffill").plot(style="b--")
        lppu = longs[(longs.pprice != longs.pprice.shift(1)) & (longs.pprice != 0.0)]
        for i in range(len(lppu) - 1):
            plt.plot(
                [lppu.index[i], lppu.index[i + 1]], [lppu.pprice.iloc[i], lppu.pprice.iloc[i]], "b--"
            )
    if side <= 0:
        shorts = fdf[fdf.type.str.contains("short")]
        types = shorts.type.unique()
        if any(x in types for x in ["entry_ema_short", "close_ema_short", "close_markup_short"]):
            # emas mode
            shorts[shorts.type == "entry_ema_short"].price.plot(style="ro")
            shorts[shorts.type == "close_ema_short"].price.plot(style="bo")
            shorts[shorts.type == "close_markup_short"].price.plot(style="bx")
        else:
            sentry = shorts[
                shorts.type.str.contains("rentry")
                | shorts.type.str.contains("ientry")
                | (shorts.type == "entry_short")
            ]
            snclose = shorts[shorts.type.str.contains("nclose") | (shorts.type == "close_short")]
            suentry = shorts[shorts.type.str.contains("unstuck_entry")]
            suclose = shorts[shorts.type.str.contains("unstuck_close")]
            sdca = shorts[shorts.type.str.contains("secondary")]
            sentry.price.plot(style="r.")
            snclose.price.plot(style="b.")
            sdca.price.plot(style="go")
            suentry.price.plot(style="rx")
            suclose.price.plot(style="bx")
        # shorts.where(shorts.pprice != 0.0).pprice.fillna(method="ffill").plot(style="r--")
        sppu = shorts[(shorts.pprice != shorts.pprice.shift(1)) & (shorts.pprice != 0.0)]
        for i in range(len(sppu) - 1):
            plt.plot(
                [sppu.index[i], sppu.index[i + 1]], [sppu.pprice.iloc[i], sppu.pprice.iloc[i]], "r--"
            )

    return plt
