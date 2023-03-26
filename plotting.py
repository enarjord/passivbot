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
    table.add_row(["No. days", round_dynamic(result["result"]["n_days"], 2)])
    table.add_row(["Starting balance", round_dynamic(result["result"]["starting_balance"], 6)])
    for side in ["long", "short"]:
        if side not in result:
            result[side] = {"enabled": result[f"do_{side}"]}
        if result[side]["enabled"]:
            table.add_row([" ", " "])
            table.add_row([side.capitalize(), True])
            profit_color = (
                Fore.RED
                if result["result"][f"final_balance_{side}"] < result["result"]["starting_balance"]
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
                ("Final balance", f"final_balance_{side}", 6, 1, ""),
                ("Final equity", f"final_equity_{side}", 6, 1, ""),
                ("Net PNL + fees", f"net_pnl_plus_fees_{side}", 6, 1, ""),
                ("Net Total gain", f"gain_{side}", 4, 100, "%"),
                ("Average daily gain", f"adg_{side}", 3, 100, "%"),
                ("Average daily gain weighted", f"adg_weighted_{side}", 3, 100, "%"),
                ("Loss to profit ratio", f"loss_profit_ratio_{side}", 4, 1, ""),
                (f"Price action distance mean", f"pa_distance_mean_{side}", 6, 1, ""),
                (f"Price action distance std", f"pa_distance_std_{side}", 6, 1, ""),
                (f"Price action distance max", f"pa_distance_max_{side}", 6, 1, ""),
                ("Closest bankruptcy", f"closest_bkr_{side}", 4, 100, "%"),
                ("Lowest equity/balance ratio", f"eqbal_ratio_min_{side}", 4, 1, ""),
                ("Mean of 10 worst eq/bal ratios", f"eqbal_ratio_mean_of_10_worst_{side}", 4, 1, ""),
                ("Equity/balance ratio std", f"equity_balance_ratio_std_{side}", 4, 1, ""),
            ]:
                if key in result["result"]:
                    val = round_dynamic(result["result"][key] * mul, precision)
                    table.add_row(
                        [
                            title,
                            f"{profit_color}{val}{suffix}{Fore.RESET}",
                        ]
                    )
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
    table = make_table(result)

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
                ema_bands = pd.DataFrame(
                    {
                        "ema_band_lower": emas.min(axis=1) * (1 - ema_dist_lower),
                        "ema_band_upper": emas.max(axis=1) * (1 + ema_dist_upper),
                    },
                    index=df.index,
                )
                df = df.join(ema_bands)
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
                df = df.drop(["ema_band_lower", "ema_band_upper"], axis=1)

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
    if "ema_band_lower" in dfc.columns and "ema_band_upper" in dfc.columns:
        dfc.ema_band_lower.plot(style="b--")
        dfc.ema_band_upper.plot(style="r--")
    if side >= 0:
        longs = fdf[fdf.type.str.contains("long")]

        longs[longs.type.str.contains("rentry") | longs.type.str.contains("ientry")].price.plot(
            style="bo"
        )
        longs[longs.type.str.contains("secondary")].price.plot(style="go")
        longs[longs.type == "long_nclose"].price.plot(style="ro")
        longs[
            (longs.type.str.contains("unstuck_entry")) | (longs.type == "clock_entry_long")
        ].price.plot(style="bx")
        longs[
            (longs.type.str.contains("unstuck_close")) | (longs.type == "clock_close_long")
        ].price.plot(style="rx")

        lppu = longs[(longs.pprice != longs.pprice.shift(1)) & (longs.pprice != 0.0)]
        for i in range(len(lppu) - 1):
            plt.plot(
                [lppu.index[i], lppu.index[i + 1]], [lppu.pprice.iloc[i], lppu.pprice.iloc[i]], "b--"
            )
    if side <= 0:
        shorts = fdf[fdf.type.str.contains("short")]

        shorts[shorts.type.str.contains("rentry") | shorts.type.str.contains("ientry")].price.plot(
            style="ro"
        )
        shorts[shorts.type.str.contains("secondary")].price.plot(style="go")
        shorts[shorts.type == "short_nclose"].price.plot(style="bo")
        shorts[
            (shorts.type.str.contains("unstuck_entry")) | (shorts.type == "clock_entry_short")
        ].price.plot(style="rx")
        shorts[
            (shorts.type.str.contains("unstuck_close")) | (shorts.type == "clock_close_short")
        ].price.plot(style="bx")

        sppu = shorts[(shorts.pprice != shorts.pprice.shift(1)) & (shorts.pprice != 0.0)]
        for i in range(len(sppu) - 1):
            plt.plot(
                [sppu.index[i], sppu.index[i + 1]], [sppu.pprice.iloc[i], sppu.pprice.iloc[i]], "r--"
            )

    return plt
