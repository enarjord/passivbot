import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from colorama import init, Fore
from prettytable import PrettyTable

from njit_funcs import round_up
from procedures import dump_live_config
from pure_funcs import round_dynamic, denumpyize


def dump_plots_emas(
    table,
    result: dict,
    fdf: pd.DataFrame,
    sdf: pd.DataFrame,
    closes: pd.DataFrame,
    disable_plotting: bool = False,
):
    for key in result["result"]:
        try:
            table.add_row(
                [
                    " ".join([x.capitalize() for x in key.split("_")]),
                    round_dynamic(result["result"][key], 4),
                ]
            )
        except:
            pass
    print("writing backtest_result.txt...\n")
    with open(f"{result['plots_dirpath']}backtest_result.txt", "w") as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output))
    if disable_plotting:
        return
    print(f"\nplotting balance and equity...")
    plt.clf()
    sdf.balance.plot()
    sdf.equity.plot(title=f"Balance and equity", xlabel="Time", ylabel="Balance")
    plt.savefig(f"{result['plots_dirpath']}balance_and_equity_sampled.png")
    plt.clf()
    eqnorm = sdf.set_index("timestamp").equity
    eqnorm = (eqnorm - eqnorm.min()) / (eqnorm.max() - eqnorm.min())
    eqnorm = eqnorm * sdf.price.max() + sdf.price.min()
    eqnorm.plot()
    buys = fdf[fdf.qty > 0.0].set_index("timestamp")
    sells = fdf[fdf.qty < 0.0].set_index("timestamp")
    sdf.set_index("timestamp").price.plot(style="y-")
    buys.price.plot(style="bo")
    sells.price.plot(style="ro", title="Whole backtest with eq", xlabel="Time", ylabel="Fills")
    plt.savefig(f"{result['plots_dirpath']}whole_backtest.png")


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
    table = PrettyTable(["Metric", "Value"])
    table.align["Metric"] = "l"
    table.align["Value"] = "l"
    table.title = "Summary"

    table.add_row(["Exchange", result["exchange"] if "exchange" in result else "unknown"])
    table.add_row(["Market type", result["market_type"] if "market_type" in result else "unknown"])
    table.add_row(["Symbol", result["symbol"] if "symbol" in result else "unknown"])
    table.add_row(["No. days", round_dynamic(result["result"]["n_days"], 2)])
    table.add_row(["Starting balance", round_dynamic(result["result"]["starting_balance"], 6)])
    if result["passivbot_mode"] == "emas":
        return dump_plots_emas(table, result, longs, sdf, df, disable_plotting)
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
    n_parts = n_parts if n_parts is not None else max(3, int(round_up(result["n_days"] / 14, 1.0)))
    for side, fdf in [("long", longs), ("short", shorts)]:
        if result[side]["enabled"]:
            plt.clf()
            fig = plot_fills(df, fdf, plot_whole_df=True, title=f"Overview Fills {side.capitalize()}")
            if not fig:
                continue
            fig.savefig(f"{result['plots_dirpath']}whole_backtest_{side}.png")
            print(f"\nplotting balance and equity {side}...")
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

            print(f"plotting {side} initial entry band")
            if "timestamp" in df.columns:
                tick_interval = df.timestamp.iloc[1] - df.timestamp.iloc[0]
            else:
                tick_interval = df.index[1] - df.index[0]
            try:
                spans_multiplier = 60 / (tick_interval / 1000)
                spans = [
                    result[side]["ema_span_0"] * spans_multiplier,
                    ((result[side]["ema_span_0"] * result[side]["ema_span_1"]) ** 0.5)
                    * spans_multiplier,
                    result[side]["ema_span_1"] * spans_multiplier,
                ]
                emas = pd.DataFrame(
                    {
                        str(span): df.iloc[::100]
                        .price.ewm(span=max(1.0, span / 100), adjust=False)
                        .mean()
                        for span in spans
                    }
                )
                ema_band_lower = emas.min(axis=1)
                ema_band_upper = emas.max(axis=1)
                if side == "long":
                    ientry_band = ema_band_lower * (1 - result[side]["initial_eprice_ema_dist"])
                else:
                    ientry_band = ema_band_upper * (1 + result[side]["initial_eprice_ema_dist"])
                plt.clf()
                df.price.iloc[::100].plot(style="y-", title=f"{side.capitalize()} Initial Entry Band")
                ientry_band.plot(style=f"{('b' if side == 'long' else 'r')}-.")
                plt.savefig(f"{result['plots_dirpath']}initial_entry_band_{side}.png")
                if result[side]["auto_unstuck_wallet_exposure_threshold"] != 0.0:
                    print(f"plotting {side} unstucking bands...")
                    unstucking_band_lower = ema_band_lower * (
                        1 - result[side]["auto_unstuck_ema_dist"]
                    )
                    unstucking_band_upper = ema_band_upper * (
                        1 + result[side]["auto_unstuck_ema_dist"]
                    )
                    plt.clf()
                    df.price.iloc[::100].plot(
                        style="y-", title=f"{side.capitalize()} Auto Unstucking Bands"
                    )
                    unstucking_band_lower.plot(style="b-.")
                    unstucking_band_upper.plot(style="r-.")
                    plt.savefig(f"{result['plots_dirpath']}auto_unstuck_bands_{side}.png")
            except:
                print("skipping ema bands")
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
    fdf = fdf_.set_index("timestamp")
    dfc = df  # .iloc[::max(1, int(len(df) * 0.00001))]
    if dfc.index.name != "timestamp":
        dfc = dfc.set_index("timestamp")
    if not plot_whole_df:
        dfc = dfc[(dfc.index > fdf.index[0]) & (dfc.index < fdf.index[-1])]
        dfc = dfc.loc[fdf.index[0] : fdf.index[-1]]
    dfc.price.plot(style="y-", title=title, xlabel="Time", ylabel="Price + Fills")
    if side >= 0:
        longs = fdf[fdf.type.str.contains("long")]
        lentry = longs[
            longs.type.str.contains("rentry")
            | longs.type.str.contains("ientry")
            | (longs.type == "entry_long")
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
