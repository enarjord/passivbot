import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from colorama import init, Fore
from prettytable import PrettyTable

from njit_funcs import round_up
from procedures import dump_live_config
from pure_funcs import round_dynamic, denumpyize


def dump_plots(
    result: dict,
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    sdf: pd.DataFrame,
    df: pd.DataFrame,
    n_parts: int = None,
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
    table.add_row(["No. days", round_dynamic(result["result"]["n_days"], 6)])
    table.add_row(["Starting balance", round_dynamic(result["result"]["starting_balance"], 6)])

    for side in ["long", "short"]:
        if result[side]["enabled"]:
            table.add_row([" ", " "])
            table.add_row([side.capitalize(), result[side]["enabled"]])
            adg_per_exp = result["result"][f"adg_{side}"] / result[side]["wallet_exposure_limit"]
            table.add_row(["ADG per exposure", f"{round_dynamic(adg_per_exp * 100, 3)}%"])
            profit_color = (
                Fore.RED
                if result["result"][f"final_balance_{side}"] < result["result"]["starting_balance"]
                else Fore.RESET
            )
            table.add_row(
                [
                    "Final balance",
                    f"{profit_color}{round_dynamic(result['result'][f'final_balance_{side}'], 6)}{Fore.RESET}",
                ]
            )
            table.add_row(
                [
                    "Final equity",
                    f"{profit_color}{round_dynamic(result['result'][f'final_equity_{side}'], 6)}{Fore.RESET}",
                ]
            )
            table.add_row(
                [
                    "Net PNL + fees",
                    f"{profit_color}{round_dynamic(result['result'][f'net_pnl_plus_fees_{side}'], 6)}{Fore.RESET}",
                ]
            )
            table.add_row(
                [
                    "Total gain",
                    f"{profit_color}{round_dynamic(result['result'][f'gain_{side}'] * 100, 4)}%{Fore.RESET}",
                ]
            )
            table.add_row(
                [
                    "Average daily gain",
                    f"{profit_color}{round_dynamic((result['result'][f'adg_{side}']) * 100, 3)}%{Fore.RESET}",
                ]
            )
            gain_per_exp = result["result"][f"gain_{side}"] / result[side]["wallet_exposure_limit"]
            table.add_row(["Gain per exposure", f"{round_dynamic(gain_per_exp * 100, 3)}%"])
            table.add_row(
                [
                    "DG mean std ratio",
                    f"{round_dynamic(result['result'][f'adg_DGstd_ratio_{side}'], 4)}",
                ]
            )
            table.add_row(
                [
                    f"Price action distance mean",
                    round_dynamic(result["result"][f"pa_distance_mean_{side}"], 6),
                ]
            )
            table.add_row(
                [
                    f"Price action distance std",
                    round_dynamic(result["result"][f"pa_distance_std_{side}"], 6),
                ]
            )
            table.add_row(
                [
                    f"Price action distance max",
                    round_dynamic(result["result"][f"pa_distance_max_{side}"], 6),
                ]
            )
            table.add_row(
                [
                    "Closest bankruptcy",
                    f'{round_dynamic(result["result"][f"closest_bkr_{side}"] * 100, 4)}%',
                ]
            )
            table.add_row(
                [
                    "Lowest equity/balance ratio",
                    f'{round_dynamic(result["result"][f"eqbal_ratio_min_{side}"], 4)}',
                ]
            )
            table.add_row(["No. fills", result["result"][f"n_fills_{side}"]])
            table.add_row(["No. entries", result["result"][f"n_entries_{side}"]])
            table.add_row(["No. closes", result["result"][f"n_closes_{side}"]])
            table.add_row(["No. initial entries", result["result"][f"n_ientries_{side}"]])
            table.add_row(["No. reentries", result["result"][f"n_rentries_{side}"]])
            table.add_row(["No. unstuck entries", result["result"][f"n_unstuck_entries_{side}"]])
            table.add_row(["No. unstuck closes", result["result"][f"n_unstuck_closes_{side}"]])
            table.add_row(["No. normal closes", result["result"][f"n_normal_closes_{side}"]])
            table.add_row(
                [
                    "Average n fills per day",
                    round_dynamic(result["result"][f"avg_fills_per_day_{side}"], 3),
                ]
            )
            table.add_row(
                [
                    "Mean hours stuck",
                    round_dynamic(result["result"][f"hrs_stuck_avg_{side}"], 6),
                ]
            )
            table.add_row(
                [
                    "Max hours stuck",
                    round_dynamic(result["result"][f"hrs_stuck_max_{side}"], 6),
                ]
            )
            profit_color = Fore.RED if result["result"][f"pnl_sum_{side}"] < 0 else Fore.RESET
            table.add_row(
                [
                    "PNL sum",
                    f"{profit_color}{round_dynamic(result['result'][f'pnl_sum_{side}'], 4)}{Fore.RESET}",
                ]
            )
            table.add_row(["Profit sum", round_dynamic(result["result"][f"profit_sum_{side}"], 4)])
            table.add_row(["Loss sum", round_dynamic(result["result"][f"loss_sum_{side}"], 4)])
            table.add_row(["Fee sum", round_dynamic(result["result"][f"fee_sum_{side}"], 4)])
            table.add_row(["Biggest pos size", result["result"][f"biggest_psize_{side}"]])
            table.add_row(
                [
                    "Biggest pos cost",
                    round_dynamic(result["result"][f"biggest_psize_quote_{side}"], 4),
                ]
            )
            table.add_row(
                ["Volume quote", round_dynamic(result["result"][f"volume_quote_{side}"], 6)]
            )

    dump_live_config(result, result["plots_dirpath"] + "live_config.json")
    json.dump(denumpyize(result), open(result["plots_dirpath"] + "result.json", "w"), indent=4)

    print("writing backtest_result.txt...\n")
    with open(f"{result['plots_dirpath']}backtest_result.txt", "w") as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output))

    n_parts = n_parts if n_parts is not None else max(3, int(round_up(result["n_days"] / 14, 1.0)))
    for side, fdf in [("long", longs), ("short", shorts)]:
        if result[side]["enabled"]:
            plt.clf()
            fig = plot_fills(df, fdf, plot_whole_df=True, title=f"Overview Fills {side.capitalize()}")
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
            spans = [
                result[side]["ema_span_0"] * 60,
                ((result[side]["ema_span_0"] * result[side]["ema_span_1"]) ** 0.5) * 60,
                result[side]["ema_span_1"] * 60,
            ]
            emas = pd.DataFrame(
                {
                    str(span): df.iloc[::100].price.ewm(span=span / 100, adjust=False).mean()
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
                unstucking_band_lower = ema_band_lower * (1 - result[side]["auto_unstuck_ema_dist"])
                unstucking_band_upper = ema_band_upper * (1 + result[side]["auto_unstuck_ema_dist"])
                plt.clf()
                df.price.iloc[::100].plot(
                    style="y-", title=f"{side.capitalize()} Auto Unstucking Bands"
                )
                unstucking_band_lower.plot(style="b-.")
                unstucking_band_upper.plot(style="r-.")
                plt.savefig(f"{result['plots_dirpath']}auto_unstuck_bands_{side}.png")
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
        lientry = longs[longs.type.str.contains("ientry")]
        lrentry = longs[longs.type.str.contains("rentry")]
        lnclose = longs[longs.type.str.contains("nclose")]
        luentry = longs[longs.type.str.contains("unstuck_entry")]
        luclose = longs[longs.type.str.contains("unstuck_close")]
        ldca = longs[longs.type.str.contains("secondary")]
        lientry.price.plot(style="b.")
        lrentry.price.plot(style="b.")
        lnclose.price.plot(style="r.")
        ldca.price.plot(style="go")
        luentry.price.plot(style="bx")
        luclose.price.plot(style="rx")

        longs.where(longs.pprice != 0.0).pprice.fillna(method="ffill").plot(style="b--")
    if side <= 0:
        shorts = fdf[fdf.type.str.contains("short")]
        sientry = shorts[shorts.type.str.contains("ientry")]
        srentry = shorts[shorts.type.str.contains("rentry")]
        snclose = shorts[shorts.type.str.contains("nclose")]
        suentry = shorts[shorts.type.str.contains("unstuck_entry")]
        suclose = shorts[shorts.type.str.contains("unstuck_close")]
        sdca = shorts[shorts.type.str.contains("secondary")]
        sientry.price.plot(style="r.")
        srentry.price.plot(style="r.")
        snclose.price.plot(style="b.")
        sdca.price.plot(style="go")
        suentry.price.plot(style="rx")
        suclose.price.plot(style="bx")
        shorts.where(shorts.pprice != 0.0).pprice.fillna(method="ffill").plot(style="r--")
    return plt
