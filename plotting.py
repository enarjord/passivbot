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
    result: dict, longs: pd.DataFrame, shorts: pd.DataFrame, sdf: pd.DataFrame, df: pd.DataFrame
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
    table.add_row(
        [
            "Price action distance mean long",
            round_dynamic(result["result"]["pa_distance_mean_long"], 6),
        ]
    )
    table.add_row(
        [
            "Price action distance max long",
            round_dynamic(result["result"]["pa_distance_max_long"], 6),
        ]
    )
    table.add_row(
        [
            "Price action distance mean short",
            round_dynamic(result["result"]["pa_distance_mean_short"], 6),
        ]
    )
    table.add_row(
        [
            "Price action distance max short",
            round_dynamic(result["result"]["pa_distance_max_short"], 6),
        ]
    )
    table.add_row([" ", " "])

    if result["long"]["enabled"]:
        table.add_row([" ", " "])
        table.add_row(["Long", result["long"]["enabled"]])
        adg_per_exp = result["result"]["adg_long"] / result["long"]["wallet_exposure_limit"]
        table.add_row(["ADG per exposure", f"{round_dynamic(adg_per_exp * 100, 3)}%"])
        table.add_row(
            ["DG mean std ratio", f"{round_dynamic(result['result']['adg_DGstd_ratio_long'], 4)}"]
        )
        table.add_row(["No. inital entries", result["result"]["n_ientries_long"]])
        table.add_row(["No. reentries", result["result"]["n_rentries_long"]])
        table.add_row(["No. normal closes", result["result"]["n_normal_closes_long"]])
        table.add_row(
            [
                "Mean hours stuck",
                round_dynamic(result["result"]["hrs_stuck_avg_long"], 6),
            ]
        )
        table.add_row(
            [
                "Max hours stuck",
                round_dynamic(result["result"]["hrs_stuck_max_long"], 6),
            ]
        )
        profit_color = Fore.RED if result["result"]["pnl_sum_long"] < 0 else Fore.RESET
        table.add_row(
            [
                "PNL sum",
                f"{profit_color}{round_dynamic(result['result']['pnl_sum_long'], 4)}{Fore.RESET}",
            ]
        )
        table.add_row(["Loss sum", round_dynamic(result["result"]["loss_sum_long"], 4)])
        table.add_row(["Fee sum", round_dynamic(result["result"]["fee_sum_long"], 4)])
        table.add_row(["Biggest pos size", result["result"]["biggest_psize_long"]])
        table.add_row(
            [
                "Biggest pos cost",
                round_dynamic(result["result"]["biggest_psize_quote_long"], 4),
            ]
        )
        table.add_row(
            [
                "Average n fills per day",
                round_dynamic(result["result"]["avg_fills_per_day_long"], 3),
            ]
        )
        table.add_row(["Volume quote", round_dynamic(result["result"]["volume_quote_long"], 6)])

    if result["short"]["enabled"]:
        table.add_row([" ", " "])
        table.add_row(["Short", result["short"]["enabled"]])
        adg_per_exp = result["result"]["adg_short"] / result["short"]["wallet_exposure_limit"]
        table.add_row(["ADG per exposure", f"{round_dynamic(adg_per_exp * 100, 3)}%"])
        table.add_row(
            ["DG mean std ratio", f"{round_dynamic(result['result']['adg_DGstd_ratio_short'], 4)}"]
        )
        table.add_row(["No. inital entries", result["result"]["n_ientries_short"]])
        table.add_row(["No. inital entries", result["result"]["n_ientries_short"]])
        table.add_row(["No. reentries", result["result"]["n_rentries_short"]])
        table.add_row(["No. normal closes", result["result"]["n_normal_closes_short"]])
        table.add_row(
            [
                "Mean hours stuck",
                round_dynamic(result["result"]["hrs_stuck_avg_short"], 6),
            ]
        )
        table.add_row(
            [
                "Max hours stuck",
                round_dynamic(result["result"]["hrs_stuck_max_short"], 6),
            ]
        )
        profit_color = Fore.RED if result["result"]["pnl_sum_short"] < 0 else Fore.RESET
        table.add_row(
            [
                "PNL sum",
                f"{profit_color}{round_dynamic(result['result']['pnl_sum_short'], 4)}{Fore.RESET}",
            ]
        )
        table.add_row(["Loss sum", round_dynamic(result["result"]["loss_sum_short"], 4)])
        table.add_row(["Fee sum", round_dynamic(result["result"]["fee_sum_short"], 4)])
        table.add_row(["Biggest pos size", result["result"]["biggest_psize_short"]])
        table.add_row(
            [
                "Biggest pos cost",
                round_dynamic(result["result"]["biggest_psize_quote_short"], 4),
            ]
        )
        table.add_row(
            [
                "Average n fills per day",
                round_dynamic(result["result"]["avg_fills_per_day_short"], 3),
            ]
        )
        table.add_row(["Volume quote", round_dynamic(result["result"]["volume_quote_short"], 6)])

    dump_live_config(result, result["plots_dirpath"] + "live_config.json")
    json.dump(denumpyize(result), open(result["plots_dirpath"] + "result.json", "w"), indent=4)

    print("writing backtest_result.txt...\n")
    with open(f"{result['plots_dirpath']}backtest_result.txt", "w") as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub("\033\\[([0-9]+)(;[0-9]+)*m", "", output))

    print("\nplotting balance and equity...")
    plt.clf()
    sdf.balance_long.plot()
    sdf.equity_long.plot(title="Balance and equity Long", xlabel="Time", ylabel="Balance")
    plt.savefig(f"{result['plots_dirpath']}balance_and_equity_sampled_long.png")
    plt.clf()
    sdf.balance_short.plot()
    sdf.equity_short.plot(title="Balance and equity Long", xlabel="Time", ylabel="Balance")
    plt.savefig(f"{result['plots_dirpath']}balance_and_equity_sampled_short.png")

    # TODO... fix
    return
    print("plotting backtest in chunks...")
    n_parts = max(3, int(round_up(result["n_days"] / 14, 1.0)))
    for z in range(n_parts):
        start_ = z / n_parts
        end_ = (z + 1) / n_parts
        print(f"{z} of {n_parts} {start_ * 100:.2f}% to {end_ * 100:.2f}%")
        fig = plot_fills(
            df,
            fdf.iloc[int(len(fdf) * start_) : int(len(fdf) * end_)],
            title=f"Fills {z+1} of {n_parts}",
        )
        if fig is not None:
            fig.savefig(f"{result['plots_dirpath']}backtest_{z + 1}of{n_parts}.png")
        else:
            print("no fills...")
    print("plotting backtest whole...")
    fig = plot_fills(df, fdf, plot_whole_df=True, title="Overview Fills")
    fig.savefig(f"{result['plots_dirpath']}whole_backtest.png")
    if result["long"]["enabled"]:
        print("plotting long fills...")
        plt.clf()
        fig = plot_fills(df, fdf, side=1, plot_whole_df=True, title="Overview Long Fills")
        fig.savefig(f"{result['plots_dirpath']}whole_backtest_long.png")
        print("plotting long initial entry band")
        spans = [
            result["long"]["ema_span_0"] * 60,
            ((result["long"]["ema_span_0"] * result["long"]["ema_span_1"]) ** 0.5) * 60,
            result["long"]["ema_span_1"] * 60,
        ]
        emas = pd.DataFrame(
            {str(span): df.price.ewm(span=span, adjust=False).mean() for span in spans}
        )
        ema_band_lower = emas.min(axis=1)
        ema_band_upper = emas.max(axis=1)
        long_ientry_band = ema_band_lower * (1 - result["long"]["initial_eprice_ema_dist"])
        plt.clf()
        df.price.plot(style="y-", title="Long Initial Entry Band")
        long_ientry_band.plot(style="b-.")
        plt.savefig(f"{result['plots_dirpath']}initial_entry_band_long.png")
        if result["long"]["auto_unstuck_wallet_exposure_threshold"] != 0.0:
            print("plotting long unstucking bands...")
            unstucking_band_lower = ema_band_lower * (1 - result["long"]["auto_unstuck_ema_dist"])
            unstucking_band_upper = ema_band_lower * (1 + result["long"]["auto_unstuck_ema_dist"])
            plt.clf()
            df.price.plot(style="y-", title="Long Auto Unstucking Bands")
            unstucking_band_lower.plot(style="b-.")
            unstucking_band_upper.plot(style="r-.")
            plt.savefig(f"{result['plots_dirpath']}auto_unstuck_bands_long.png")
    if result["short"]["enabled"]:
        print("plotting short fills...")
        plt.clf()
        fig = plot_fills(df, fdf, side=-1, plot_whole_df=True, title="Overview Short Fills")
        fig.savefig(f"{result['plots_dirpath']}whole_backtest_short.png")
        print("plotting short initial entry band")
        spans = [
            result["short"]["ema_span_0"] * 60,
            ((result["short"]["ema_span_0"] * result["short"]["ema_span_1"]) ** 0.5) * 60,
            result["short"]["ema_span_1"] * 60,
        ]
        emas = pd.DataFrame(
            {str(span): df.price.ewm(span=span, adjust=False).mean() for span in spans}
        )
        ema_band_lower = emas.min(axis=1)
        ema_band_upper = emas.max(axis=1)
        short_ientry_band = ema_band_lower * (1 + result["short"]["initial_eprice_ema_dist"])
        plt.clf()
        df.price.plot(style="y-", title="Short Initial Entry Band")
        short_ientry_band.plot(style="r-.")
        plt.savefig(f"{result['plots_dirpath']}initial_entry_band_short.png")
        if result["short"]["auto_unstuck_wallet_exposure_threshold"] != 0.0:
            print("plotting short unstucking bands...")
            unstucking_band_lower = ema_band_lower * (1 - result["short"]["auto_unstuck_ema_dist"])
            unstucking_band_upper = ema_band_lower * (1 + result["short"]["auto_unstuck_ema_dist"])
            plt.clf()
            df.price.plot(style="y-", title="short Auto Unstucking Bands")
            unstucking_band_lower.plot(style="b-.")
            unstucking_band_upper.plot(style="r-.")
            plt.savefig(f"{result['plots_dirpath']}auto_unstuck_bands_short.png")

    print("plotting pos sizes...")
    plt.clf()
    longs.psize.plot()
    shorts.psize.plot(
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
