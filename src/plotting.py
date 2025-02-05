import json
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from colorama import init, Fore
from prettytable import PrettyTable

from procedures import dump_live_config, make_get_filepath
from pure_funcs import denumpyize, ts_to_date
import passivbot_rust as pbr


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
    plt.rcParams["figure.figsize"] = [29, 18]
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


def scale_array(xs, bottom, top):
    # Calculate the midpoint
    midpoint = (bottom + top) / 2

    # Scale the array
    scaled_xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))  # Scale between 0 and 1
    scaled_xs = (
        scaled_xs * (top - bottom) + midpoint - (top - bottom) / 2
    )  # Scale to the desired range and shift to the midpoint

    return scaled_xs


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


def plot_fills_forager(fdf: pd.DataFrame, hlcvs_df: pd.DataFrame, start_pct=0.0, end_pct=1.0):
    plt.clf()
    if len(fdf) == 0:
        return
    hlcc = hlcvs_df[["high", "low", "close"]].loc[fdf.iloc[0].minute : fdf.iloc[-1].minute]
    fdfc = fdf.set_index(fdf.minute.astype(int))

    start_minute = int(hlcc.index[0] + hlcc.index[-1] * start_pct)
    end_minute = int(hlcc.index[0] + hlcc.index[-1] * end_pct)
    hlcc = hlcc.loc[start_minute:end_minute]
    fdfc = fdfc.loc[start_minute:end_minute]
    ax = hlcc.close.plot(style="y--")
    hlcc.low.plot(style="g--")
    hlcc.high.plot(style="g--")
    longs = fdfc[fdfc.type.str.contains("long")]
    shorts = fdfc[fdfc.type.str.contains("short")]
    if len(longs) == 0 and len(shorts) == 0:
        return plt
    legend = ["close", "high", "low"]
    if len(longs) > 0:
        pprices_long = hlcc.join(longs[["pprice", "psize"]]).astype(float).ffill()
        pprices_long.loc[pprices_long.pprice.pct_change() != 0.0, "pprice"] = np.nan
        pprices_long = pprices_long[pprices_long.psize != 0.0].pprice
        longs[longs.type.str.contains("entry")].price.plot(style="b.")
        longs[longs.type.str.contains("close")].price.plot(style="r.")
        pprices_long.plot(style="b|")
        legend.extend(
            [
                "entries_long",
                "closes_long",
                "pprices_long",
            ]
        )
    if len(shorts) > 0:
        pprices_short = hlcc.join(shorts[["pprice", "psize"]]).astype(float).ffill()
        pprices_short.loc[pprices_short.pprice.pct_change() != 0.0, "pprice"] = np.nan
        pprices_short = pprices_short[pprices_short.psize != 0.0].pprice
        shorts[shorts.type.str.contains("entry")].price.plot(style="mx")
        shorts[shorts.type.str.contains("close")].price.plot(style="cx")
        pprices_short.plot(style="r|")
        legend.extend(
            [
                "entries_short",
                "closes_short",
                "pprices_short",
            ]
        )
    ax.legend(legend)
    return plt


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
