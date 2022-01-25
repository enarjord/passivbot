import datetime
import pprint
from collections import OrderedDict

import numpy as np
import pandas as pd
from dateutil import parser

from njit_funcs import round_dynamic, qty_to_cost


def format_float(num):
    return np.format_float_positional(num, trim="-")


def compress_float(n: float, d: int) -> str:
    if n / 10 ** d >= 1:
        n = round(n)
    else:
        n = round_dynamic(n, d)
    nstr = format_float(n)
    if nstr.startswith("0."):
        nstr = nstr[1:]
    elif nstr.startswith("-0."):
        nstr = "-" + nstr[2:]
    elif nstr.endswith(".0"):
        nstr = nstr[:-2]
    return nstr


def calc_spans(min_span: int, max_span: int, n_spans: int) -> np.ndarray:
    return np.array(
        [min_span * ((max_span / min_span) ** (1 / (n_spans - 1))) ** i for i in range(0, n_spans)]
    )
    return np.array([min_span, (min_span * max_span) ** 0.5, max_span])


def get_xk_keys(passivbot_mode="static_grid"):
    if passivbot_mode == "recursive_grid":
        return [
            "inverse",
            "do_long",
            "do_short",
            "qty_step",
            "price_step",
            "min_qty",
            "min_cost",
            "c_mult",
            "ema_span_0",
            "ema_span_1",
            "iqty_pct",
            "iprice_ema_dist",
            "wallet_exposure_limit",
            "ddown_factor",
            "rentry_pprice_dist",
            "rentry_pprice_dist_wallet_exposure_weighting",
            "min_markup",
            "markup_range",
            "n_close_orders",
            "auto_unstuck_wallet_exposure_threshold",
            "auto_unstuck_ema_dist",
        ]
    return [
        "spot",
        "hedge_mode",
        "inverse",
        "do_long",
        "do_short",
        "qty_step",
        "price_step",
        "min_qty",
        "min_cost",
        "c_mult",
        "grid_span",
        "wallet_exposure_limit",
        "max_n_entry_orders",
        "initial_qty_pct",
        "eprice_pprice_diff",
        "secondary_allocation",
        "secondary_pprice_diff",
        "eprice_exp_base",
        "min_markup",
        "markup_range",
        "n_close_orders",
        "ema_span_min",
        "ema_span_max",
        "initial_eprice_ema_dist",
        "auto_unstuck_wallet_exposure_threshold",
        "auto_unstuck_ema_dist",
    ]


def determine_passivbot_mode(config: dict) -> str:
    if all(k in config["long"] for k in get_template_live_config("recursive_grid")["long"]):
        return "recursive_grid"
    elif all(k in config["long"] for k in get_template_live_config("static_grid")["long"]):
        return "static_grid"
    else:
        raise Exception("unable to determine passivbot mode")


def create_xk(config: dict) -> dict:
    xk = {}
    config_ = config.copy()
    if "spot" in config_["market_type"]:
        config_ = spotify_config(config_)
    else:
        config_["spot"] = False
        config_["do_long"] = config["long"]["enabled"]
        config_["do_short"] = config["short"]["enabled"]
    config["passivbot_mode"] = determine_passivbot_mode(config)
    keys = get_xk_keys(config["passivbot_mode"])
    config_["long"]["n_close_orders"] = int(round(config_["long"]["n_close_orders"]))
    config_["short"]["n_close_orders"] = int(round(config_["short"]["n_close_orders"]))
    if config["passivbot_mode"] == "static_grid":
        config_["long"]["max_n_entry_orders"] = int(round(config_["long"]["max_n_entry_orders"]))
        config_["short"]["max_n_entry_orders"] = int(round(config_["short"]["max_n_entry_orders"]))
    for k in keys:
        if k in config_["long"]:
            xk[k] = (config_["long"][k], config_["short"][k])
        elif k in config_:
            xk[k] = config_[k]
        else:
            raise Exception("failed to create xk", k)
    return xk


def numpyize(x):
    if type(x) in [list, tuple]:
        return np.array([numpyize(e) for e in x])
    elif type(x) == dict:
        numpyd = {}
        for k, v in x.items():
            numpyd[k] = numpyize(v)
        return numpyd
    else:
        return x


def denumpyize(x):
    if type(x) in [np.float64, np.float32, np.float16]:
        return float(x)
    elif type(x) in [np.int64, np.int32, np.int16, np.int8]:
        return int(x)
    elif type(x) == np.ndarray:
        return [denumpyize(e) for e in x]
    elif type(x) == np.bool_:
        return bool(x)
    elif type(x) in [dict, OrderedDict]:
        denumpyd = {}
        for k, v in x.items():
            denumpyd[k] = denumpyize(v)
        return denumpyd
    elif type(x) == list:
        return [denumpyize(z) for z in x]
    elif type(x) == tuple:
        return tuple([denumpyize(z) for z in x])
    else:
        return x


def denanify(x, nan=0.0, posinf=0.0, neginf=0.0):
    try:
        assert type(x) != str
        _ = float(x)
        return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    except:
        if type(x) == list:
            return [denanify(e) for e in x]
        elif type(x) == tuple:
            return tuple(denanify(e) for e in x)
        elif type(x) == np.ndarray:
            return np.array([denanify(e) for e in x], dtype=x.dtype)
        elif type(x) == dict:
            denanified = {}
            for k, v in x.items():
                denanified[k] = denanify(v)
            return denanified
        else:
            return x


def ts_to_date(timestamp: float) -> str:
    if timestamp > 253402297199:
        return str(datetime.datetime.fromtimestamp(timestamp / 1000)).replace(" ", "T")
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(" ", "T")


def date_to_ts(d):
    return int(parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def get_utc_now_timestamp() -> int:
    """
    Creates a millisecond based timestamp of UTC now.
    :return: Millisecond based timestamp of UTC now.
    """
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)


def config_pretty_str(config: dict):
    pretty_str = pprint.pformat(config)
    for r in [("'", '"'), ("True", "true"), ("False", "false")]:
        pretty_str = pretty_str.replace(*r)
    return pretty_str


def candidate_to_live_config(candidate: dict, recursive_grid=False) -> dict:
    result_dict = candidate["result"] if "result" in candidate else candidate
    passivbot_mode = name = determine_passivbot_mode(candidate)
    if passivbot_mode == "recursive_grid":
        live_config = get_template_live_config("recursive_grid")
        for side in ["long", "short"]:
            for k in live_config[side]:
                live_config[side][k] = candidate[side][k]
            live_config[side]["n_close_orders"] = int(round(live_config[side]["n_close_orders"]))
    elif passivbot_mode == "static_grid":
        live_config = get_template_live_config("static_grid")
        sides = ["long", "short"]
        for side in sides:
            for k in live_config[side]:
                if k in candidate[side]:
                    live_config[side][k] = candidate[side][k]
        for k in live_config:
            if k not in sides and k in candidate:
                live_config[k] = candidate[k]
    else:
        raise Exception("unknown passivbot mode")
    if "symbols" in result_dict:
        if len(result_dict["symbols"]) > 1:
            name += f"_{len(result_dict['symbols'])}_symbols"
        else:
            name += f"_{result_dict['symbols'][0]}"
    elif "symbol" in result_dict:
        name += f"_{result_dict['symbol']}"
    if "n_days" in result_dict:
        n_days = result_dict["n_days"]
    elif "start_date" in result_dict:
        n_days = (date_to_ts(result_dict["end_date"]) - date_to_ts(result_dict["start_date"])) / (
            1000 * 60 * 60 * 24
        )
    else:
        n_days = 0
    name += f"_{n_days:.0f}days"
    if "average_daily_gain" in result_dict:
        name += f"_adg{(result_dict['average_daily_gain']) * 100:.3f}%"
    elif "daily_gain" in result_dict:
        name += f"_adg{(result_dict['daily_gain'] - 1) * 100:.3f}%"
    live_config["config_name"] = name
    return denumpyize(live_config)


def unpack_config(d):
    new = {}
    for k, v in flatten_dict(d, sep="£").items():
        try:
            assert type(v) != str
            for _ in v:
                break
            for i in range(len(v)):
                new[f"{k}${str(i).zfill(2)}"] = v[i]
        except:
            new[k] = v
    if new == d:
        return new
    return unpack_config(new)


def pack_config(d):
    result = {}
    while any("$" in k for k in d):
        new = {}
        for k, v in denumpyize(d).items():
            if "$" in k:
                ks = k.split("$")
                k0 = "$".join(ks[:-1])
                if k0 in new:
                    new[k0].append(v)
                else:
                    new[k0] = [v]
            else:
                new[k] = v
        d = new
    new = {}
    for k, v in d.items():
        if type(v) == list:
            new[k] = np.array(v)
        else:
            new[k] = v
    d = new

    new = {}
    for k, v in d.items():
        if "£" in k:
            k0, k1 = k.split("£")
            if k0 in new:
                new[k0][k1] = v
            else:
                new[k0] = {k1: v}
        else:
            new[k] = v
    return new


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if type(v) == dict:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def filter_orders(
    actual_orders: [dict],
    ideal_orders: [dict],
    keys: [str] = ("symbol", "side", "qty", "price"),
) -> ([dict], [dict]):
    # returns (orders_to_delete, orders_to_create)

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []
    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_orders_cropped = [{k: o[k] for k in keys} for o in ideal_orders]
    actual_orders_cropped = [{k: o[k] for k in keys} for o in actual_orders]
    for ioc, io in zip(ideal_orders_cropped, ideal_orders):
        matches = [(aoc, ao) for aoc, ao in zip(actual_orders_cropped, actual_orders) if aoc == ioc]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_orders_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(io)
    return actual_orders, orders_to_create


def get_dummy_settings(config: dict):
    dummy_settings = get_template_live_config()
    dummy_settings.update({k: 1.0 for k in get_xk_keys()})
    dummy_settings.update(
        {
            "user": config["user"],
            "exchange": config["exchange"],
            "symbol": config["symbol"],
            "config_name": "",
            "logging_level": 0,
        }
    )
    return {**config, **dummy_settings}


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


def get_template_live_config(passivbot_mode="static_grid"):
    if passivbot_mode == "recursive_grid":
        return sort_dict_keys(
            {
                "config_name": "recursive_grid_test",
                "logging_level": 0,
                "long": {
                    "enabled": True,
                    "ema_span_0": 1036.4758617491368,
                    "ema_span_1": 1125.5167077975314,
                    "iqty_pct": 0.01,
                    "iprice_ema_dist": -0.02,
                    "wallet_exposure_limit": 1.0,
                    "ddown_factor": 0.6,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.15,
                    "auto_unstuck_ema_dist": 0.02,
                },
                "short": {
                    "enabled": False,
                    "ema_span_0": 1036.4758617491368,
                    "ema_span_1": 1125.5167077975314,
                    "iqty_pct": 0.01,
                    "iprice_ema_dist": -0.02,
                    "wallet_exposure_limit": 1.0,
                    "ddown_factor": 0.6,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.15,
                    "auto_unstuck_ema_dist": 0.02,
                },
            }
        )
    return sort_dict_keys(
        {
            "config_name": "template",
            "logging_level": 0,
            "long": {
                "enabled": True,
                "ema_span_min": 1440,  # in minutes
                "ema_span_max": 4320,
                "grid_span": 0.16,
                "wallet_exposure_limit": 1.6,
                "max_n_entry_orders": 10,
                "initial_qty_pct": 0.01,
                "initial_eprice_ema_dist": -0.01,  # negative is closer; positive is further away
                "eprice_pprice_diff": 0.0025,
                "secondary_allocation": 0.5,
                "secondary_pprice_diff": 0.35,
                "eprice_exp_base": 1.618034,
                "min_markup": 0.0045,
                "markup_range": 0.0075,
                "n_close_orders": 7,
                "auto_unstuck_wallet_exposure_threshold": 0.1,  # percentage of wallet_exposure_limit to trigger soft stop.
                # e.g. wallet_exposure_limit=0.06 and auto_unstuck_wallet_exposure_threshold=0.1: soft stop when wallet_exposure > 0.06 * (1 - 0.1) == 0.054
                "auto_unstuck_ema_dist": 0.02,
            },
            "short": {
                "enabled": True,
                "ema_span_min": 1440,  # in minutes
                "ema_span_max": 4320,
                "grid_span": 0.16,
                "wallet_exposure_limit": 1.6,
                "max_n_entry_orders": 10,
                "initial_qty_pct": 0.01,
                "initial_eprice_ema_dist": -0.01,  # negative is closer; positive is further away
                "eprice_pprice_diff": 0.0025,
                "secondary_allocation": 0.5,
                "secondary_pprice_diff": 0.35,
                "eprice_exp_base": 1.618034,
                "min_markup": 0.0045,
                "markup_range": 0.0075,
                "n_close_orders": 7,
                "auto_unstuck_wallet_exposure_threshold": 0.1,  # percentage of wallet_exposure_limit to trigger soft stop.
                # e.g. wallet_exposure_limit=0.06 and auto_unstuck_wallet_exposure_threshold=0.1: soft stop when wallet_exposure > 0.06 * (1 - 0.1) == 0.054
                "auto_unstuck_ema_dist": 0.02,
            },
        }
    )


def analyze_fills(fills: list, stats: list, config: dict) -> (pd.DataFrame, pd.DataFrame, dict):
    sdf = pd.DataFrame(
        stats,
        columns=[
            "timestamp",
            "balance",
            "equity",
            "bkr_price",
            "long_psize",
            "long_pprice",
            "short_psize",
            "short_pprice",
            "price",
            "closest_bkr",
            "balance_long",
            "balance_short",
            "equity_long",
            "equity_short",
        ],
    )
    fdf = pd.DataFrame(
        fills,
        columns=[
            "trade_id",
            "timestamp",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    fdf.loc[:, "wallet_exposure"] = [
        qty_to_cost(x.psize, x.pprice, config["inverse"], config["c_mult"]) / x.balance
        if x.balance > 0.0
        else 0.0
        for x in fdf.itertuples()
    ]
    sdf.loc[:, "long_wallet_exposure"] = [
        qty_to_cost(x.long_psize, x.long_pprice, config["inverse"], config["c_mult"]) / x.balance
        if x.balance > 0.0
        else 0.0
        for x in sdf.itertuples()
    ]
    sdf.loc[:, "short_wallet_exposure"] = [
        qty_to_cost(x.short_psize, x.short_pprice, config["inverse"], config["c_mult"]) / x.balance
        if x.balance > 0.0
        else 0.0
        for x in sdf.itertuples()
    ]
    gain = sdf.balance.iloc[-1] / sdf.balance.iloc[0]
    n_days = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[0]) / (1000 * 60 * 60 * 24)
    adg = gain ** (1 / n_days) - 1
    gain -= 1
    fills_per_day = len(fills) / n_days
    long_pos_changes = sdf[sdf.long_psize != sdf.long_psize.shift()]
    long_pos_changes_ms_diff = np.diff(
        [sdf.timestamp.iloc[0]] + list(long_pos_changes.timestamp) + [sdf.timestamp.iloc[-1]]
    )
    hrs_stuck_max_long = long_pos_changes_ms_diff.max() / (1000 * 60 * 60)
    hrs_stuck_avg_long = long_pos_changes_ms_diff.mean() / (1000 * 60 * 60)
    short_pos_changes = sdf[sdf.short_psize != sdf.short_psize.shift()]
    short_pos_changes_ms_diff = np.diff(
        [sdf.timestamp.iloc[0]] + list(short_pos_changes.timestamp) + [sdf.timestamp.iloc[-1]]
    )
    hrs_stuck_max_short = short_pos_changes_ms_diff.max() / (1000 * 60 * 60)
    hrs_stuck_avg_short = short_pos_changes_ms_diff.mean() / (1000 * 60 * 60)
    lpprices = sdf[sdf.long_psize != 0.0]
    spprices = sdf[sdf.short_psize != 0.0]
    pa_distance_long = (
        ((lpprices.long_pprice - lpprices.price).abs() / lpprices.price)
        if len(lpprices) > 0
        else pd.Series([100.0])
    )
    pa_distance_short = (
        ((spprices.short_pprice - spprices.price).abs() / spprices.price)
        if len(spprices) > 0
        else pd.Series([100.0])
    )
    longs = fdf[fdf.type.str.contains("long")].set_index("timestamp")
    shorts = fdf[fdf.type.str.contains("short")].set_index("timestamp")
    gain_long = longs.pnl.sum() / sdf.balance.iloc[0]
    adg_long = (gain_long + 1) ** (1 / n_days) - 1
    gain_short = shorts.pnl.sum() / sdf.balance.iloc[0]
    adg_short = (gain_short + 1) ** (1 / n_days) - 1

    ms2d = 1000 * 60 * 60 * 24
    if len(longs) > 0:
        daily_equity_long = sdf.groupby(sdf.timestamp // ms2d).equity_long.last()
        daily_gains_long = daily_equity_long / daily_equity_long.shift(1) - 1
        adg_long = daily_gains_long.mean()
        DGstd_long = daily_gains_long.std()
        adg_DGstd_ratio_long = adg_long / DGstd_long if len(daily_gains_long) > 0 else 0.0
    else:
        adg_long = adg_DGstd_ratio_long = 0.0
        DGstd_long = 100.0

    if len(shorts) > 0:
        daily_equity_short = sdf.groupby(sdf.timestamp // ms2d).equity_short.last()
        daily_gains_short = daily_equity_short / daily_equity_short.shift(1) - 1
        adg_short = daily_gains_short.mean()
        DGstd_short = daily_gains_short.std()
        adg_DGstd_ratio_short = adg_short / DGstd_short if len(daily_gains_short) > 0 else 0.0
    else:
        adg_short = adg_DGstd_ratio_short = 0.0
        DGstd_short = 100.0

    pos_costs = fdf.apply(
        lambda x: qty_to_cost(x["psize"], x["pprice"], config["inverse"], config["c_mult"]),
        axis=1,
    )
    biggest_pos_cost_long = (
        longs.apply(
            lambda x: qty_to_cost(x["psize"], x["pprice"], config["inverse"], config["c_mult"]),
            axis=1,
        ).max()
        if len(longs) > 0
        else 0.0
    )
    biggest_pos_cost_short = (
        shorts.apply(
            lambda x: qty_to_cost(x["psize"], x["pprice"], config["inverse"], config["c_mult"]),
            axis=1,
        ).max()
        if len(shorts) > 0
        else 0.0
    )
    volume_quote = fdf.apply(
        lambda x: qty_to_cost(x["qty"], x["price"], config["inverse"], config["c_mult"]),
        axis=1,
    ).sum()
    volume_quote_long = (
        longs.apply(
            lambda x: qty_to_cost(x["qty"], x["price"], config["inverse"], config["c_mult"]),
            axis=1,
        ).sum()
        if len(longs) > 0
        else 0.0
    )
    volume_quote_short = (
        shorts.apply(
            lambda x: qty_to_cost(x["qty"], x["price"], config["inverse"], config["c_mult"]),
            axis=1,
        ).sum()
        if len(shorts) > 0
        else 0.0
    )

    analysis = {
        "exchange": config["exchange"] if "exchange" in config else "unknown",
        "symbol": config["symbol"] if "symbol" in config else "unknown",
        "starting_balance": sdf.balance.iloc[0],
        "pa_distance_mean_long": pa_distance_long.mean(),
        "pa_distance_max_long": pa_distance_long.max(),
        "pa_distance_std_long": pa_distance_long.std(),
        "pa_distance_mean_short": pa_distance_short.mean(),
        "pa_distance_max_short": pa_distance_short.max(),
        "pa_distance_std_short": pa_distance_short.std(),
        "gain_long": gain_long,
        "adg_long": adg_long if adg_long == adg_long else -1.0,
        "adg_per_exposure_long": adg_long / config["long"]["wallet_exposure_limit"]
        if config["long"]["enabled"] and config["long"]["wallet_exposure_limit"] > 0.0
        else 0.0,
        "gain_short": gain_short,
        "adg_short": adg_short if adg_short == adg_short else -1.0,
        "adg_per_exposure_short": adg_short / config["short"]["wallet_exposure_limit"]
        if config["short"]["enabled"] and config["short"]["wallet_exposure_limit"] > 0.0
        else 0.0,
        "adg_DGstd_ratio_long": adg_DGstd_ratio_long,
        "adg_DGstd_ratio_short": adg_DGstd_ratio_short,
        "DGstd_long": DGstd_long,
        "DGstd_short": DGstd_short,
        "average_daily_gain": adg,
        "gain": gain,
        "n_days": n_days,
        "n_fills": len(fills),
        "n_entries": len(fdf[fdf.type.str.contains("entry")]),
        "n_closes": len(fdf[fdf.type.str.contains("close")]),
        "n_normal_closes": len(fdf[fdf.type.str.contains("nclose")]),
        "n_normal_closes_long": len(longs[longs.type.str.contains("nclose")]),
        "n_normal_closes_short": len(shorts[shorts.type.str.contains("nclose")]),
        "n_ientries": len(fdf[fdf.type.str.contains("ientry")]),
        "n_ientries_long": len(longs[longs.type.str.contains("ientry")]),
        "n_ientries_short": len(shorts[shorts.type.str.contains("ientry")]),
        "n_rentries": len(fdf[fdf.type.str.contains("rentry")]),
        "n_rentries_long": len(longs[longs.type.str.contains("rentry")]),
        "n_rentries_short": len(shorts[shorts.type.str.contains("rentry")]),
        "n_unstuck_closes": len(fdf[fdf.type.str.contains("unstuck_close")]),
        "n_unstuck_closes_long": len(longs[longs.type.str.contains("unstuck_close")]),
        "n_unstuck_closes_short": len(shorts[shorts.type.str.contains("unstuck_close")]),
        "n_unstuck_entries": len(fdf[fdf.type.str.contains("unstuck_entry")]),
        "n_unstuck_entries_long": len(longs[longs.type.str.contains("unstuck_entry")]),
        "n_unstuck_entries_short": len(shorts[shorts.type.str.contains("unstuck_entry")]),
        "avg_fills_per_day": fills_per_day,
        "avg_fills_per_day_long": len(longs) / n_days,
        "avg_fills_per_day_short": len(shorts) / n_days,
        "hrs_stuck_max_long": hrs_stuck_max_long,
        "hrs_stuck_avg_long": hrs_stuck_avg_long,
        "hrs_stuck_max_short": hrs_stuck_max_short,
        "hrs_stuck_avg_short": hrs_stuck_avg_short,
        "hrs_stuck_max": max(hrs_stuck_max_long, hrs_stuck_max_short),
        "hrs_stuck_avg": max(hrs_stuck_avg_long, hrs_stuck_avg_short),
        "loss_sum": fdf[fdf.pnl < 0.0].pnl.sum(),
        "loss_sum_long": longs[longs.pnl < 0.0].pnl.sum(),
        "loss_sum_short": shorts[shorts.pnl < 0.0].pnl.sum(),
        "profit_sum": fdf[fdf.pnl > 0.0].pnl.sum(),
        "profit_sum_long": longs[longs.pnl > 0.0].pnl.sum(),
        "profit_sum_short": shorts[shorts.pnl > 0.0].pnl.sum(),
        "pnl_sum": (pnl_sum := fdf.pnl.sum()),
        "pnl_sum_long": (pnl_sum_long := longs.pnl.sum()),
        "pnl_sum_short": (pnl_sum_short := shorts.pnl.sum()),
        "fee_sum": (fee_sum := fdf.fee_paid.sum()),
        "fee_sum_long": (fee_sum_long := longs.fee_paid.sum()),
        "fee_sum_short": (fee_sum_short := shorts.fee_paid.sum()),
        "net_pnl_plus_fees": pnl_sum + fee_sum,
        "net_pnl_plus_fees_long": pnl_sum_long + fee_sum_long,
        "net_pnl_plus_fees_short": pnl_sum_short + fee_sum_short,
        "final_equity": sdf.equity.iloc[-1],
        "final_balance": sdf.balance.iloc[-1],
        "closest_bkr": sdf.closest_bkr.min(),
        "eqbal_ratio_min": (eqbal_ratios := sdf.equity / sdf.balance).min(),
        "eqbal_ratio_mean": eqbal_ratios.mean(),
        "biggest_psize": fdf.psize.abs().max(),
        "biggest_psize_long": longs.psize.abs().max(),
        "biggest_psize_short": shorts.psize.abs().max(),
        "biggest_psize_quote": pos_costs.max(),
        "biggest_psize_quote_long": biggest_pos_cost_long,
        "biggest_psize_quote_short": biggest_pos_cost_short,
        "volume_quote": volume_quote,
        "volume_quote_long": volume_quote_long,
        "volume_quote_short": volume_quote_short,
    }
    return fdf, sdf, sort_dict_keys(analysis)


def calc_pprice_from_fills(coin_balance, fills, n_fills_limit=100):
    # assumes fills are sorted old to new
    if coin_balance == 0.0 or len(fills) == 0:
        return 0.0
    relevant_fills = []
    qty_sum = 0.0
    for fill in fills[::-1][:n_fills_limit]:
        abs_qty = fill["qty"]
        if fill["side"] == "buy":
            adjusted_qty = min(abs_qty, coin_balance - qty_sum)
            qty_sum += adjusted_qty
            relevant_fills.append({**fill, **{"qty": adjusted_qty}})
            if qty_sum >= coin_balance * 0.999:
                break
        else:
            qty_sum -= abs_qty
            relevant_fills.append(fill)
    psize, pprice = 0.0, 0.0
    for fill in relevant_fills[::-1]:
        abs_qty = abs(fill["qty"])
        if fill["side"] == "buy":
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill["price"] * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize -= abs_qty
    return pprice


def get_position_fills(long_psize: float, short_psize: float, fills: [dict]) -> ([dict], [dict]):
    """
    assumes fills are sorted old to new
    returns fills since and including initial entry
    """
    long_psize *= 0.999
    short_psize *= 0.999
    long_qty_sum = 0.0
    short_qty_sum = 0.0
    long_done, short_done = long_psize == 0.0, short_psize == 0.0
    if long_done and short_done:
        return [], []
    long_pfills, short_pfills = [], []
    for x in fills[::-1]:
        if x["position_side"] == "long":
            if not long_done:
                long_qty_sum += x["qty"] * (1.0 if x["side"] == "buy" else -1.0)
                long_pfills.append(x)
                long_done = long_qty_sum >= long_psize
        elif x["position_side"] == "short":
            if not short_done:
                short_qty_sum += x["qty"] * (1.0 if x["side"] == "sell" else -1.0)
                short_pfills.append(x)
                short_done = short_qty_sum >= short_psize
    return long_pfills[::-1], short_pfills[::-1]


def calc_long_pprice(long_psize, long_pfills):
    """
    assumes long pfills are sorted old to new
    """
    psize, pprice = 0.0, 0.0
    for fill in long_pfills:
        abs_qty = abs(fill["qty"])
        if fill["side"] == "buy":
            new_psize = psize + abs_qty
            pprice = pprice * (psize / new_psize) + fill["price"] * (abs_qty / new_psize)
            psize = new_psize
        else:
            psize = max(0.0, psize - abs_qty)
    return pprice


def nullify(x):
    if type(x) in [list, tuple]:
        return [nullify(x1) for x1 in x]
    elif type(x) == np.ndarray:
        return numpyize([nullify(x1) for x1 in x])
    elif type(x) == dict:
        return {k: nullify(x[k]) for k in x}
    elif type(x) in [bool, np.bool_]:
        return x
    else:
        return 0.0


def spotify_config(config: dict, nullify_short=True) -> dict:
    spotified = config.copy()

    spotified["spot"] = True
    if "market_type" not in spotified:
        spotified["market_type"] = "spot"
    elif "spot" not in spotified["market_type"]:
        spotified["market_type"] += "_spot"
    spotified["do_long"] = spotified["long"]["enabled"] = config["long"]["enabled"]
    spotified["do_short"] = spotified["short"]["enabled"] = False
    spotified["long"]["wallet_exposure_limit"] = min(1.0, spotified["long"]["wallet_exposure_limit"])
    if nullify_short:
        spotified["short"] = nullify(spotified["short"])
    return spotified


def tuplify(xs, sort=False):
    if type(xs) in [list]:
        if sort:
            return tuple(sorted(tuplify(x, sort=sort) for x in xs))
        return tuple(tuplify(x, sort=sort) for x in xs)
    elif type(xs) in [dict, OrderedDict]:
        if sort:
            return tuple(sorted({k: tuplify(v, sort=sort) for k, v in xs.items()}.items()))
        return tuple({k: tuplify(v, sort=sort) for k, v in xs.items()}.items())
    return xs


def round_values(xs, n: int):
    if type(xs) in [float, np.float64]:
        return round_dynamic(xs, n)
    if type(xs) == dict:
        return {k: round_values(xs[k], n) for k in xs}
    if type(xs) == list:
        return [round_values(x, n) for x in xs]
    if type(xs) == np.ndarray:
        return numpyize([round_values(x, n) for x in xs])
    if type(xs) == tuple:
        return tuple([round_values(x, n) for x in xs])
    if type(xs) == OrderedDict:
        return OrderedDict([(k, round_values(xs[k], n)) for k in xs])
    return xs


def floatify(xs):
    try:
        return float(xs)
    except (ValueError, TypeError):
        if type(xs) == list:
            return [floatify(x) for x in xs]
        if type(xs) == dict:
            return {k: floatify(v) for k, v in xs.items()}
        if type(xs) == tuple:
            return tuple([floatify(x) for x in xs])
    return xs


def get_daily_from_income(
    income: [dict], balance: float, start_time: int = None, end_time: int = None
):
    if start_time is None:
        start_time = income[0]["timestamp"]
    if end_time is None:
        end_time = income[-1]["timestamp"]
    idf = pd.DataFrame(income)
    idf.loc[:, "datetime"] = idf.timestamp.apply(ts_to_date)
    idf.index = idf.timestamp
    ms_per_day = 1000 * 60 * 60 * 24
    days = idf.timestamp // ms_per_day * ms_per_day
    groups = idf.groupby(days)
    daily_income = (
        groups.income.sum()
        .reindex(
            np.arange(
                start_time // ms_per_day * ms_per_day,
                end_time // ms_per_day * ms_per_day + ms_per_day,
                ms_per_day,
            )
        )
        .fillna(0.0)
    )
    cumulative = daily_income.cumsum()
    starting_balance = balance - cumulative.iloc[-1]
    plus_balance = cumulative + starting_balance
    daily_pct = daily_income / plus_balance
    bdf = pd.DataFrame(
        {
            "abs_income": daily_income.values,
            "gain": daily_pct.values,
            "cumulative": plus_balance.values,
        },
        index=[ts_to_date(x) for x in daily_income.index],
    )
    return idf, bdf
