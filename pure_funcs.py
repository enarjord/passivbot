import datetime
import pprint
from collections import OrderedDict

import json
import numpy as np
from dateutil import parser
from njit_funcs import round_dynamic, qty_to_cost

try:
    import pandas as pd
except:
    print("pandas not found, trying without...")

    class PD:
        # dummy class when running without pandas
        def __init__(self):
            self.DataFrame = None

    pd = PD()


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
            "backwards_tp",
            "qty_step",
            "price_step",
            "min_qty",
            "min_cost",
            "c_mult",
            "ema_span_0",
            "ema_span_1",
            "initial_qty_pct",
            "initial_eprice_ema_dist",
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
    elif passivbot_mode == "neat_grid":
        return [
            "inverse",
            "do_long",
            "do_short",
            "backwards_tp",
            "qty_step",
            "price_step",
            "min_qty",
            "min_cost",
            "c_mult",
            "grid_span",
            "wallet_exposure_limit",
            "max_n_entry_orders",
            "initial_qty_pct",
            "eqty_exp_base",
            "eprice_exp_base",
            "min_markup",
            "markup_range",
            "n_close_orders",
            "ema_span_0",
            "ema_span_1",
            "initial_eprice_ema_dist",
            "auto_unstuck_wallet_exposure_threshold",
            "auto_unstuck_ema_dist",
        ]
    elif passivbot_mode == "clock":
        return [
            "inverse",
            "do_long",
            "do_short",
            "backwards_tp",
            "qty_step",
            "price_step",
            "min_qty",
            "min_cost",
            "c_mult",
            "ema_span_0",
            "ema_span_1",
            "ema_dist_entry",
            "ema_dist_close",
            "qty_pct_entry",
            "qty_pct_close",
            "we_multiplier_entry",
            "we_multiplier_close",
            "delay_weight_entry",
            "delay_weight_close",
            "delay_between_fills_minutes_entry",
            "delay_between_fills_minutes_close",
            "min_markup",
            "markup_range",
            "n_close_orders",
            "wallet_exposure_limit",
        ]
    elif passivbot_mode == "static_grid":
        return [
            "inverse",
            "do_long",
            "do_short",
            "backwards_tp",
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
            "ema_span_0",
            "ema_span_1",
            "initial_eprice_ema_dist",
            "auto_unstuck_wallet_exposure_threshold",
            "auto_unstuck_ema_dist",
        ]
    else:
        raise Exception(f"unknown passivbot mode {passivbot_mode}")


def determine_passivbot_mode(config: dict, skip=[]) -> str:
    # print('dpm devbug',config)
    if all(k in config["long"] for k in get_template_live_config("clock")["long"] if k not in skip):
        return "clock"
    elif all(
        k in config["long"]
        for k in get_template_live_config("recursive_grid")["long"]
        if k not in skip
    ):
        return "recursive_grid"
    if all(
        k in config["long"] for k in get_template_live_config("neat_grid")["long"] if k not in skip
    ):
        return "neat_grid"
    elif all(
        k in config["long"] for k in get_template_live_config("static_grid")["long"] if k not in skip
    ):
        return "static_grid"
    else:
        raise Exception("unable to determine passivbot mode")


def create_xk(config: dict) -> dict:
    xk = {}
    config_ = make_compatible(config.copy())
    config_["passivbot_mode"] = determine_passivbot_mode(config_)
    if "spot" in config_["market_type"]:
        config_ = spotify_config(config_)
    else:
        config_["spot"] = False
        config_["do_long"] = config_["long"]["enabled"]
        config_["do_short"] = config_["short"]["enabled"]
    keys = get_xk_keys(config_["passivbot_mode"])
    config_["long"]["n_close_orders"] = int(round(config_["long"]["n_close_orders"]))
    config_["short"]["n_close_orders"] = int(round(config_["short"]["n_close_orders"]))
    if config_["passivbot_mode"] in ["static_grid", "neat_grid"]:
        config_["long"]["max_n_entry_orders"] = int(round(config_["long"]["max_n_entry_orders"]))
        config_["short"]["max_n_entry_orders"] = int(round(config_["short"]["max_n_entry_orders"]))
    for k in keys:
        if "long" in config_ and k in config_["long"]:
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


def ts_to_date_utc(timestamp: float) -> str:
    if timestamp > 253402297199:
        return str(datetime.datetime.utcfromtimestamp(timestamp / 1000)).replace(" ", "T")
    return str(datetime.datetime.utcfromtimestamp(timestamp)).replace(" ", "T")


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


def candidate_to_live_config(candidate_: dict) -> dict:
    result_dict = candidate_["result"] if "result" in candidate_ else candidate_
    candidate = make_compatible(candidate_)
    passivbot_mode = name = determine_passivbot_mode(candidate)
    live_config = get_template_live_config(passivbot_mode)
    sides = ["long", "short"]
    for side in sides:
        live_config[side]["n_close_orders"] = int(round(live_config[side]["n_close_orders"]))
        for k in live_config[side]:
            if k in candidate[side]:
                live_config[side][k] = candidate[side][k]
            else:
                print(
                    f"warning: {side} {k} missing in config; using default value {live_config[side][k]}"
                )
        for k in live_config:
            if k not in sides and k in candidate:
                live_config[k] = candidate[k]
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
    elif "config_name" in candidate and "days" in candidate["config_name"]:
        try:
            cn = candidate["config_name"]
            for i in range(len(cn) - 1, -1, -1):
                if cn[i] == "_":
                    break
            n_days = int(cn[i + 1 : cn.find("days")])
        except:
            n_days = 0
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
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.02,
                    "wallet_exposure_limit": 1.0,
                    "ddown_factor": 0.6,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.15,
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                },
                "short": {
                    "enabled": False,
                    "ema_span_0": 1036.4758617491368,
                    "ema_span_1": 1125.5167077975314,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.02,
                    "wallet_exposure_limit": 1.0,
                    "ddown_factor": 0.6,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.15,
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                },
            }
        )
    elif passivbot_mode == "neat_grid":
        return sort_dict_keys(
            {
                "config_name": "neat_template",
                "logging_level": 0,
                "long": {
                    "enabled": True,
                    "ema_span_0": 1440,  # in minutes
                    "ema_span_1": 4320,
                    "grid_span": 0.16,
                    "wallet_exposure_limit": 1.6,
                    "max_n_entry_orders": 10,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.01,  # negative is closer; positive is further away
                    "eqty_exp_base": 1.8,
                    "eprice_exp_base": 1.618034,
                    "min_markup": 0.0045,
                    "markup_range": 0.0075,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.1,  # percentage of wallet_exposure_limit to trigger soft stop.
                    # e.g. wallet_exposure_limit=0.06 and auto_unstuck_wallet_exposure_threshold=0.1: soft stop when wallet_exposure > 0.06 * (1 - 0.1) == 0.054
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                },
                "short": {
                    "enabled": True,
                    "ema_span_0": 1440,  # in minutes
                    "ema_span_1": 4320,
                    "grid_span": 0.16,
                    "wallet_exposure_limit": 1.6,
                    "max_n_entry_orders": 10,
                    "initial_qty_pct": 0.01,
                    "initial_eprice_ema_dist": -0.01,  # negative is closer; positive is further away
                    "eqty_exp_base": 1.8,
                    "eprice_exp_base": 1.618034,
                    "min_markup": 0.0045,
                    "markup_range": 0.0075,
                    "n_close_orders": 7,
                    "auto_unstuck_wallet_exposure_threshold": 0.1,  # percentage of wallet_exposure_limit to trigger soft stop.
                    # e.g. wallet_exposure_limit=0.06 and auto_unstuck_wallet_exposure_threshold=0.1: soft stop when wallet_exposure > 0.06 * (1 - 0.1) == 0.054
                    "auto_unstuck_ema_dist": 0.02,
                    "backwards_tp": False,
                },
            }
        )
    elif passivbot_mode == "clock":
        return sort_dict_keys(
            {
                "config_name": "clock_template",
                "long": {
                    "enabled": True,
                    "wallet_exposure_limit": 1.0,
                    "ema_span_0": 700.0,
                    "ema_span_1": 5300.0,
                    "ema_dist_entry": 0.005,
                    "ema_dist_close": 0.005,
                    "qty_pct_entry": 0.01,
                    "qty_pct_close": 0.01,
                    "we_multiplier_entry": 30.0,
                    "we_multiplier_close": 30.0,
                    "delay_weight_entry": 30.0,
                    "delay_weight_close": 30.0,
                    "delay_between_fills_minutes_entry": 2000.0,
                    "delay_between_fills_minutes_close": 2000.0,
                    "min_markup": 0.0075,
                    "markup_range": 0.03,
                    "n_close_orders": 10,
                    "backwards_tp": True,
                },
                "short": {
                    "enabled": True,
                    "wallet_exposure_limit": 1.0,
                    "ema_span_0": 700.0,
                    "ema_span_1": 5300.0,
                    "ema_dist_entry": 0.0039,
                    "ema_dist_close": 0.0045,
                    "qty_pct_entry": 0.013,
                    "qty_pct_close": 0.03,
                    "we_multiplier_entry": 20.0,
                    "we_multiplier_close": 20.0,
                    "delay_weight_entry": 20.0,
                    "delay_weight_close": 20.0,
                    "delay_between_fills_minutes_entry": 2000.0,
                    "delay_between_fills_minutes_close": 2000.0,
                    "min_markup": 0.0075,
                    "markup_range": 0.03,
                    "n_close_orders": 10,
                    "backwards_tp": True,
                },
            }
        )
    elif passivbot_mode == "static_grid":
        return sort_dict_keys(
            {
                "config_name": "static_template",
                "logging_level": 0,
                "long": {
                    "enabled": True,
                    "ema_span_0": 1440,  # in minutes
                    "ema_span_1": 4320,
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
                    "backwards_tp": False,
                },
                "short": {
                    "enabled": True,
                    "ema_span_0": 1440,  # in minutes
                    "ema_span_1": 4320,
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
                    "backwards_tp": False,
                },
            }
        )
    else:
        raise Exception(f"unknown passivbot mode {passivbot_mode}")


def analyze_fills(
    fills_long: list, fills_short: list, stats: list, config: dict
) -> (pd.DataFrame, pd.DataFrame, dict):
    sdf = pd.DataFrame(
        stats,
        columns=[
            "timestamp",
            "bkr_price_long",
            "bkr_price_short",
            "psize_long",
            "pprice_long",
            "psize_short",
            "pprice_short",
            "price",
            "closest_bkr_long",
            "closest_bkr_short",
            "balance_long",
            "balance_short",
            "equity_long",
            "equity_short",
        ],
    )
    longs = pd.DataFrame(
        fills_long,
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
    longs.index = longs.timestamp
    shorts = pd.DataFrame(
        fills_short,
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
    shorts.index = shorts.timestamp
    n_days = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[0]) / 1000 / 60 / 60 / 24.0
    if config["inverse"]:
        longs.loc[:, "pcost"] = (longs.psize / longs.pprice).abs() * config["c_mult"]
        shorts.loc[:, "pcost"] = (shorts.psize / shorts.pprice).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_long"] = (
            sdf.psize_long / sdf.pprice_long / sdf.balance_long
        ).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_short"] = (
            sdf.psize_short / sdf.pprice_short / sdf.balance_short
        ).abs() * config["c_mult"]
    else:
        longs.loc[:, "pcost"] = (longs.psize * longs.pprice).abs() * config["c_mult"]
        shorts.loc[:, "pcost"] = (shorts.psize * shorts.pprice).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_long"] = (
            sdf.psize_long * sdf.pprice_long / sdf.balance_long
        ).abs() * config["c_mult"]
        sdf.loc[:, "wallet_exposure_short"] = (
            sdf.psize_short * sdf.pprice_short / sdf.balance_short
        ).abs() * config["c_mult"]
    longs.loc[:, "wallet_exposure"] = longs.pcost / longs.balance
    shorts.loc[:, "wallet_exposure"] = shorts.pcost / shorts.balance

    ms_diffs_long = longs.timestamp.diff()
    ms_diffs_short = shorts.timestamp.diff()
    longs.loc[:, "mins_since_prev_fill"] = ms_diffs_long / 1000.0 / 60.0
    shorts.loc[:, "mins_since_prev_fill"] = ms_diffs_short / 1000.0 / 60.0

    profit_sum_long = longs[longs.pnl > 0.0].pnl.sum()
    loss_sum_long = longs[longs.pnl < 0.0].pnl.sum()
    pnl_sum_long = profit_sum_long + loss_sum_long
    gain_long = sdf.balance_long.iloc[-1] / sdf.balance_long.iloc[0] - 1

    profit_sum_short = shorts[shorts.pnl > 0.0].pnl.sum()
    loss_sum_short = shorts[shorts.pnl < 0.0].pnl.sum()
    pnl_sum_short = profit_sum_short + loss_sum_short
    gain_short = sdf.balance_short.iloc[-1] / sdf.balance_short.iloc[0] - 1

    # adgs:
    # adg
    # adg_per_exposure
    # adg_weighted
    # adg_weighted_per_exposure

    if "adg_n_subdivisions" not in config:
        config["adg_n_subdivisions"] = 1

    if sdf.balance_long.iloc[-1] <= 0.0:
        adg_long = adg_weighted_long = sdf.balance_long.iloc[-1]
    else:
        adgs_long = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            adgs_long.append(
                (sdf.balance_long.iloc[-1] / sdf.balance_long.iloc[idx]) ** (1 / n_days_) - 1
            )
        adg_long = adgs_long[0]
        adg_weighted_long = np.mean(adgs_long)
    if sdf.balance_short.iloc[-1] <= 0.0:
        adg_short = adg_weighted_short = sdf.balance_short.iloc[-1]

    else:
        adgs_short = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            adgs_short.append(
                (sdf.balance_short.iloc[-1] / sdf.balance_short.iloc[idx]) ** (1 / n_days_) - 1
            )
        adg_short = adgs_short[0]
        adg_weighted_short = np.mean(adgs_short)
    if config["long"]["wallet_exposure_limit"] > 0.0:
        adg_per_exposure_long = adg_long / config["long"]["wallet_exposure_limit"]
        adg_weighted_per_exposure_long = adg_weighted_long / config["long"]["wallet_exposure_limit"]
    else:
        adg_per_exposure_long = adg_weighted_per_exposure_long = 0.0
    if config["short"]["wallet_exposure_limit"] > 0.0:
        adg_per_exposure_short = adg_short / config["short"]["wallet_exposure_limit"]
        adg_weighted_per_exposure_short = (
            adg_weighted_short / config["short"]["wallet_exposure_limit"]
        )
    else:
        adg_per_exposure_short = adg_weighted_per_exposure_short = 0.0

    volume_quote_long = longs.pcost.sum()
    volume_quote_short = shorts.pcost.sum()

    lpprices = sdf[sdf.psize_long != 0.0]
    spprices = sdf[sdf.psize_short != 0.0]
    pa_dists_long = (
        ((lpprices.pprice_long - lpprices.price).abs() / lpprices.price)
        if len(lpprices) > 0
        else pd.Series([100.0])
    )
    pa_dists_short = (
        ((spprices.pprice_short - spprices.price).abs() / spprices.price)
        if len(spprices) > 0
        else pd.Series([100.0])
    )
    pa_distance_std_long = pa_dists_long.std()
    pa_distance_std_short = pa_dists_short.std()
    pa_distance_mean_long = pa_dists_long.mean()
    pa_distance_mean_short = pa_dists_short.mean()

    eqbal_ratios_long = longs.equity / longs.balance
    eqbal_ratios_sdf_long = sdf.equity_long / sdf.balance_long
    eqbal_ratio_std_long = eqbal_ratios_sdf_long.std()
    eqbal_ratios_short = shorts.equity / shorts.balance
    eqbal_ratios_sdf_short = sdf.equity_short / sdf.balance_short
    eqbal_ratio_std_short = eqbal_ratios_sdf_short.std()

    analysis = {
        "exchange": config["exchange"] if "exchange" in config else "unknown",
        "symbol": config["symbol"] if "symbol" in config else "unknown",
        "starting_balance": sdf.balance_long.iloc[0],
        "pa_distance_mean_long": pa_distance_mean_long
        if pa_distance_mean_long == pa_distance_mean_long
        else 1.0,
        "pa_distance_max_long": pa_dists_long.max(),
        "pa_distance_std_long": pa_distance_std_long
        if pa_distance_std_long == pa_distance_std_long
        else 1.0,
        "pa_distance_mean_short": pa_distance_mean_short
        if pa_distance_mean_short == pa_distance_mean_short
        else 1.0,
        "pa_distance_max_short": pa_dists_short.max(),
        "pa_distance_std_short": pa_distance_std_short
        if pa_distance_std_short == pa_distance_std_short
        else 1.0,
        "equity_balance_ratio_mean_long": (sdf.equity_long / sdf.balance_long).mean(),
        "equity_balance_ratio_std_long": eqbal_ratio_std_long,
        "equity_balance_ratio_mean_short": (sdf.equity_short / sdf.balance_short).mean(),
        "equity_balance_ratio_std_short": eqbal_ratio_std_short,
        "gain_long": gain_long,
        "adg_long": adg_long if adg_long == adg_long else -1.0,
        "adg_weighted_long": adg_weighted_long if adg_weighted_long == adg_weighted_long else -1.0,
        "adg_per_exposure_long": adg_per_exposure_long,
        "adg_weighted_per_exposure_long": adg_weighted_per_exposure_long,
        "gain_short": gain_short,
        "adg_short": adg_short if adg_short == adg_short else -1.0,
        "adg_weighted_short": adg_weighted_short
        if adg_weighted_short == adg_weighted_short
        else -1.0,
        "adg_per_exposure_short": adg_per_exposure_short,
        "adg_weighted_per_exposure_short": adg_weighted_per_exposure_short,
        "n_days": n_days,
        "n_fills_long": len(fills_long),
        "n_fills_short": len(fills_short),
        "n_closes_long": len(longs[longs.type.str.contains("close")]),
        "n_closes_short": len(shorts[shorts.type.str.contains("close")]),
        "n_normal_closes_long": len(longs[longs.type.str.contains("nclose")]),
        "n_normal_closes_short": len(shorts[shorts.type.str.contains("nclose")]),
        "n_entries_long": len(longs[longs.type.str.contains("entry")]),
        "n_entries_short": len(shorts[shorts.type.str.contains("entry")]),
        "n_ientries_long": len(longs[longs.type.str.contains("ientry")]),
        "n_ientries_short": len(shorts[shorts.type.str.contains("ientry")]),
        "n_rentries_long": len(longs[longs.type.str.contains("rentry")]),
        "n_rentries_short": len(shorts[shorts.type.str.contains("rentry")]),
        "n_unstuck_closes_long": len(
            longs[longs.type.str.contains("unstuck_close") | longs.type.str.contains("clock_close")]
        ),
        "n_unstuck_closes_short": len(
            shorts[
                shorts.type.str.contains("unstuck_close") | shorts.type.str.contains("clock_close")
            ]
        ),
        "n_unstuck_entries_long": len(
            longs[longs.type.str.contains("unstuck_entry") | longs.type.str.contains("clock_entry")]
        ),
        "n_unstuck_entries_short": len(
            shorts[
                shorts.type.str.contains("unstuck_entry") | shorts.type.str.contains("clock_entry")
            ]
        ),
        "avg_fills_per_day_long": len(longs) / n_days,
        "avg_fills_per_day_short": len(shorts) / n_days,
        "hrs_stuck_max_long": ms_diffs_long.max() / (1000.0 * 60 * 60),
        "hrs_stuck_avg_long": ms_diffs_long.mean() / (1000.0 * 60 * 60),
        "hrs_stuck_max_short": ms_diffs_short.max() / (1000.0 * 60 * 60),
        "hrs_stuck_avg_short": ms_diffs_short.mean() / (1000.0 * 60 * 60),
        "loss_sum_long": loss_sum_long,
        "loss_sum_short": loss_sum_short,
        "profit_sum_long": profit_sum_long,
        "profit_sum_short": profit_sum_short,
        "pnl_sum_long": pnl_sum_long,
        "pnl_sum_short": pnl_sum_short,
        "loss_profit_ratio_long": (abs(loss_sum_long) / profit_sum_long) if profit_sum_long else 1.0,
        "loss_profit_ratio_short": (abs(loss_sum_short) / profit_sum_short)
        if profit_sum_short
        else 1.0,
        "fee_sum_long": (fee_sum_long := longs.fee_paid.sum()),
        "fee_sum_short": (fee_sum_short := shorts.fee_paid.sum()),
        "net_pnl_plus_fees_long": pnl_sum_long + fee_sum_long,
        "net_pnl_plus_fees_short": pnl_sum_short + fee_sum_short,
        "final_equity_long": sdf.equity_long.iloc[-1],
        "final_balance_long": sdf.balance_long.iloc[-1],
        "final_equity_short": sdf.equity_short.iloc[-1],
        "final_balance_short": sdf.balance_short.iloc[-1],
        "closest_bkr_long": sdf.closest_bkr_long.min(),
        "closest_bkr_short": sdf.closest_bkr_short.min(),
        "eqbal_ratio_min_long": min(eqbal_ratios_long.min(), eqbal_ratios_sdf_long.min()),
        "eqbal_ratio_mean_of_10_worst_long": eqbal_ratios_long.sort_values().iloc[:10].mean(),
        "eqbal_ratio_mean_long": eqbal_ratios_sdf_long.mean(),
        "eqbal_ratio_std_long": eqbal_ratio_std_long,
        "eqbal_ratio_min_short": min(eqbal_ratios_short.min(), eqbal_ratios_sdf_short.min()),
        "eqbal_ratio_mean_of_10_worst_short": eqbal_ratios_short.sort_values().iloc[:10].mean(),
        "eqbal_ratio_mean_short": eqbal_ratios_sdf_short.mean(),
        "eqbal_ratio_std_short": eqbal_ratio_std_short,
        "volume_quote_long": volume_quote_long,
        "volume_quote_short": volume_quote_short,
    }
    return longs, shorts, sdf, sort_dict_keys(analysis)


def get_empty_analysis():
    return {
        "exchange": "unknown",
        "symbol": "unknown",
        "starting_balance": 0.0,
        "pa_distance_mean_long": 1000.0,
        "pa_distance_max_long": 1000.0,
        "pa_distance_std_long": 1000.0,
        "pa_distance_mean_short": 1000.0,
        "pa_distance_max_short": 1000.0,
        "pa_distance_std_short": 1000.0,
        "gain_long": 0.0,
        "adg_long": 0.0,
        "adg_per_exposure_long": 0.0,
        "gain_short": 0.0,
        "adg_short": 0.0,
        "adg_per_exposure_short": 0.0,
        "adg_DGstd_ratio_long": 0.0,
        "adg_DGstd_ratio_short": 0.0,
        "adg_realized_long": 0.0,
        "adg_realized_short": 0.0,
        "adg_realized_per_exposure_long": 0.0,
        "adg_realized_per_exposure_short": 0.0,
        "DGstd_long": 0.0,
        "DGstd_short": 0.0,
        "n_days": 0.0,
        "n_fills_long": 0.0,
        "n_fills_short": 0.0,
        "n_closes_long": 0.0,
        "n_closes_short": 0.0,
        "n_normal_closes_long": 0.0,
        "n_normal_closes_short": 0.0,
        "n_entries_long": 0.0,
        "n_entries_short": 0.0,
        "n_ientries_long": 0.0,
        "n_ientries_short": 0.0,
        "n_rentries_long": 0.0,
        "n_rentries_short": 0.0,
        "n_unstuck_closes_long": 0.0,
        "n_unstuck_closes_short": 0.0,
        "n_unstuck_entries_long": 0.0,
        "n_unstuck_entries_short": 0.0,
        "avg_fills_per_day_long": 0.0,
        "avg_fills_per_day_short": 0.0,
        "hrs_stuck_max_long": 1000.0,
        "hrs_stuck_avg_long": 1000.0,
        "hrs_stuck_max_short": 1000.0,
        "hrs_stuck_avg_short": 1000.0,
        "hrs_stuck_max": 1000.0,
        "hrs_stuck_avg": 1000.0,
        "loss_sum_long": 0.0,
        "loss_sum_short": 0.0,
        "profit_sum_long": 0.0,
        "profit_sum_short": 0.0,
        "pnl_sum_long": 0.0,
        "pnl_sum_short": 0.0,
        "fee_sum_long": 0.0,
        "fee_sum_short": 0.0,
        "net_pnl_plus_fees_long": 0.0,
        "net_pnl_plus_fees_short": 0.0,
        "final_equity_long": 0.0,
        "final_balance_long": 0.0,
        "final_equity_short": 0.0,
        "final_balance_short": 0.0,
        "closest_bkr_long": 0.0,
        "closest_bkr_short": 0.0,
        "eqbal_ratio_min_long": 0.0,
        "eqbal_ratio_mean_long": 0.0,
        "eqbal_ratio_min_short": 0.0,
        "eqbal_ratio_mean_short": 0.0,
        "biggest_psize_long": 0.0,
        "biggest_psize_short": 0.0,
        "biggest_psize_quote_long": 0.0,
        "biggest_psize_quote_short": 0.0,
        "volume_quote_long": 0.0,
        "volume_quote_short": 0.0,
    }


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


def get_position_fills(psize_long: float, psize_short: float, fills: [dict]) -> ([dict], [dict]):
    """
    assumes fills are sorted old to new
    returns fills since and including initial entry
    """
    psize_long *= 0.999
    psize_short *= 0.999
    long_qty_sum = 0.0
    short_qty_sum = 0.0
    long_done, short_done = psize_long == 0.0, psize_short == 0.0
    if long_done and short_done:
        return [], []
    long_pfills, short_pfills = [], []
    for x in fills[::-1]:
        if x["position_side"] == "long":
            if not long_done:
                long_qty_sum += x["qty"] * (1.0 if x["side"] == "buy" else -1.0)
                long_pfills.append(x)
                long_done = long_qty_sum >= psize_long
        elif x["position_side"] == "short":
            if not short_done:
                short_qty_sum += x["qty"] * (1.0 if x["side"] == "sell" else -1.0)
                short_pfills.append(x)
                short_done = short_qty_sum >= psize_short
    return long_pfills[::-1], short_pfills[::-1]


def calc_pprice_long(psize_long, long_pfills):
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


def make_compatible(live_config_: dict) -> dict:
    live_config = live_config_.copy()
    for src, dst in [
        ("iprice_ema_dist", "initial_eprice_ema_dist"),
        ("iqty_pct", "initial_qty_pct"),
        ("secondary_grid_spacing", "secondary_pprice_diff"),
        ("shrt", "short"),
        ("secondary_pbr_allocation", "secondary_allocation"),
        ("pbr_limit", "wallet_exposure_limit"),
        ("ema_span_min", "ema_span_0"),
        ("ema_span_max", "ema_span_1"),
    ]:
        live_config = json.loads(json.dumps(live_config).replace(src, dst))
    for side, src, dst in [
        ("long", "ema_dist_lower", "ema_dist_entry"),
        ("long", "ema_dist_upper", "ema_dist_close"),
        ("short", "ema_dist_upper", "ema_dist_entry"),
        ("short", "ema_dist_lower", "ema_dist_close"),
    ]:
        if src in live_config[side]:
            live_config[side][dst] = live_config[side].pop(src)
    passivbot_mode = determine_passivbot_mode(live_config, skip=["backwards_tp"])
    for side in ["long", "short"]:
        for k0 in [
            "delay_weight_close",
            "delay_weight_entry",
            "we_multiplier_close",
            "we_multiplier_entry",
        ]:
            if k0 in live_config[side]:
                # apply abs()
                live_config[side][k0] = abs(live_config[side][k0])
        for k0, lb, ub in [
            ("auto_unstuck_wallet_exposure_threshold", 0.0, 1.0),
            ("auto_unstuck_ema_dist", -10.0, 10.0),
            ("ema_span_0", 1.0, 1000000.0),
            ("ema_span_1", 1.0, 1000000.0),
            ("max_n_entry_orders", 1.0, 100.0),  # don't let's spam the exchange
            ("n_close_orders", 1.0, 100.0),
            ("initial_eprice_ema_dist", -10.0, 10.0),
            ("ema_dist_entry", -10.0, 10.0),
            ("ema_dist_close", -10.0, 10.0),
            ("grid_span", 0.0, 10.0),
            ("delay_between_fills_minutes_entry", 1.0, 1000000.0),  # one million minutes...
            ("delay_between_fills_minutes_close", 1.0, 1000000.0),  #  ...is almost two years
            ("min_markup", 0.0, 10.0),
            ("markup_range", 0.0, 10.0),
            ("wallet_exposure_limit", 0.0, 10000.0),  # 10000x leverage
            ("qty_pct_entry", 0.0, 1.0),  # cannot enter more than whole balance
            ("qty_pct_close", 0.0, 1.0),
            ("initial_qty_pct", 0.0, 1.0),
            ("eqty_exp_base", 0.0, 100.0),
            ("eprice_exp_base", 0.0, 100.0),
            ("ddown_factor", 0.0, 1000.0),
            ("rentry_pprice_dist", 0.0, 100.0),
            ("rentry_pprice_dist_wallet_exposure_weighting", 0.0, 1000000.0),
            ("eprice_pprice_diff", 0.0, 100.0),
            ("eprice_exp_base", 0.0, 100.0),
            ("secondary_allocation", 0.0, 1.0),
            ("secondary_pprice_diff", 0.0, 100.0),
        ]:
            # keep within bounds
            if k0 in live_config[side]:
                live_config[side][k0] = min(ub, max(lb, live_config[side][k0]))

        if passivbot_mode in ["recursive_grid", "static_grid", "neat_grid"]:
            if "initial_eprice_ema_dist" not in live_config[side]:
                live_config[side]["initial_eprice_ema_dist"] = -10.0
            if "auto_unstuck_wallet_exposure_threshold" not in live_config[side]:
                live_config[side]["auto_unstuck_wallet_exposure_threshold"] = 0.0
            if "auto_unstuck_ema_dist" not in live_config[side]:
                live_config[side]["auto_unstuck_ema_dist"] = 0.0
            if "backwards_tp" not in live_config[side]:
                live_config[side]["backwards_tp"] = False
        elif passivbot_mode == "clock":
            if "backwards_tp" not in live_config[side]:
                live_config[side]["backwards_tp"] = True

        if "ema_span_0" not in live_config[side]:
            live_config[side]["ema_span_0"] = 1.0
        if "ema_span_1" not in live_config[side]:
            live_config[side]["ema_span_1"] = 1.0
        live_config[side]["n_close_orders"] = int(round(live_config[side]["n_close_orders"]))
        if "max_n_entry_orders" in live_config[side]:
            live_config[side]["max_n_entry_orders"] = int(
                round(live_config[side]["max_n_entry_orders"])
            )
    return sort_dict_keys(live_config)


def strip_config(cfg: dict) -> dict:
    pm = determine_passivbot_mode(cfg)
    template = get_template_live_config(pm)
    for k in template["long"]:
        template["long"][k] = cfg["long"][k]
        template["short"][k] = cfg["short"][k]
    return template


def calc_scores(config: dict, results: dict):
    sides = ["long", "short"]
    # keys are sorted by reverse importance
    # [(key_name, higher_is_better)]
    keys = [
        ("adg_weighted_per_exposure", True),
        ("hrs_stuck_max", False),
        ("pa_distance_mean", False),
        ("pa_distance_std", False),
        ("loss_profit_ratio", False),
        ("eqbal_ratio_mean_of_10_worst", True),
        ("eqbal_ratio_std", False),
    ]
    means = {side: {} for side in sides}  # adjusted means
    scores = {side: 0.0 for side in sides}
    raws = {side: {} for side in sides}  # unadjusted means
    individual_raws = {side: {sym: {} for sym in results} for side in sides}
    individual_vals = {side: {sym: {} for sym in results} for side in sides}
    individual_scores = {side: {sym: 0.0 for sym in results} for side in sides}
    symbols_to_include = {side: [] for side in sides}
    for side in sides:
        for sym in results:
            for i, (key, higher_is_better) in enumerate(keys):
                key_side = f"{key}_{side}"
                individual_raws[side][sym][key] = results[sym][key_side]
                if (max_key := f"maximum_{key}_{side}") in config:
                    if config[max_key] >= 0.0:
                        val = max(config[max_key], results[sym][key_side])
                    else:
                        val = 0.0
                elif (min_key := f"minimum_{key}_{side}") in config:
                    if config[min_key] >= 0.0:
                        val = min(config[min_key], results[sym][key_side])
                    else:
                        val = 0.0
                else:
                    val = results[sym][key_side]
                individual_vals[side][sym][key] = val
                if higher_is_better:
                    individual_scores[side][sym] += val * (10 ** i)
                else:
                    individual_scores[side][sym] -= val * (10 ** i)
            individual_scores[side][sym] *= -1
        raws[side] = {
            key: np.mean([individual_raws[side][sym][key] for sym in results]) for key, _ in keys
        }
        symbols_to_include[side] = sorted(
            individual_scores[side], key=lambda x: individual_scores[side][x]
        )[: max(1, int(len(individual_scores[side]) * (1 - config["clip_threshold"])))]
        # print(symbols_to_include, individual_scores[side], config["clip_threshold"])
        means[side] = {
            key: np.mean([individual_vals[side][sym][key] for sym in symbols_to_include[side]])
            for key, _ in keys
        }
        for i, (key, higher_is_better) in enumerate(keys):
            if higher_is_better:
                scores[side] += means[side][key] * (10 ** i)
            else:
                scores[side] -= means[side][key] * (10 ** i)
        scores[side] *= -1
    return {
        "scores": scores,
        "means": means,
        "raws": raws,
        "individual_scores": individual_scores,
        "keys": keys,
        "symbols_to_include": symbols_to_include,
    }


def calc_scores_old(config: dict, results: dict):
    sides = ["long", "short"]
    keys = [
        ("adg_realized_per_exposure", True),
        ("pa_distance_std", False),
        ("pa_distance_mean", False),
        ("hrs_stuck_max", False),
        ("loss_profit_ratio", False),
        ("eqbal_ratio_min", True),
    ]
    means = {side: {} for side in sides}  # adjusted means
    scores = {side: -1.0 for side in sides}
    raws = {side: {} for side in sides}  # unadjusted means
    individual_raws = {side: {sym: {} for sym in results} for side in sides}
    individual_vals = {side: {sym: {} for sym in results} for side in sides}
    individual_scores = {side: {sym: -1.0 for sym in results} for side in sides}
    symbols_to_include = {side: [] for side in sides}
    for side in sides:
        for sym in results:
            for key, mult in keys:
                key_side = f"{key}_{side}"
                if key_side not in results[sym]:
                    results[sym][key_side] = results[sym][key]
                individual_raws[side][sym][key] = results[sym][key_side]
                if (max_key := f"maximum_{key}_{side}") in config:
                    if config[max_key] >= 0.0:
                        val = max(config[max_key], results[sym][key_side])
                    else:
                        val = 1.0
                elif (min_key := f"minimum_{key}_{side}") in config:
                    if config[min_key] >= 0.0:
                        val = min(config[min_key], results[sym][key_side])
                    else:
                        val = 1.0
                else:
                    val = results[sym][key_side]
                individual_vals[side][sym][key] = val
                if mult:
                    individual_scores[side][sym] *= val
                else:
                    individual_scores[side][sym] /= val
        raws[side] = {
            key: np.mean([individual_raws[side][sym][key] for sym in results]) for key, _ in keys
        }
        symbols_to_include[side] = sorted(
            individual_scores[side], key=lambda x: individual_scores[side][x]
        )[: max(1, int(len(individual_scores[side]) * (1 - config["clip_threshold"])))]
        # print(symbols_to_include, individual_scores[side], config["clip_threshold"])
        means[side] = {
            key: np.mean([individual_vals[side][sym][key] for sym in symbols_to_include[side]])
            for key, _ in keys
        }
        for key, mult in keys:
            if mult:
                scores[side] *= means[side][key]
            else:
                scores[side] /= means[side][key]
    return {
        "scores": scores,
        "means": means,
        "raws": raws,
        "individual_scores": individual_scores,
        "keys": keys,
        "symbols_to_include": symbols_to_include,
    }


def configs_are_equal(cfg0, cfg1) -> bool:
    try:
        cfg0 = candidate_to_live_config(cfg0)
        cfg1 = candidate_to_live_config(cfg1)
        pm0 = determine_passivbot_mode(cfg0)
        pm1 = determine_passivbot_mode(cfg1)
        if pm0 != pm1:
            return False
        for side in ["long", "short"]:
            for key in cfg0[side]:
                if cfg0[side][key] != cfg1[side][key]:
                    return False
        return True
    except Exception as e:
        print(f"error checking whether configs are equal {e}")
        return False


def shorten_custom_id(id_: str) -> str:
    id0 = id_
    for k_, r_ in [("clock", "clk"), ("close", "cls"), ("entry", "etr"), ("_", "")]:
        id0 = id0.replace(k_, r_)
    return id0
