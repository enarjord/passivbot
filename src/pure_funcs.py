import datetime
import pprint
from collections import OrderedDict
from hashlib import sha256
from copy import deepcopy

import json
import re
import numpy as np
import dateutil.parser
from njit_funcs import calc_pnl_long, calc_pnl_short
import passivbot_rust as pbr

try:
    import pandas as pd
except:
    print("pandas not found, trying without...")

    class PD:
        # dummy class when running without pandas
        def __init__(self):
            self.DataFrame = None

    pd = PD()


def safe_filename(symbol: str) -> str:
    """Convert symbol to a safe filename by replacing invalid characters."""
    # Replace / and : with underscores, and remove any other potentially problematic characters
    return re.sub(r'[<>:"/\\|?*]', "_", symbol)


def format_float(num):
    return np.format_float_positional(num, trim="0")


def compress_float(n: float, d: int) -> str:
    if n / 10**d >= 1:
        n = round(n)
    else:
        n = pbr.round_dynamic(n, d)
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


def get_xk_keys(passivbot_mode="neat_grid"):
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
            "auto_unstuck_delay_minutes",
            "auto_unstuck_qty_pct",
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
            "auto_unstuck_delay_minutes",
            "auto_unstuck_qty_pct",
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
    elif all(
        k in config["long"] for k in get_template_live_config("neat_grid")["long"] if k not in skip
    ):
        return "neat_grid"
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
    if config_["passivbot_mode"] in ["neat_grid"]:
        config_["long"]["max_n_entry_orders"] = int(round(config_["long"]["max_n_entry_orders"]))
        config_["short"]["max_n_entry_orders"] = int(round(config_["short"]["max_n_entry_orders"]))
    if config_["passivbot_mode"] in ["clock"]:
        config_["long"]["auto_unstuck_delay_minutes"] = 0.0
        config_["short"]["auto_unstuck_qty_pct"] = 0.0
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
    return int(dateutil.parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def date_to_ts2(datetime_string):
    return (
        dateutil.parser.parse(datetime_string).replace(tzinfo=datetime.timezone.utc).timestamp()
        * 1000
    )


def date2ts_utc(datetime_string):
    return (
        dateutil.parser.parse(datetime_string).replace(tzinfo=datetime.timezone.utc).timestamp()
        * 1000
    )


def date_to_ts2_old(datetime_string):
    try:
        date_formats = [
            "%Y",
            "%Y-%m",
            "%Y-%m-%d",
            "%Y-%m-%dT%H",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%SZ",
        ]
        for format in date_formats:
            try:
                date_obj = datetime.datetime.strptime(datetime_string, format)
                if format == "%Y" or format == "%Y-%m" or format == "%Y-%m-%d":
                    date_obj = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
                timestamp = date_obj.replace(tzinfo=datetime.timezone.utc).timestamp()
                timestamp_ms = int(timestamp * 1000)
                return timestamp_ms
            except ValueError:
                pass
        raise ValueError("Invalid datetime format")
    except Exception as e:
        print("Error:", e)
        return None


def get_day(date):
    # date can be str datetime or float/int timestamp
    try:
        return ts_to_date_utc(date_to_ts2(date))[:10]
    except:
        pass
    try:
        return ts_to_date_utc(date)[:10]
    except:
        pass
    raise Exception(f"failed to get day from {date}")


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
    if isinstance(d, list):
        return [sort_dict_keys(e) for e in d]
    if not isinstance(d, dict):
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


def get_template_live_config(passivbot_mode="v7"):
    if passivbot_mode == "v7":
        return {
            "backtest": {
                "base_dir": "backtests",
                "combine_ohlcvs": True,
                "compress_cache": True,
                "end_date": "now",
                "exchanges": ["binance", "bybit", "gateio", "bitget"],
                "gap_tolerance_ohlcvs_minutes": 120.0,
                "start_date": "2021-04-01",
                "starting_balance": 100000.0,
            },
            "bot": {
                "long": {
                    "close_grid_markup_range": 0.0255,
                    "close_grid_min_markup": 0.0089,
                    "close_grid_qty_pct": 0.125,
                    "close_trailing_grid_ratio": 0.5,
                    "close_trailing_qty_pct": 0.125,
                    "close_trailing_retracement_pct": 0.002,
                    "close_trailing_threshold_pct": 0.008,
                    "ema_span_0": 1318.0,
                    "ema_span_1": 1435.0,
                    "enforce_exposure_limit": True,
                    "entry_grid_double_down_factor": 0.894,
                    "entry_grid_spacing_pct": 0.04,
                    "entry_grid_spacing_weight": 0.697,
                    "entry_initial_ema_dist": -0.00738,
                    "entry_initial_qty_pct": 0.00592,
                    "entry_trailing_grid_ratio": 0.5,
                    "entry_trailing_retracement_pct": 0.01,
                    "entry_trailing_threshold_pct": 0.05,
                    "filter_rolling_window": 60,
                    "filter_relative_volume_clip_pct": 0.95,
                    "n_positions": 10.0,
                    "total_wallet_exposure_limit": 1.7,
                    "unstuck_close_pct": 0.001,
                    "unstuck_ema_dist": 0.0,
                    "unstuck_loss_allowance_pct": 0.03,
                    "unstuck_threshold": 0.916,
                },
                "short": {
                    "close_grid_markup_range": 0.0255,
                    "close_grid_min_markup": 0.0089,
                    "close_grid_qty_pct": 0.125,
                    "close_trailing_grid_ratio": 0.5,
                    "close_trailing_qty_pct": 0.125,
                    "close_trailing_retracement_pct": 0.002,
                    "close_trailing_threshold_pct": 0.008,
                    "ema_span_0": 1318.0,
                    "ema_span_1": 1435.0,
                    "enforce_exposure_limit": True,
                    "entry_grid_double_down_factor": 0.894,
                    "entry_grid_spacing_pct": 0.04,
                    "entry_grid_spacing_weight": 0.697,
                    "entry_initial_ema_dist": -0.00738,
                    "entry_initial_qty_pct": 0.00592,
                    "entry_trailing_grid_ratio": 0.5,
                    "entry_trailing_retracement_pct": 0.01,
                    "entry_trailing_threshold_pct": 0.05,
                    "filter_rolling_window": 60,
                    "filter_relative_volume_clip_pct": 0.95,
                    "n_positions": 10.0,
                    "total_wallet_exposure_limit": 1.7,
                    "unstuck_close_pct": 0.001,
                    "unstuck_ema_dist": 0.0,
                    "unstuck_loss_allowance_pct": 0.03,
                    "unstuck_threshold": 0.916,
                },
            },
            "live": {
                "approved_coins": [],
                "auto_gs": True,
                "coin_flags": {},
                "empty_means_all_approved": False,
                "execution_delay_seconds": 2.0,
                "filter_by_min_effective_cost": True,
                "forced_mode_long": "",
                "forced_mode_short": "",
                "ignored_coins": [],
                "leverage": 10.0,
                "market_orders_allowed": True,
                "max_n_cancellations_per_batch": 5,
                "max_n_creations_per_batch": 3,
                "max_n_restarts_per_day": 10,
                "minimum_coin_age_days": 7.0,
                "ohlcvs_1m_rolling_window_days": 7.0,
                "ohlcvs_1m_update_after_minutes": 10.0,
                "pnls_max_lookback_days": 30.0,
                "price_distance_threshold": 0.002,
                "time_in_force": "good_till_cancelled",
                "user": "bybit_01",
            },
            "optimize": {
                "bounds": {
                    "long_close_grid_markup_range": [0.0, 0.03],
                    "long_close_grid_min_markup": [0.001, 0.03],
                    "long_close_grid_qty_pct": [0.05, 1.0],
                    "long_close_trailing_grid_ratio": [-1.0, 1.0],
                    "long_close_trailing_qty_pct": [0.05, 1.0],
                    "long_close_trailing_retracement_pct": [0.0, 0.1],
                    "long_close_trailing_threshold_pct": [-0.1, 0.1],
                    "long_ema_span_0": [200.0, 1440.0],
                    "long_ema_span_1": [200.0, 1440.0],
                    "long_entry_grid_double_down_factor": [0.1, 3.0],
                    "long_entry_grid_spacing_pct": [0.005, 0.12],
                    "long_entry_grid_spacing_weight": [0.0, 2.0],
                    "long_entry_initial_ema_dist": [-0.1, 0.002],
                    "long_entry_initial_qty_pct": [0.005, 0.1],
                    "long_entry_trailing_grid_ratio": [-1.0, 1.0],
                    "long_entry_trailing_retracement_pct": [0.0, 0.1],
                    "long_entry_trailing_threshold_pct": [-0.1, 0.1],
                    "long_filter_rolling_window": [10.0, 1440.0],
                    "long_filter_relative_volume_clip_pct": [0.0, 1.0],
                    "long_n_positions": [1.0, 20.0],
                    "long_total_wallet_exposure_limit": [0.0, 2.0],
                    "long_unstuck_close_pct": [0.001, 0.1],
                    "long_unstuck_ema_dist": [-0.1, 0.01],
                    "long_unstuck_loss_allowance_pct": [0.001, 0.05],
                    "long_unstuck_threshold": [0.4, 0.95],
                    "short_close_grid_markup_range": [0.0, 0.03],
                    "short_close_grid_min_markup": [0.001, 0.03],
                    "short_close_grid_qty_pct": [0.05, 1.0],
                    "short_close_trailing_grid_ratio": [-1.0, 1.0],
                    "short_close_trailing_qty_pct": [0.05, 1.0],
                    "short_close_trailing_retracement_pct": [0.0, 0.1],
                    "short_close_trailing_threshold_pct": [-0.1, 0.1],
                    "short_ema_span_0": [200.0, 1440.0],
                    "short_ema_span_1": [200.0, 1440.0],
                    "short_entry_grid_double_down_factor": [0.1, 3.0],
                    "short_entry_grid_spacing_pct": [0.005, 0.12],
                    "short_entry_grid_spacing_weight": [0.0, 2.0],
                    "short_entry_initial_ema_dist": [-0.1, 0.002],
                    "short_entry_initial_qty_pct": [0.005, 0.1],
                    "short_entry_trailing_grid_ratio": [-1.0, 1.0],
                    "short_entry_trailing_retracement_pct": [0.0, 0.1],
                    "short_entry_trailing_threshold_pct": [-0.1, 0.1],
                    "short_filter_rolling_window": [10.0, 1440.0],
                    "short_filter_relative_volume_clip_pct": [0.0, 1.0],
                    "short_n_positions": [1.0, 20.0],
                    "short_total_wallet_exposure_limit": [0.0, 2.0],
                    "short_unstuck_close_pct": [0.001, 0.1],
                    "short_unstuck_ema_dist": [-0.1, 0.01],
                    "short_unstuck_loss_allowance_pct": [0.001, 0.05],
                    "short_unstuck_threshold": [0.4, 0.95],
                },
                "compress_results_file": True,
                "crossover_probability": 0.7,
                "iters": 30000,
                "limits": {
                    "lower_bound_drawdown_worst": 0.25,
                    "lower_bound_drawdown_worst_mean_1pct": 0.15,
                    "lower_bound_equity_balance_diff_neg_max": 0.35,
                    "lower_bound_equity_balance_diff_neg_mean": 0.005,
                    "lower_bound_equity_balance_diff_pos_max": 0.5,
                    "lower_bound_equity_balance_diff_pos_mean": 0.01,
                    "lower_bound_loss_profit_ratio": 0.6,
                    "lower_bound_position_held_hours_max": 336.0,
                },
                "mutation_probability": 0.2,
                "n_cpus": 5,
                "population_size": 500,
                "scoring": ["adg", "sharpe_ratio"],
            },
        }
    elif passivbot_mode == "multi_hjson":
        return {
            "user": "bybit_01",
            "pnls_max_lookback_days": 30,
            "loss_allowance_pct": 0.005,
            "stuck_threshold": 0.89,
            "unstuck_close_pct": 0.005,
            "execution_delay_seconds": 2,
            "max_n_cancellations_per_batch": 8,
            "max_n_creations_per_batch": 4,
            "price_distance_threshold": 0.002,
            "filter_by_min_effective_cost": False,
            "auto_gs": True,
            "leverage": 10.0,
            "TWE_long": 2.0,
            "TWE_short": 0.1,
            "long_enabled": True,
            "short_enabled": False,
            "approved_symbols": {
                "COIN1": "-lm n -sm gs -lc configs/live/custom/COIN1USDT.json",
                "COIN2": "-lm n -sm gs -sw 0.4",
                "COIN3": "-lm gs -sm n  -lw 0.15 -lev 12",
            },
            "ignored_symbols": ["COIN4", "COIN5"],
            "n_longs": 0,
            "n_shorts": 0,
            "forced_mode_long": "",
            "forced_mode_short": "",
            "minimum_coin_age_days": 60,
            "ohlcv_interval": "15m",
            "relative_volume_filter_clip_pct": 0.1,
            "n_ohlcvs": 100,
            "live_configs_dir": "configs/live/multisymbol/no_AU/",
            "default_config_path": "configs/live/recursive_grid_mode.example.json",
            "universal_live_config": {
                "long": {
                    "ddown_factor": 0.8783,
                    "ema_span_0": 1054.0,
                    "ema_span_1": 1307.0,
                    "initial_eprice_ema_dist": -0.002641,
                    "initial_qty_pct": 0.01151,
                    "markup_range": 0.0008899,
                    "min_markup": 0.007776,
                    "n_close_orders": 3.724,
                    "rentry_pprice_dist": 0.04745,
                    "rentry_pprice_dist_wallet_exposure_weighting": 0.111,
                },
                "short": {
                    "ddown_factor": 0.8783,
                    "ema_span_0": 1054.0,
                    "ema_span_1": 1307.0,
                    "initial_eprice_ema_dist": -0.002641,
                    "initial_qty_pct": 0.01151,
                    "markup_range": 0.0008899,
                    "min_markup": 0.007776,
                    "n_close_orders": 3.724,
                    "rentry_pprice_dist": 0.04745,
                    "rentry_pprice_dist_wallet_exposure_weighting": 0.111,
                },
            },
        }
    elif passivbot_mode == "multi_json":
        return {
            "analysis": {
                "adg": 0.003956322219722023,
                "adg_weighted": 0.0028311504314386944,
                "drawdowns_daily_mean": 0.017813249628287595,
                "loss_profit_ratio": 0.11029048146750035,
                "loss_profit_ratio_long": 0.11029048146750035,
                "loss_profit_ratio_short": 1.0,
                "n_days": 1073,
                "n_iters": 30100,
                "pnl_ratio_long_short": 1.0,
                "pnl_ratios_symbols": {
                    "AVAXUSDT": 0.27468845221661337,
                    "MATICUSDT": 0.2296174818198483,
                    "SOLUSDT": 0.2953751960870163,
                    "SUSHIUSDT": 0.20031886987652528,
                },
                "price_action_distance_mean": 0.04008693098008042,
                "sharpe_ratio": 0.16774229057551596,
                "w_adg_weighted": -0.0028311504314386944,
                "w_drawdowns_daily_mean": 0.017813249628287595,
                "w_loss_profit_ratio": 0.11029048146750035,
                "w_price_action_distance_mean": 0.04008693098008042,
                "w_sharpe_ratio": -0.16774229057551596,
                "worst_drawdown": 0.4964442148721724,
            },
            "args": {
                "end_date": "2024-04-07",
                "exchange": "binance",
                "long_enabled": True,
                "short_enabled": False,
                "start_date": "2021-05-01",
                "starting_balance": 1000000,
                "symbols": ["AVAXUSDT", "MATICUSDT", "SOLUSDT", "SUSHIUSDT"],
                "worst_drawdown_lower_bound": 0.5,
            },
            "live_config": {
                "global": {
                    "TWE_long": 1.5444230850628553,
                    "TWE_short": 9.649688432169954,
                    "loss_allowance_pct": 0.0026679762307641607,
                    "stuck_threshold": 0.8821459931849173,
                    "unstuck_close_pct": 0.0010155575341165876,
                },
                "long": {
                    "ddown_factor": 2.629714810883098,
                    "ema_span_0": 899.2508850110795,
                    "ema_span_1": 421.7063898877953,
                    "enabled": True,
                    "initial_eprice_ema_dist": -0.1,
                    "initial_qty_pct": 0.014476246820125136,
                    "markup_range": 0.0053184619781202315,
                    "min_markup": 0.007118561833656905,
                    "n_close_orders": 1.8921222249558793,
                    "rentry_pprice_dist": 0.053886357819123286,
                    "rentry_pprice_dist_wallet_exposure_weighting": 2.399828941237894,
                    "wallet_exposure_limit": 0.3861057712657138,
                },
                "short": {
                    "ddown_factor": 2.4945922781706855,
                    "ema_span_0": 455.44131691615075,
                    "ema_span_1": 802.61831996626,
                    "enabled": False,
                    "initial_eprice_ema_dist": -0.1,
                    "initial_qty_pct": 0.010939831544335615,
                    "markup_range": 0.003907075073595213,
                    "min_markup": 0.00126517818899668,
                    "n_close_orders": 3.1853269137597926,
                    "rentry_pprice_dist": 0.04288693053869011,
                    "rentry_pprice_dist_wallet_exposure_weighting": 0.48577214018315135,
                    "wallet_exposure_limit": 2.4124221080424886,
                },
            },
        }
    elif passivbot_mode == "recursive_grid":
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
                    "auto_unstuck_delay_minutes": 300.0,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_qty_pct": 0.04,
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
                    "auto_unstuck_delay_minutes": 300.0,
                    "rentry_pprice_dist": 0.015,
                    "rentry_pprice_dist_wallet_exposure_weighting": 15,
                    "min_markup": 0.02,
                    "markup_range": 0.02,
                    "n_close_orders": 7,
                    "auto_unstuck_qty_pct": 0.04,
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
                    "auto_unstuck_delay_minutes": 300.0,
                    "auto_unstuck_qty_pct": 0.04,
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
                    "auto_unstuck_delay_minutes": 300.0,
                    "auto_unstuck_qty_pct": 0.04,
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
    else:
        raise Exception(f"unknown passivbot mode {passivbot_mode}")


def calc_drawdowns(equity_series):
    """
    Calculate the drawdowns of a portfolio of equities over time.

    Parameters:
    equity_series (pandas.Series): A pandas Series containing the portfolio's equity values over time.

    Returns:
    drawdowns (pandas.Series): The drawdowns as a percentage (expressed as a negative value).
    """
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    # Calculate the cumulative returns of the portfolio
    cumulative_returns = (1 + equity_series.pct_change()).cumprod()

    # Calculate the cumulative maximum value over time
    cumulative_max = cumulative_returns.cummax()

    # Return the drawdown as the percentage decline from the cumulative maximum
    return (cumulative_returns - cumulative_max) / cumulative_max


def calc_max_drawdown(equity_series):
    return calc_drawdowns(equity_series).min()


def calc_sharpe_ratio(equity_series):
    """
    Calculate the Sharpe ratio for a portfolio of equities assuming a zero risk-free rate.

    Parameters:
    equity_series (pandas.Series): A pandas Series containing daily equity values.

    Returns:
    float: The Sharpe ratio.
    """
    if not isinstance(equity_series, pd.Series):
        equity_series = pd.Series(equity_series)

    # Calculate the hourly returns
    returns = equity_series.pct_change().dropna()
    std_dev = returns.std()
    return returns.mean() / std_dev if std_dev != 0.0 else 0.0


def analyze_fills_slim(fills_long: list, fills_short: list, stats: list, config: dict) -> dict:
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

    if "adg_n_subdivisions" not in config:
        config["adg_n_subdivisions"] = 1

    if sdf.balance_long.iloc[-1] <= 0.0:
        adg_long = adg_weighted_long = sdf.balance_long.iloc[-1]
    else:
        adgs_long = []
        for i in range(config["adg_n_subdivisions"]):
            idx = round(int(len(sdf) * (1 - 1 / (i + 1))))
            n_days_ = (sdf.timestamp.iloc[-1] - sdf.timestamp.iloc[idx]) / (1000 * 60 * 60 * 24)
            if n_days_ == 0.0 or sdf.balance_long.iloc[idx] == 0.0:
                adgs_long.append(0.0)
            else:
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
            if n_days_ == 0.0 or sdf.balance_short.iloc[idx] == 0.0:
                adgs_short.append(0.0)
            else:
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
    pa_distance_mean_long = pa_dists_long.mean()
    pa_distance_mean_short = pa_dists_short.mean()
    pa_distance_std_long = pa_dists_long.std()
    pa_distance_std_short = pa_dists_short.std()

    ms_diffs_long = longs.timestamp.diff()
    ms_diffs_short = shorts.timestamp.diff()
    hrs_stuck_max_long = max(
        ms_diffs_long.max(),
        (sdf.iloc[-1].timestamp - longs.iloc[-1].timestamp if len(longs) > 0 else 0.0),
    ) / (1000.0 * 60 * 60)
    hrs_stuck_max_short = max(
        ms_diffs_short.max(),
        (sdf.iloc[-1].timestamp - shorts.iloc[-1].timestamp if len(shorts) > 0 else 0.0),
    ) / (1000.0 * 60 * 60)

    profit_sum_long = longs[longs.pnl > 0.0].pnl.sum()
    loss_sum_long = longs[longs.pnl < 0.0].pnl.sum()

    profit_sum_short = shorts[shorts.pnl > 0.0].pnl.sum()
    loss_sum_short = shorts[shorts.pnl < 0.0].pnl.sum()

    exposure_ratios_long = sdf.wallet_exposure_long / config["long"]["wallet_exposure_limit"]
    time_at_max_exposure_long = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_long[exposure_ratios_long > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_long = exposure_ratios_long.mean()
    exposure_ratios_short = sdf.wallet_exposure_short / config["short"]["wallet_exposure_limit"]
    time_at_max_exposure_short = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_short[exposure_ratios_short > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_short = exposure_ratios_short.mean()

    drawdowns_long = calc_drawdowns(sdf.equity_long)
    drawdown_max_long = drawdowns_long.min()
    mean_of_10_worst_drawdowns = drawdowns_long.sort_values().iloc[:10].mean()

    drawdowns_long = calc_drawdowns(sdf.equity_long)
    drawdowns_short = calc_drawdowns(sdf.equity_short)

    daily_sdf = sdf.groupby(sdf.timestamp // (1000 * 60 * 60 * 24)).last()
    sharpe_ratio_long = calc_sharpe_ratio(daily_sdf.equity_long)
    sharpe_ratio_short = calc_sharpe_ratio(daily_sdf.equity_short)

    return {
        "adg_weighted_per_exposure_long": adg_weighted_long / config["long"]["wallet_exposure_limit"],
        "adg_weighted_per_exposure_short": adg_weighted_short
        / config["short"]["wallet_exposure_limit"],
        "adg_per_exposure_long": adg_long / config["long"]["wallet_exposure_limit"],
        "adg_per_exposure_short": adg_short / config["short"]["wallet_exposure_limit"],
        "n_days": n_days,
        "starting_balance": sdf.balance_long.iloc[0],
        "pa_distance_mean_long": (
            pa_distance_mean_long if pa_distance_mean_long == pa_distance_mean_long else 1.0
        ),
        "pa_distance_max_long": pa_dists_long.max(),
        "pa_distance_std_long": (
            pa_distance_std_long if pa_distance_std_long == pa_distance_std_long else 1.0
        ),
        "pa_distance_mean_short": (
            pa_distance_mean_short if pa_distance_mean_short == pa_distance_mean_short else 1.0
        ),
        "pa_distance_max_short": pa_dists_short.max(),
        "pa_distance_std_short": (
            pa_distance_std_short if pa_distance_std_short == pa_distance_std_short else 1.0
        ),
        "pa_distance_1pct_worst_mean_long": pa_dists_long.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "pa_distance_1pct_worst_mean_short": pa_dists_short.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "hrs_stuck_max_long": hrs_stuck_max_long,
        "hrs_stuck_max_short": hrs_stuck_max_short,
        "loss_profit_ratio_long": (
            abs(loss_sum_long) / profit_sum_long if profit_sum_long > 0.0 else 1.0
        ),
        "loss_profit_ratio_short": (
            abs(loss_sum_short) / profit_sum_short if profit_sum_short > 0.0 else 1.0
        ),
        "exposure_ratios_mean_long": exposure_ratios_mean_long,
        "exposure_ratios_mean_short": exposure_ratios_mean_short,
        "time_at_max_exposure_long": time_at_max_exposure_long,
        "time_at_max_exposure_short": time_at_max_exposure_short,
        "drawdown_max_long": -drawdowns_long.min(),
        "drawdown_max_short": -drawdowns_short.min(),
        "drawdown_1pct_worst_mean_long": -drawdowns_long.sort_values()
        .iloc[: (len(drawdowns_long) // 100)]
        .mean(),
        "drawdown_1pct_worst_mean_short": -drawdowns_short.sort_values()
        .iloc[: (len(drawdowns_short) // 100)]
        .mean(),
        "sharpe_ratio_long": sharpe_ratio_long,
        "sharpe_ratio_short": sharpe_ratio_short,
    }


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
            if n_days_ == 0.0 or sdf.balance_long.iloc[idx] == 0.0:
                adgs_long.append(0.0)
            else:
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
            if n_days_ == 0.0 or sdf.balance_short.iloc[idx] == 0.0:
                adgs_short.append(0.0)
            else:
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

    exposure_ratios_long = sdf.wallet_exposure_long / config["long"]["wallet_exposure_limit"]
    time_at_max_exposure_long = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_long[exposure_ratios_long > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_long = exposure_ratios_long.mean()
    exposure_ratios_short = sdf.wallet_exposure_short / config["short"]["wallet_exposure_limit"]
    time_at_max_exposure_short = (
        1.0 if len(sdf) == 0 else (len(exposure_ratios_short[exposure_ratios_short > 0.9]) / len(sdf))
    )
    exposure_ratios_mean_short = exposure_ratios_short.mean()

    drawdowns_long = calc_drawdowns(sdf.equity_long)
    drawdowns_short = calc_drawdowns(sdf.equity_short)

    daily_sdf = sdf.groupby(sdf.timestamp // (1000 * 60 * 60 * 24)).last()
    sharpe_ratio_long = calc_sharpe_ratio(daily_sdf.equity_long)
    sharpe_ratio_short = calc_sharpe_ratio(daily_sdf.equity_short)

    analysis = {
        "exchange": config["exchange"] if "exchange" in config else "unknown",
        "symbol": config["symbol"] if "symbol" in config else "unknown",
        "starting_balance": sdf.balance_long.iloc[0],
        "pa_distance_mean_long": (
            pa_distance_mean_long if pa_distance_mean_long == pa_distance_mean_long else 1.0
        ),
        "pa_distance_max_long": pa_dists_long.max(),
        "pa_distance_std_long": (
            pa_distance_std_long if pa_distance_std_long == pa_distance_std_long else 1.0
        ),
        "pa_distance_mean_short": (
            pa_distance_mean_short if pa_distance_mean_short == pa_distance_mean_short else 1.0
        ),
        "pa_distance_max_short": pa_dists_short.max(),
        "pa_distance_std_short": (
            pa_distance_std_short if pa_distance_std_short == pa_distance_std_short else 1.0
        ),
        "pa_distance_1pct_worst_mean_long": pa_dists_long.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
        "pa_distance_1pct_worst_mean_short": pa_dists_short.sort_values()
        .iloc[-max(1, len(sdf) // 100) :]
        .mean(),
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
        "adg_weighted_short": (
            adg_weighted_short if adg_weighted_short == adg_weighted_short else -1.0
        ),
        "adg_per_exposure_short": adg_per_exposure_short,
        "adg_weighted_per_exposure_short": adg_weighted_per_exposure_short,
        "exposure_ratios_mean_long": exposure_ratios_mean_long,
        "exposure_ratios_mean_short": exposure_ratios_mean_short,
        "time_at_max_exposure_long": time_at_max_exposure_long,
        "time_at_max_exposure_short": time_at_max_exposure_short,
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
        "loss_profit_ratio_short": (
            (abs(loss_sum_short) / profit_sum_short) if profit_sum_short else 1.0
        ),
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
        "eqbal_ratio_mean_of_10_worst_long": eqbal_ratios_sdf_long.sort_values().iloc[:10].mean(),
        "eqbal_ratio_mean_long": eqbal_ratios_sdf_long.mean(),
        "eqbal_ratio_std_long": eqbal_ratio_std_long,
        "eqbal_ratio_min_short": min(eqbal_ratios_short.min(), eqbal_ratios_sdf_short.min()),
        "eqbal_ratio_mean_of_10_worst_short": eqbal_ratios_sdf_short.sort_values().iloc[:10].mean(),
        "eqbal_ratio_mean_short": eqbal_ratios_sdf_short.mean(),
        "eqbal_ratio_std_short": eqbal_ratio_std_short,
        "volume_quote_long": volume_quote_long,
        "volume_quote_short": volume_quote_short,
        "drawdown_max_long": -drawdowns_long.min(),
        "drawdown_max_short": -drawdowns_short.min(),
        "drawdown_1pct_worst_mean_long": -drawdowns_long.sort_values()
        .iloc[: (len(drawdowns_long) // 100)]
        .mean(),
        "drawdown_1pct_worst_mean_short": -drawdowns_short.sort_values()
        .iloc[: (len(drawdowns_short) // 100)]
        .mean(),
        "sharpe_ratio_long": sharpe_ratio_long,
        "sharpe_ratio_short": sharpe_ratio_short,
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
    returns fills since and including initial entry
    """
    fills = sorted(fills, key=lambda x: x["timestamp"])  # sort old to new
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
        return pbr.round_dynamic(xs, n)
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
    if isinstance(xs, (int, float)):
        return float(xs)
    elif isinstance(xs, str):
        try:
            return float(xs)
        except (ValueError, TypeError):
            return xs
    elif isinstance(xs, bool):
        return xs
    elif isinstance(xs, list):
        return [floatify(x) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(floatify(x) for x in xs)
    elif isinstance(xs, dict):
        return {k: floatify(v) for k, v in xs.items()}
    else:
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
    passivbot_mode = determine_passivbot_mode(
        live_config, skip=["backwards_tp", "auto_unstuck_qty_pct", "auto_unstuck_delay_minutes"]
    )
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
        ]:
            # keep within bounds
            if k0 in live_config[side]:
                live_config[side][k0] = min(ub, max(lb, live_config[side][k0]))

        if passivbot_mode in ["recursive_grid", "neat_grid"]:
            if "initial_eprice_ema_dist" not in live_config[side]:
                live_config[side]["initial_eprice_ema_dist"] = -10.0
            if "auto_unstuck_wallet_exposure_threshold" not in live_config[side]:
                live_config[side]["auto_unstuck_wallet_exposure_threshold"] = 0.0
            if "auto_unstuck_ema_dist" not in live_config[side]:
                live_config[side]["auto_unstuck_ema_dist"] = 0.0
            if "backwards_tp" not in live_config[side]:
                live_config[side]["backwards_tp"] = False
            if "auto_unstuck_delay_minutes" not in live_config[side]:
                live_config[side]["auto_unstuck_delay_minutes"] = 0.0
            if "auto_unstuck_qty_pct" not in live_config[side]:
                live_config[side]["auto_unstuck_qty_pct"] = 0.0
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
        ("exposure_ratios_mean", False),
        ("time_at_max_exposure", False),
        ("pa_distance_mean", False),
        ("pa_distance_std", False),
        ("hrs_stuck_max", False),
        ("pa_distance_1pct_worst_mean", False),
        ("loss_profit_ratio", False),
        ("drawdown_1pct_worst_mean", False),
        ("drawdown_max", False),
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
                    individual_scores[side][sym] += val * (10**i)
                else:
                    individual_scores[side][sym] -= val * (10**i)
            individual_scores[side][sym] *= -1
        raws[side] = {
            key: np.mean([individual_raws[side][sym][key] for sym in results]) for key, _ in keys
        }
        n_symbols_to_include = (
            max(1, int(len(individual_scores[side]) * (1 - config["clip_threshold"])))
            if config["clip_threshold"] < 1.0
            else int(round(config["clip_threshold"]))
        )
        symbols_to_include[side] = sorted(
            individual_scores[side], key=lambda x: individual_scores[side][x]
        )[:n_symbols_to_include]
        # print(symbols_to_include, individual_scores[side], config["clip_threshold"])
        means[side] = {
            key: np.mean([individual_vals[side][sym][key] for sym in symbols_to_include[side]])
            for key, _ in keys
        }
        for i, (key, higher_is_better) in enumerate(keys):
            if higher_is_better:
                scores[side] += means[side][key] * (10**i)
            else:
                scores[side] -= means[side][key] * (10**i)
        scores[side] *= -1
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
    for k_, r_ in [
        ("clock", "clk"),
        ("close", "cls"),
        ("entry", "etr"),
        ("_", ""),
        ("normal", "nrml"),
        ("long", "lng"),
        ("short", "shrt"),
        ("primary", "prm"),
        ("unstuck", "ustk"),
        ("partial", "prtl"),
        ("panic", "pnc"),
    ]:
        id0 = id0.replace(k_, r_)
    return id0


def determine_pos_side_ccxt(open_order: dict) -> str:
    if "info" in open_order:
        oo = open_order["info"]
    else:
        oo = open_order
    if "positionIdx" in oo:  # bybit position
        if float(oo["positionIdx"]) == 1.0:
            return "long"
        if float(oo["positionIdx"]) == 2.0:
            return "short"
    keys_map = {key.lower().replace("_", ""): key for key in oo}
    for poskey in ["posside", "positionside"]:
        if poskey in keys_map:
            return oo[keys_map[poskey]].lower()
    if oo["side"].lower() == "buy":
        if "reduceonly" in keys_map:
            if oo[keys_map["reduceonly"]]:
                return "short"
            else:
                return "long"
        if "closedsize" in keys_map:
            if float(oo[keys_map["closedsize"]]) != 0.0:
                return "short"
            else:
                return "long"
    if oo["side"].lower() == "sell":
        if "reduceonly" in keys_map:
            if oo[keys_map["reduceonly"]]:
                return "long"
            else:
                return "short"
        if "closedsize" in keys_map:
            if float(oo[keys_map["closedsize"]]) != 0.0:
                return "long"
            else:
                return "short"
    for key in ["order_link_id", "clOrdId", "clientOid", "orderLinkId"]:
        if key in oo:
            if "long" in oo[key] or "lng" in oo[key]:
                return "long"
            if "short" in oo[key] or "shrt" in oo[key]:
                return "short"
    return "both"


def calc_hash(data):
    # Convert the data to a JSON string and calculate the SHA-256 hash
    data_string = json.dumps(data, sort_keys=True)
    data_hash = sha256(data_string.encode("utf-8")).hexdigest()
    return data_hash


def stats_multi_to_df(stats, symbols, c_mults):
    dicts = []
    for x in stats:
        d = {"minute": x[0], "balance": x[4], "equity": x[5]}
        for i in range(len(symbols)):
            d[f"{symbols[i]}_psize_l"] = x[1][i][0]
            d[f"{symbols[i]}_pprice_l"] = x[1][i][1]
            d[f"{symbols[i]}_psize_s"] = x[2][i][0]
            d[f"{symbols[i]}_pprice_s"] = x[2][i][1]
            d[f"{symbols[i]}_price"] = x[3][i]

            d[f"{symbols[i]}_upnl_pct_l"] = (
                calc_pnl_long(
                    d[f"{symbols[i]}_pprice_l"],
                    d[f"{symbols[i]}_price"],
                    d[f"{symbols[i]}_psize_l"],
                    False,
                    c_mults[i],
                )
                / d["balance"]
            )
            d[f"{symbols[i]}_upnl_pct_s"] = (
                calc_pnl_short(
                    d[f"{symbols[i]}_pprice_s"],
                    d[f"{symbols[i]}_price"],
                    d[f"{symbols[i]}_psize_s"],
                    False,
                    c_mults[i],
                )
                / d["balance"]
            )

        dicts.append(d)
    sdf = pd.DataFrame(dicts).set_index("minute")
    for s in symbols:
        sdf_tmp = sdf[[c for c in sdf.columns if s in c or "bal" in c]]
        sdf.loc[:, f"{s}_WE_l"] = sdf_tmp[f"{s}_psize_l"] * sdf_tmp[f"{s}_pprice_l"] / sdf.balance
        sdf.loc[:, f"{s}_WE_s"] = (
            sdf_tmp[f"{s}_psize_s"].abs() * sdf_tmp[f"{s}_pprice_s"] / sdf.balance
        )
    return sdf.replace(0.0, np.nan)


def calc_upnl(row, c_mults_d):
    if row.psize == 0.0:
        return 0.0
    if row.psize < 0.0:
        return calc_pnl_short(row.pprice, row.price, row.psize, False, c_mults_d[row.symbol])
    return calc_pnl_long(row.pprice, row.price, row.psize, False, c_mults_d[row.symbol])


def fills_multi_to_df(fills, symbols, c_mults):
    fdf = pd.DataFrame(
        fills,
        columns=[
            "minute",
            "symbol",
            "pnl",
            "fee_paid",
            "balance",
            "equity",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
            "stuckness",
        ],
    )
    s2i = {symbol: i for i, symbol in enumerate(symbols)}
    c_mults_d = {s: c_mults[i] for i, s in enumerate(symbols)}
    c_mults_array = fdf.symbol.apply(lambda s: c_mults[s2i[s]])
    fdf.loc[:, "cost"] = (fdf.qty * fdf.price).abs() * c_mults_array
    fdf.loc[:, "WE"] = (fdf.psize * fdf.pprice).abs() * c_mults_array / fdf.balance
    fdf.loc[:, "upnl_pct"] = fdf.apply(lambda x: calc_upnl(x, c_mults_d), axis=1) / fdf.balance
    return fdf.set_index("minute")


def analyze_fills_multi(sdf, fdf, params):
    symbols = [c[: c.find("_price")] for c in sdf.columns if "_price" in c]
    starting_balance = sdf.iloc[0].balance
    final_balance = sdf.iloc[-1].balance
    final_equity = sdf.iloc[-1].equity
    daily_samples = sdf.groupby(sdf.index // (60 * 24)).last()
    daily_gains = daily_samples.equity.pct_change().dropna()
    n_days = (sdf.index[-1] - sdf.index[0]) / 60 / 24

    eq_threshold = starting_balance * 1e-4
    if daily_samples.equity.iloc[-1] <= eq_threshold:
        # ensure adg is negative if final equity is low
        adg = (max(eq_threshold, final_equity) / starting_balance) ** (1 / n_days) - 1
    else:
        adg = daily_gains.mean()

    adg_weighted = np.mean(
        [
            daily_gains.iloc[round(int(len(daily_gains) * (1 - 1 / (i + 1)))) :].mean()
            for i in range(10)
        ]
    )
    daily_gains_std = daily_gains.std()

    longs = fdf[fdf.type.str.contains("long")]
    shorts = fdf[fdf.type.str.contains("short")]

    sdf_daily = sdf.groupby(sdf.index // (60 * 24)).last()
    sdf_daily.loc[:, "upnl_l"] = sdf_daily[[c for c in sdf_daily if "upnl_l" in c]].sum(axis=1)
    sdf_daily.loc[:, "upnl_s"] = sdf_daily[[c for c in sdf_daily if "upnl_s" in c]].sum(axis=1)

    daily_pnls_long = longs.groupby(longs.index // (60 * 24)).pnl.sum().cumsum()
    daily_pnls_short = shorts.groupby(shorts.index // (60 * 24)).pnl.sum().cumsum()

    daily_pnls_long = daily_pnls_long.reindex(np.arange(sdf.index[0], sdf.index[-1] + 1)).ffill()
    daily_pnls_short = daily_pnls_short.reindex(np.arange(sdf.index[0], sdf.index[-1] + 1)).ffill()

    sdf_daily.loc[:, "equity_long"] = daily_pnls_long + sdf.balance.iloc[0]
    sdf_daily.loc[:, "equity_short"] = daily_pnls_short + sdf.balance.iloc[0]

    adg_long = sdf_daily.equity_long.pct_change().mean()
    adg_weighted_long = np.mean(
        [
            sdf_daily.equity_long.iloc[round(int(len(sdf_daily) * (1 - 1 / (i + 1)))) :]
            .pct_change()
            .mean()
            for i in range(10)
        ]
    )
    adg_weighted_short = np.mean(
        [
            sdf_daily.equity_short.iloc[round(int(len(sdf_daily) * (1 - 1 / (i + 1)))) :]
            .pct_change()
            .mean()
            for i in range(10)
        ]
    )

    adg_short = sdf_daily.equity_short.pct_change().mean()

    pnl_sum = pnl_sum_total = fdf.pnl.sum()

    pnl_long = longs.pnl.sum()
    pnl_short = shorts.pnl.sum()
    sum_profit_long = longs[longs.pnl > 0.0].pnl.sum()
    sum_loss_long = longs[longs.pnl < 0.0].pnl.sum()
    loss_profit_ratio_long = abs(sum_loss_long) / sum_profit_long if sum_profit_long > 0.0 else 1.0
    sum_profit_short = shorts[shorts.pnl > 0.0].pnl.sum()
    sum_loss_short = shorts[shorts.pnl < 0.0].pnl.sum()
    loss_profit_ratio_short = (
        abs(sum_loss_short) / sum_profit_short if sum_profit_short > 0.0 else 1.0
    )

    minute_mult = 60 * 24
    drawdowns = calc_drawdowns(pd.concat([fdf.equity, sdf.equity]).sort_index())
    drawdowns_daily = drawdowns.groupby(drawdowns.index // minute_mult * minute_mult).min()
    drawdowns_mean = abs(drawdowns_daily.mean())
    drawdowns_ten_worst = drawdowns_daily.sort_values().iloc[:10]
    drawdown_max = drawdowns.abs().max()
    mean_of_10_worst_drawdowns = drawdowns_ten_worst.abs().mean()

    is_stuck_long = (
        sdf[[c for c in sdf.columns if "WE_l" in c]] / (params["TWE_long"] / len(symbols)) > 0.9
    )
    is_stuck_short = (
        sdf[[c for c in sdf.columns if "WE_s" in c]] / (params["TWE_short"] / len(symbols)) > 0.9
    )
    any_stuck = pd.concat([is_stuck_long, is_stuck_short], axis=1).any(axis=1)
    stuck_time_ratio_long = len(is_stuck_long[is_stuck_long.any(axis=1)]) / len(is_stuck_long)
    stuck_time_ratio_short = len(is_stuck_short[is_stuck_short.any(axis=1)]) / len(is_stuck_short)
    stuck_time_ratio_any = len(any_stuck[any_stuck]) / len(any_stuck)
    eqbal_ratios = sdf.equity / sdf.balance
    analysis = {
        "n_days": n_days,
        "starting_balance": starting_balance,
        "final_balance": final_balance,
        "final_equity": final_equity,
        "drawdown_max": drawdown_max,
        "drawdown_mean_daily_10_worst": mean_of_10_worst_drawdowns,
        "drawdown_mean_daily": drawdowns_mean,
        "adg": adg,
        "adg_weighted": adg_weighted,
        "adg_long": adg_long,
        "adg_weighted_long": adg_weighted_long,
        "adg_short": adg_short,
        "adg_weighted_short": adg_weighted_short,
        "pnl_sum": pnl_sum,
        "pnl_long": pnl_long,
        "pnl_short": pnl_short,
        "pnl_ratio_long_short": pnl_long / pnl_sum,
        "sum_profit_long": sum_profit_long,
        "sum_profit_short": sum_profit_short,
        "sum_loss_long": sum_loss_long,
        "sum_loss_short": sum_loss_short,
        "loss_profit_ratio_long": loss_profit_ratio_long,
        "loss_profit_ratio_short": loss_profit_ratio_short,
        "stuck_time_ratio_long": stuck_time_ratio_long,
        "stuck_time_ratio_short": stuck_time_ratio_short,
        "stuck_time_ratio_any": stuck_time_ratio_any,
        "eqbal_ratio_mean": eqbal_ratios.mean(),
        "eqbal_ratio_min": eqbal_ratios.min(),
        "n_fills_per_day": len(fdf) / n_days,
        "shape_ratio_daily": adg / daily_gains_std if daily_gains_std != 0.0 else 0.0,
        "individual_analyses": {},
    }
    upnl_pcts_mins = dict(sdf[[c for c in sdf.columns if "upnl" in c]].min().sort_values())
    for symbol in symbols:
        fdfc = fdf[fdf.symbol == symbol]
        longs = fdfc[fdfc.type.str.contains("long")]
        shorts = fdfc[fdfc.type.str.contains("short")]
        pnl_sum = fdfc.pnl.sum()
        pnl_long = longs.pnl.sum()
        pnl_short = shorts.pnl.sum()
        sum_profit_long = longs[longs.pnl > 0.0].pnl.sum()
        sum_loss_long = longs[longs.pnl < 0.0].pnl.sum()
        loss_profit_ratio_long = (
            abs(sum_loss_long) / sum_profit_long if sum_profit_long > 0.0 else 1.0
        )
        sum_profit_short = shorts[shorts.pnl > 0.0].pnl.sum()
        sum_loss_short = shorts[shorts.pnl < 0.0].pnl.sum()
        loss_profit_ratio_short = (
            abs(sum_loss_short) / sum_profit_short if sum_profit_short > 0.0 else 1.0
        )

        stuck_l = is_stuck_long[f"{symbol}_WE_l"]
        stuck_time_ratio_long = len(stuck_l[stuck_l]) / len(stuck_l)
        stuck_s = is_stuck_short[f"{symbol}_WE_s"]
        stuck_time_ratio_short = len(stuck_s[stuck_s]) / len(stuck_s)

        analysis["individual_analyses"][symbol] = {
            "pnl_ratio": pnl_sum / pnl_sum_total,
            "pnl_ratio_long_short": pnl_long / pnl_sum,
            "pnl_long": pnl_long,
            "pnl_short": pnl_short,
            "sum_profit_long": sum_profit_long,
            "sum_loss_long": sum_loss_long,
            "loss_profit_ratio_long": loss_profit_ratio_long,
            "sum_profit_short": sum_profit_short,
            "sum_loss_short": sum_loss_short,
            "loss_profit_ratio_short": loss_profit_ratio_short,
            "stuck_time_ratio_long": stuck_time_ratio_long,
            "stuck_time_ratio_short": stuck_time_ratio_short,
            "n_fills_per_day": len(fdfc) / n_days,
            "upnl_pct_min_long": min(longs.upnl_pct.min(), upnl_pcts_mins[f"{symbol}_upnl_pct_l"]),
            "upnl_pct_min_short": min(shorts.upnl_pct.min(), upnl_pcts_mins[f"{symbol}_upnl_pct_s"]),
        }
    return analysis


def multi_replace(input_data, replacements: [(str, str)]):
    if isinstance(input_data, str):
        new_data = input_data
        for old, new in replacements:
            new_data = new_data.replace(old, new)
    elif isinstance(input_data, list):
        new_data = []
        for string in input_data:
            for old, new in replacements:
                string = string.replace(old, new)
            new_data.append(string)
    elif isinstance(input_data, dict):
        new_data = {}
        for key, string in input_data.items():
            for old, new in replacements:
                string = string.replace(old, new)
            new_data[key] = string
    return new_data


def live_config_dict_to_list_recursive_grid(live_config: dict) -> list:
    keys = [
        ("auto_unstuck_delay_minutes", 0.0),
        ("auto_unstuck_ema_dist", 0.0),
        ("auto_unstuck_qty_pct", 0.0),
        ("auto_unstuck_wallet_exposure_threshold", 0.0),
        ("backwards_tp", 1.0),
        ("ddown_factor", None),
        ("ema_span_0", None),
        ("ema_span_1", None),
        ("enabled", 1.0),
        ("initial_eprice_ema_dist", None),
        ("initial_qty_pct", None),
        ("markup_range", None),
        ("min_markup", None),
        ("n_close_orders", None),
        ("rentry_pprice_dist", None),
        ("rentry_pprice_dist_wallet_exposure_weighting", None),
        ("wallet_exposure_limit", 1.0),
    ]
    tuples = []
    for key, default in keys:
        tpl = {}
        for pside in ["long", "short"]:
            if key in live_config[pside]:
                tpl[pside] = live_config[pside][key]
            else:
                if default is None:
                    raise Exception(f"necessary key missing from live config: {key}")
                tpl[pside] = default
        tuples.append((float(tpl["long"]), float(tpl["short"])))
    return numpyize(tuples)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


def determine_side_from_order_tuple(order_tuple):
    if "long" in order_tuple[2]:
        if "entry" in order_tuple[2]:
            return "buy"
        elif "close" in order_tuple[2]:
            return "sell"
        else:
            raise Exception(f"malformed order tuple {order_tuple}")
    elif "short" in order_tuple[2]:
        if "entry" in order_tuple[2]:
            return "sell"
        elif "close" in order_tuple[2]:
            return "buy"
        else:
            raise Exception(f"malformed order tuple {order_tuple}")
    else:
        raise Exception(f"malformed order tuple {order_tuple}")


def symbol_to_coin(symbol: str) -> str:
    if symbol == "":
        return ""
    if "/" in symbol:
        coin = symbol[: symbol.find("/")]
    else:
        coin = symbol
    for x in ["USDT", "USDC", "BUSD", "USD", "/:"]:
        coin = coin.replace(x, "")
    if "1000" in coin:
        istart = coin.find("1000")
        iend = istart + 1
        while True:
            if iend >= len(coin):
                break
            if coin[iend] != "0":
                break
            iend += 1
        coin = coin[:istart] + coin[iend:]
    if coin.startswith("k") and coin[1:].isupper():
        # hyperliquid uses e.g. kSHIB instead of 1000SHIB
        coin = coin[1:]
    return coin


def coin2symbol(coin: str, quote="USDT") -> str:
    # ccxt formatting
    return f"{coin}/{quote}:{quote}"


def backtested_multiconfig2singleconfig(backtested_config: dict) -> dict:
    template = get_template_live_config("recursive_grid")
    for pside in ["long", "short"]:
        for key, val in [
            ("auto_unstuck_delay_minutes", 0.0),
            ("auto_unstuck_qty_pct", 0.0),
            ("auto_unstuck_wallet_exposure_threshold", 0.0),
            ("auto_unstuck_ema_dist", 0.0),
            ("backwards_tp", True),
        ]:
            template[pside][key] = val
        for key in backtested_config["live_config"][pside]:
            template[pside][key] = backtested_config["live_config"][pside][key]
    template["config_name"] = "_".join(
        [symbol_to_coin(sym) for sym in backtested_config["args"]["symbols"]]
    )
    return template


def backtested_multiconfig2live_multiconfig(backtested_config: dict) -> dict:
    template = get_template_live_config("multi_hjson")
    template["long_enabled"] = backtested_config["args"]["long_enabled"]
    template["short_enabled"] = backtested_config["args"]["short_enabled"]
    template["approved_symbols"] = backtested_config["args"]["symbols"]
    for key in ["live_configs_dir", "default_config_path"]:
        template[key] = ""
    for key in backtested_config["live_config"]["global"]:
        template[key] = backtested_config["live_config"]["global"][key]
    for pside in ["long", "short"]:
        for key in backtested_config["live_config"][pside]:
            if key in template["universal_live_config"][pside]:
                template["universal_live_config"][pside][key] = backtested_config["live_config"][
                    pside
                ][key]
    return template


def add_missing_params_to_hjson_live_multi_config(config: dict) -> (dict, [str]):
    config_copy = deepcopy(config)
    logging_lines = []
    if "approved_symbols" not in config and "symbols" in config:
        logging_lines.append(f"changed 'symbols' -> 'approved_symbols'")
        config_copy["approved_symbols"] = config["symbols"]
    if "universal_live_config" not in config:
        logging_lines.append(f"adding missing config param: 'universal_live_config': {{}}")
        config_copy["universal_live_config"] = {}
    if "minimum_coin_age_days" not in config and "minimum_market_age_days" in config:
        logging_lines.append(f"changed 'minimum_market_age_days' -> 'minimum_coin_age_days'")
        config_copy["minimum_coin_age_days"] = config_copy["minimum_market_age_days"]

    template = get_template_live_config("multi_hjson")
    for key, val in template.items():
        if key not in config_copy:
            logging_lines.append(f"adding missing config param: {key}: {val}")
            config_copy[key] = val
    return config_copy, logging_lines


def remove_OD(d: dict) -> dict:
    if isinstance(d, dict):
        return {k: remove_OD(v) for k, v in d.items()}
    if isinstance(d, list):
        return [remove_OD(x) for x in d]
    return d


def dict_keysort(d: dict):
    return sorted(d.items(), key=lambda x: x[1])


def expand_PB_mode(mode: str) -> str:
    if mode.lower() in ["gs", "graceful_stop", "graceful-stop"]:
        return "graceful_stop"
    elif mode.lower() in ["m", "manual"]:
        return "manual"
    elif mode.lower() in ["n", "normal"]:
        return "normal"
    elif mode.lower() in ["p", "panic"]:
        return "panic"
    elif mode.lower() in ["t", "tp", "tp_only", "tp-only"]:
        return "tp_only"
    else:
        raise Exception(f"unknown passivbot mode {mode}")


def extract_and_sort_by_keys_recursive(nested_dict):
    """
    Extracts values from a nested dictionary of arbitrary depth, sorted by their keys.

    Args:
    nested_dict (dict): A dictionary where each value may be another dictionary.

    Returns:
    list: A list of values, where each value is a list of values from inner dictionaries sorted by their keys.
    """
    if not isinstance(nested_dict, dict):
        return nested_dict

    sorted_values = []
    for key in sorted(nested_dict.keys()):
        value = nested_dict[key]
        sorted_values.append(extract_and_sort_by_keys_recursive(value))

    return sorted_values


def v7_to_v6(config):
    template = get_template_live_config("multi_hjson")
    live_map = {"approved_coins": "approved_symbols"}
    bot_map = {
        "entry_grid_double_down_factor": "ddown_factor",
        "entry_initial_ema_dist": "initial_eprice_ema_dist",
        "entry_initial_qty_pct": "initial_qty_pct",
        "close_grid_markup_range": "markup_range",
        "close_grid_min_markup": "min_markup",
        "entry_grid_spacing_pct": "rentry_pprice_dist",
        "entry_grid_spacing_weight": "rentry_pprice_dist_wallet_exposure_weighting",
        "ema_span_0": "ema_span_0",
        "ema_span_1": "ema_span_1",
    }
    for k, v in config["live"].items():
        if k in template:
            template[k] = v
        elif k in live_map and live_map[k] in template:
            template[live_map[k]] = v
    for pside in ["long", "short"]:
        for k, v in config["bot"][pside].items():
            if k in template["universal_live_config"][pside]:
                template["universal_live_config"][pside][k] = v
            elif k in bot_map and bot_map[k] in template["universal_live_config"][pside]:
                template["universal_live_config"][pside][bot_map[k]] = v
            elif "total_wallet_exposure" in k:
                template[f"TWE_{pside}"] = v
        template["universal_live_config"][pside]["n_close_orders"] = 1.0 / max(
            config["bot"][pside]["close_grid_qty_pct"], 0.05
        )
        template[f"{pside}_enabled"] = (
            config["bot"][pside]["n_positions"] > 0.0
            and config["bot"][pside]["total_wallet_exposure_limit"] > 0.0
        )
        template[f"n_{pside}s"] = config["bot"][pside]["n_positions"]
    template["loss_allowance_pct"] = config["bot"]["long"]["unstuck_loss_allowance_pct"]
    template["unstuck_close_pct"] = config["bot"]["long"]["unstuck_close_pct"]
    template["stuck_threshold"] = config["bot"]["long"]["unstuck_threshold"]

    return template


def hysteresis_rounding(balance, last_rounded_balance, percentage=0.02, h=0.5):
    step = last_rounded_balance * percentage
    threshold = step * h
    if balance > last_rounded_balance + threshold:
        rounded_balance = last_rounded_balance + step
    elif balance < last_rounded_balance - threshold:
        rounded_balance = last_rounded_balance - step
    else:
        rounded_balance = last_rounded_balance
    return pbr.round_dynamic(rounded_balance, 6)


def log_dict_changes(d1, d2, parent_key=""):
    """
    Compare two nested dictionaries and log the changes between them.

    Args:
        dict1: The original dictionary
        dict2: The new dictionary to compare against

    Returns:
        tuple: Lists of added items, removed items, and value changes
    """

    changes = {"added": [], "removed": [], "changed": []}

    # Handle the case where either dictionary is empty
    if not d1:
        changes["added"].extend([f"{parent_key}{k}: {v}" for k, v in d2.items()])
        return changes
    if not d2:
        changes["removed"].extend([f"{parent_key}{k}: {v}" for k, v in d1.items()])
        return changes

    # Compare items in both dictionaries
    for key in set(d1.keys()) | set(d2.keys()):
        new_parent = f"{parent_key}{key}." if parent_key else f"{key}."

        # If key exists in both dictionaries
        if key in d1 and key in d2:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested_changes = log_dict_changes(d1[key], d2[key], new_parent)
                for change_type, items in nested_changes.items():
                    changes[change_type].extend(items)
            elif d1[key] != d2[key]:
                changes["changed"].append(f"{parent_key}{key}: {d1[key]} -> {d2[key]}")
        # If key only exists in dict2
        elif key in d2:
            if isinstance(d2[key], dict):
                nested_changes = log_dict_changes({}, d2[key], new_parent)
                changes["added"].extend(nested_changes["added"])
            else:
                changes["added"].append(f"{parent_key}{key}: {d2[key]}")
        # If key only exists in dict1
        else:
            if isinstance(d1[key], dict):
                nested_changes = log_dict_changes(d1[key], {}, new_parent)
                changes["removed"].extend(nested_changes["removed"])
            else:
                changes["removed"].append(f"{parent_key}{key}: {d1[key]}")

    return changes
