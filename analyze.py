import pandas as pd
import numpy as np


def objective_function(result: dict,
                       metric: str,
                       bc: dict) -> float:
    if result['n_fills'] == 0:
        return -1
    try:
        return (
            result[metric]
            * min(1.0, bc["max_hrs_no_fills"] / result["max_hrs_no_fills"])
            * min(1.0, bc["max_hrs_no_fills_same_side"] / result["max_hrs_no_fills_same_side"])
            * min(1.0, result["closest_liq"] / bc["minimum_liquidation_distance"])
        )
    except:
        return -1


def candidate_to_live_config(candidate: dict) -> dict:
    live_config = {}
    for k in ["config_name", "logging_level", "ddown_factor", "qty_pct", "leverage",
              "n_close_orders", "grid_spacing", "pos_margin_grid_coeff",
              "volatility_grid_coeff", "volatility_qty_coeff", "min_markup",
              "markup_range", "do_long", "do_shrt", "ema_span", "ema_spread", "stop_loss_liq_diff",
              "stop_loss_pos_pct", "symbol"]:
        if k in candidate:
            live_config[k] = candidate[k]
        else:
            live_config[k] = 0.0
    for k in ['do_long', 'do_shrt']:
        live_config[k] = bool(live_config[k])
    return live_config


# TODO: Make a class Returns?
# Dict of interesting periods and their associated number of seconds
PERIODS = {
    'daily': 60 * 60 * 24,
    'weekly': 60 * 60 * 24 * 7,
    'monthly': 60 * 60 * 24 * 365.25 / 12,
    'yearly': 60 * 60 * 24 * 365.25
}

METRICS_OBJ = ["average_daily_gain", "returns_daily", "sharpe_ratio_daily", "VWR_daily"]

def result_sampled_default() -> dict:
    result = {}
    for period, sec in PERIODS.items():
        result["returns_" + period] = 0.0
        result["sharpe_ratio_" + period] = 0.0
        result["VWR_" + period] = 0.0
    return result

def analyze_samples(stats: list, bc: dict) -> (pd.DataFrame, dict):
    sdf = pd.DataFrame(stats).set_index("timestamp")

    if sdf.empty:
        return sdf, result_sampled_default()

    sample_period = "1H"
    sample_sec = pd.to_timedelta(sample_period).seconds

    equity_start = stats[0]["equity"]
    equity_end = stats[-1]["equity"]

    sdf.index = pd.to_datetime(sdf.index, unit="ms")
    sdf = sdf.resample(sample_period).last()


    returns = sdf.equity.pct_change()
    returns[0] = sdf.equity[0] / equity_start - 1
    returns.fillna(0, inplace=True)
    # returns_diff = (sdf['balance'].pad() / (equity_start * np.exp(returns_log_mean * np.arange(1, N+1)))) - 1

    N = len(returns)
    returns_mean = np.exp(np.mean(np.log(returns + 1))) - 1  # Geometrical mean

    #########################################
    ### Variability-Weighted Return (VWR) ###
    #########################################

    # See https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
    returns_log = np.log(1 + returns)
    returns_log_mean = np.log(equity_end / equity_start) / N
    # returns_mean = np.exp(returns_log_mean) - 1 # = geometrical mean != returns.mean()

    # Relative difference of the equity E_i and the zero-variability ideal equity E'_i: (E_i / E'i) - 1
    equity_diff = (sdf["equity"].pad() / (equity_start * np.exp(returns_log_mean * np.arange(1, N + 1)))) - 1

    # Standard deviation of equity differentials
    equity_diff_std = np.std(equity_diff, ddof=1)

    tau = bc["tau"]  # Rate at which weighting falls with increasing variability (investor tolerance)
    sdev_max = bc["sdev_max"]  # Maximum acceptable standard deviation (investor limit)

    # Weighting of the expected compounded returns for a given period (daily, ...). Note that
    # - this factor is always less than 1
    # - this factor is negative if equity_diff_std > sdev_max (hence this parameter name)
    # - the smaller (resp. bigger) tau is the quicker this factor tends to zero (resp. 1)
    VWR_weight = 1.0 - (equity_diff_std / sdev_max) ** tau

    result = {}
    for period, sec in PERIODS.items():
        # There are `periods_nb` times `sample_sec` in `period`
        periods_nb = sec / sample_sec

        # Expected compounded returns for `period` (daily returns = adg - 1)
        #  returns_expected_period = np.exp(returns_log_mean * periods_nb) - 1
        returns_expected_period = (returns_mean + 1) ** periods_nb - 1
        volatility_expected_period = returns.std() * np.sqrt(periods_nb)

        SR = returns_expected_period / volatility_expected_period  # Sharpe ratio (risk-free)
        VWR = returns_expected_period * VWR_weight

        result["returns_" + period] = returns_expected_period

        if equity_end > equity_start:
            result["sharpe_ratio_" + period] = SR
            result["VWR_" + period] = VWR if VWR > 0.0 else 0.0
        else:
            result["sharpe_ratio_" + period] = 0.0
            result["VWR_" + period] = result["returns_" + period]

    return sdf, result

def get_empty_analysis() -> dict:
    return {
        'net_pnl_plus_fees': 0.0,
        'profit_sum': 0.0,
        'loss_sum': 0.0,
        'fee_sum': 0.0,
        'final_equity': 0.0,
        'gain': 0.0,
        'max_drawdown': 0.0,
        'n_days': 0.0,
        'average_daily_gain': 0.0,
        'closest_liq': 0.0,
        'n_fills': 0.0,
        'n_entries': 0.0,
        'n_closes': 0.0,
        'n_reentries': 0.0,
        'n_initial_entries': 0.0,
        'n_normal_closes': 0.0,
        'n_stop_loss_closes': 0.0,
        'n_stop_loss_entries': 0.0,
        'biggest_psize': 0.0,
        'max_hrs_no_fills_same_side': 1000.0,
        'max_hrs_no_fills': 1000.0,
    }


def analyze_fills(fills: dict, bc: dict, last_ts: float) -> (pd.DataFrame, dict):
    fdf = pd.DataFrame(fills)

    if fdf.empty:
        return fdf, get_empty_analysis()

    fdf = fdf.set_index('trade_id')

    if len(longs_ := fdf[fdf.pside == 'long']) > 0:
        long_stuck = np.diff(list(longs_.timestamp) + [last_ts]).max() / (1000 * 60 * 60)
    else:
        long_stuck = 1000.0
    if len(shrts_ := fdf[fdf.pside == 'shrt']) > 0:
        shrt_stuck = np.diff(list(shrts_.timestamp) + [last_ts]).max() / (1000 * 60 * 60)
    else:
        shrt_stuck = 1000.0

    result = {
        'net_pnl_plus_fees': fdf.pnl.sum() + fdf.fee_paid.sum(),
        'profit_sum': fdf[fdf.pnl > 0.0].pnl.sum(),
        'loss_sum': fdf[fdf.pnl < 0.0].pnl.sum(),
        'fee_sum': fdf.fee_paid.sum(),
        'final_equity': fdf.iloc[-1].equity,
        'gain': (gain := fdf.iloc[-1].balance / bc['starting_balance']),
        'max_drawdown': ((fdf.equity - fdf.balance).abs() / fdf.balance).max(),
        'n_days': (n_days := (last_ts - fdf.iloc[0].timestamp) / (1000 * 60 * 60 * 24)),
        'average_daily_gain': gain ** (1 / n_days) if gain > 0.0 and n_days > 0.0 else 0.0,
        'closest_liq': fdf.closest_liq.iloc[-1],
        'n_fills': len(fdf),
        'n_entries': len(fdf[fdf.type.str.contains('entry')]),
        'n_closes': len(fdf[fdf.type.str.contains('close')]),
        'n_reentries': len(fdf[fdf.type.str.contains('reentry')]),
        'n_initial_entries': len(fdf[fdf.type.str.contains('initial')]),
        'n_normal_closes': len(fdf[(fdf.type == 'long_close') | (fdf.type == 'shrt_close')]),
        'n_stop_loss_closes': len(fdf[(fdf.type.str.contains('stop_loss')) &
                                      (fdf.type.str.contains('close'))]),
        'n_stop_loss_entries': len(fdf[(fdf.type.str.contains('stop_loss')) &
                                       (fdf.type.str.contains('entry'))]),
        'biggest_psize': fdf[['long_psize', 'shrt_psize']].abs().max(axis=1).max(),
        'max_hrs_no_fills_long': long_stuck,
        'max_hrs_no_fills_shrt': shrt_stuck,
        'max_hrs_no_fills_same_side': max(long_stuck, shrt_stuck),
        'max_hrs_no_fills': np.diff(list(fdf.timestamp) + [last_ts]).max() / (1000 * 60 * 60),
    }
    return fdf, result


def analyze_backtest(fills: list, stats: list, bc: dict) -> (pd.DataFrame, pd.DataFrame, dict):

    res = {
        "do_long": bool(bc["do_long"]),
        "do_shrt": bool(bc["do_long"]),
        'starting_balance': bc['starting_balance'],
    }

    fdf, res_fill = analyze_fills(fills, bc, stats[-1]['timestamp'] if stats else 0)
    sdf, res_samp = analyze_samples(stats, bc)

    res.update(res_fill)
    res.update(res_samp)

    # Compute the objective from interesting metrics
    for metric in METRICS_OBJ:
        res[metric + "_obj"] = objective_function(res, metric, bc)

    # Compute the objective from the metric defined in the backtesting config
    if bc["metric"] not in res:
        res[bc["metric"] + "_obj"] = objective_function(res, bc["metric"], bc)

    res["objective"] = res[bc["metric"] + "_obj"]

    return fdf, sdf, res
