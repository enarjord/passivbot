import numpy as np
import pandas as pd


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

        if volatility_expected_period == 0.0:
            SR = 0.0
        else:
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


def get_empty_analysis(bc: dict) -> dict:
    return {
        'net_pnl_plus_fees': 0.0,
        'profit_sum': 0.0,
        'loss_sum': 0.0,
        'fee_sum': 0.0,
        'final_equity': bc['starting_balance'],
        'gain': 1.0,
        'max_drawdown': 0.0,
        'n_days': 0.0,
        'average_daily_gain': 0.0,
        'adjusted_daily_gain': 0.0,
        'lowest_eqbal_ratio': 0.0,
        'closest_bkr': 1.0,
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


def analyze_fills(fills: list, bc: dict, first_ts: float, last_ts: float) -> (pd.DataFrame, dict):
    fdf = pd.DataFrame(fills)

    if fdf.empty:
        return fdf, get_empty_analysis(bc)
    fdf.columns = ['trade_id', 'timestamp', 'pnl', 'fee_paid', 'balance', 'equity', 'pbr', 'qty', 'price', 'psize', 'pprice', 'type']
    fdf = fdf.set_index('trade_id')

    longs = fdf[fdf.type.str.contains('long')]
    shrts = fdf[fdf.type.str.contains('shrt')]

    long_stuck = np.max(np.diff([first_ts] + list(longs.timestamp) + [last_ts])) / (1000 * 60 * 60) if len(longs) > 0 else 1000.0
    shrt_stuck = np.max(np.diff([first_ts] + list(shrts.timestamp) + [last_ts])) / (1000 * 60 * 60) if len(shrts) > 0 else 1000.0

    result = {
        'starting_balance': bc['starting_balance'],
        'final_balance': fdf.iloc[-1].balance,
        'final_equity': fdf.iloc[-1].equity,
        'net_pnl_plus_fees': fdf.pnl.sum() + fdf.fee_paid.sum(),
        'gain': (gain := fdf.iloc[-1].equity / bc['starting_balance']),
        'n_days': (n_days := (last_ts - first_ts) / (1000 * 60 * 60 * 24)),
        'average_daily_gain': (adg := gain ** (1 / n_days) if gain > 0.0 and n_days > 0.0 else 0.0),
        'adjusted_daily_gain': np.tanh(10 * (adg - 1)) + 1,
        'profit_sum': fdf[fdf.pnl > 0.0].pnl.sum(),
        'loss_sum': fdf[fdf.pnl < 0.0].pnl.sum(),
        'fee_sum': fdf.fee_paid.sum(),
        'lowest_eqbal_ratio': bc['lowest_eqbal_ratio'],
        'closest_bkr': bc['closest_bkr'],
        'n_fills': len(fdf),
        'n_entries': len(fdf[fdf.type.str.contains('entry')]),
        'n_closes': len(fdf[fdf.type.str.contains('close')]),
        'n_reentries': len(fdf[fdf.type.str.contains('rentry')]),
        'n_initial_entries': len(fdf[fdf.type.str.contains('ientry')]),
        'n_normal_closes': len(fdf[fdf.type.str.contains('nclose')]),
        'n_stop_loss_closes': len(fdf[fdf.type.str.contains('sclose')]),
        'biggest_psize': fdf.psize.abs().max(),
        'max_hrs_no_fills_long': long_stuck,
        'max_hrs_no_fills_shrt': shrt_stuck,
        'max_hrs_no_fills_same_side': max(long_stuck, shrt_stuck),
        'max_hrs_no_fills': np.max(np.diff([first_ts] + list(fdf.timestamp) + [last_ts])) / (1000 * 60 * 60),
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
