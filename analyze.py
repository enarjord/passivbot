import numpy as np
import pandas as pd


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

