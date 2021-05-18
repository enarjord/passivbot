import sys
import numpy as np


if '--nojit' in sys.argv:
    print('not using numba')
    def njit(pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap
else:
    print('using numba')
    from numba import njit


@njit
def round_dynamic(n: float, d: int):
    if n == 0.0:
        return n
    return round(n, d - int(np.floor(np.log10(abs(n)))) - 1)


def format_float(num):
    return np.format_float_positional(num, trim='-')


def compress_float(n: float, d: int) -> str:
    if n / 10**d >= 1:
        n = round(n)
    else:
        n = round_dynamic(n, d)
    nstr = format_float(n)
    if nstr.startswith('0.'):
        nstr = nstr[1:]
    elif nstr.startswith('-0.'):
        nstr = '-' + nstr[2:]
    elif nstr.endswith('.0'):
        nstr = nstr[:-2]
    return nstr


@njit
def round_up(n, step, safety_rounding=10) -> float:
    return np.round(np.ceil(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_dn(n, step, safety_rounding=10) -> float:
    return np.round(np.floor(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_(n, step, safety_rounding=10) -> float:
    return np.round(np.round(n / step) * step, safety_rounding)


@njit
def calc_diff(x, y):
    return abs(x - y) / abs(y)


@njit
def nan_to_0(x) -> float:
    return x if x == x else 0.0


@njit
def calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost, contract_multiplier) -> float:
    return min_qty if inverse else max(min_qty, round_up(min_cost / price if price > 0.0 else 0.0, qty_step))


@njit
def cost_to_qty(cost, price, inverse, contract_multiplier):
    return cost * price / contract_multiplier if inverse else (cost / price if price > 0.0 else 0.0)


@njit
def qty_to_cost(qty, price, inverse, contract_multiplier) -> float:
    return (abs(qty / price) if price > 0.0 else 0.0) * contract_multiplier if inverse else abs(qty * price)


@njit
def calc_available_margin(balance,
                          long_psize,
                          long_pprice,
                          shrt_psize,
                          shrt_pprice,
                          last_price,
                          inverse, contract_multiplier, leverage) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * contract_multiplier
        equity += calc_long_pnl(long_pprice, last_price, long_psize_real, inverse,
                                contract_multiplier)
        used_margin += calc_cost(long_psize_real, long_pprice,
                                 inverse, contract_multiplier) / leverage
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * contract_multiplier
        equity += calc_shrt_pnl(shrt_pprice, last_price, shrt_psize_real, inverse,
                                contract_multiplier)
        used_margin += calc_cost(shrt_psize_real, shrt_pprice,
                                 inverse, contract_multiplier) / leverage
    return max(0.0, equity - used_margin)


@njit
def queq(value, coeffs, pwr=2):
    return sum([coeff * value**i for coeff, i in zip(coeffs, [pwr - i for i in range(len(coeffs))])])


@njit
def calc_entry_qty():
    min_entry_qty = calc_min_entry_qty(price, inverse, qty_step, min_qty, min_cost, contract_multiplier)
    balance_ito_contracts = cost_to_qty(balance, price, inverse, contract_multiplier)
    max_entry_qty = round_dn(cost_to_qty(available_margin, price, inverse, contract_multiplier), qty_step)
    qty = min(max_entry_qty,
              max(min_entry_qty,
                  round_dn(abs(psize) * ddown_factor + balance_ito_contracts * queq(volatility, v_qty_coeffs), qty_step)))
    return qty if qty >= min_entry_qty else 0.0


@njit
def calc_pprice_pct():
    return queq(volatility, v_pprice_pct_coeffs) + \
           queq(qty_to_cost(psize, price, inverse, contract_multiplier) / balance if balance > 0.0 else 0.0,
                pcost_bal_coeffs)


@njit
def calc_ma_pct():
    return queq(volatility, v_ma_pct_coeffs)


@njit
def calc_ma_pct():
    pass



def get_template_live_config():
    return {
        "config_name": "name",
        "logging_level": 0,
        "long": {
            "enabled": True,
            "ddown_factor": 1.0,
            "v_pprice_pct_coeffs": [0.01, 0.01, 0.01],
            "v_qty_coeffs": [0.01, 0.01, 0.01],
            "v_ma_pct_coeffs": [0.01, 0.01, 0.01]
            "pcost_bal_coeffs": [0.01, 0.01],
            "min_markup": 0.005,
            "markup_range": 0.005,
            "ma_span": 11200.0,
            "counter_order_liq_diff": 0.21,
            "stop_loss_liq_diff": 0.1,
            "stop_loss_pos_pct": 0.05
        },
        "shrt": {
            "enabled": True,
            "ddown_factor": 1.0,
            "v_pprice_pct_coeffs": [0.01, 0.01, 0.01],
            "v_qty_coeffs": [0.01, 0.01, 0.01],
            "v_ma_pct_coeffs": [0.01, 0.01, 0.01]
            "pcost_bal_coeffs": [0.01, 0.01],
            "min_markup": 0.005,
            "markup_range": 0.005,
            "ma_span": 11200.0,
            "counter_order_liq_diff": 0.21,
            "stop_loss_liq_diff": 0.1,
            "stop_loss_pos_pct": 0.05
        }
    }















