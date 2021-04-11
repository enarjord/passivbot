import gc
import glob
from hashlib import sha256
from typing import Union

import matplotlib.pyplot as plt
import nevergrad as ng
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch

from downloader import Downloader, prep_backtest_config
from passivbot import *

os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '120'


def dump_plots(result: dict, fdf: pd.DataFrame, df: pd.DataFrame):
    plt.rcParams['figure.figsize'] = [29, 18]
    pd.set_option('precision', 10)

    def gain_conv(x):
        return x * 100 - 100

    lines = []
    lines.append(f"net pnl plus fees {result['net_pnl_plus_fees']:.6f}")
    lines.append(f"profit sum {result['profit_sum']:.6f}")
    lines.append(f"loss sum {result['loss_sum']:.6f}")
    lines.append(f"fee sum {result['fee_sum']:.6f}")
    lines.append(f"gain percentage {gain_conv(result['gain']):.2f}%")
    lines.append(f"n_days {result['n_days']}")
    lines.append(f"average_daily_gain percentage {(result['average_daily_gain'] - 1) * 100:.2f}%")
    lines.append(f"n fills {result['n_fills']}")
    lines.append(f"n entries {result['n_entries']}")
    lines.append(f"n closes {result['n_closes']}")
    lines.append(f"n reentries {result['n_reentries']}")
    lines.append(f"n stop loss closes {result['n_stop_losses']}")
    lines.append(f"biggest_pos_size {round(result['biggest_pos_size'], 10)}")
    lines.append(f"closest liq percentage {result['closest_liq'] * 100:.4f}%")
    lines.append(f"max n hours stuck {result['max_n_hours_stuck']:.2f}")
    lines.append(f"starting balance {result['starting_balance']}")
    lines.append(f"long: {result['do_long']}, short: {result['do_shrt']}")

    live_config = candidate_to_live_settings(result['exchange'], cleanup_candidate(result))
    json.dump(live_config, open(result['session_dirpath'] + 'live_config.json', 'w'), indent=4)

    json.dump(result, open(result['session_dirpath'] + 'result.json', 'w'), indent=4)

    print('plotting price with bid ask entry thresholds')
    ema = df.price.ewm(span=result['ema_span'], adjust=False).mean()
    bids_ = ema * (1 - result['ema_spread'])
    asks_ = ema * (1 + result['ema_spread'])

    plt.clf()
    df.price.iloc[::100].plot()
    bids_.iloc[::100].plot()
    asks_.iloc[::100].plot()
    plt.savefig(f"{result['session_dirpath']}ema_spread_plot.png")

    print('writing backtest_result.txt...')
    with open(f"{result['session_dirpath']}backtest_result.txt", 'w') as f:
        for line in lines:
            print(line)
            f.write(line + '\n')

    print('plotting balance and equity...')
    plt.clf()
    fdf.balance.plot()
    fdf.equity.plot()
    plt.savefig(f"{result['session_dirpath']}balance_and_equity.png")

    print('plotting backtest whole and in chunks...')
    n_parts = 7
    # n_parts = int(round_up(result['n_days'], 1.0))
    for z in range(n_parts):
        start_ = z / n_parts
        end_ = (z + 1) / n_parts
        print(start_, end_)
        fig = plot_fills(df, fdf.iloc[int(len(fdf) * start_):int(len(fdf) * end_)], liq_thr=0.1)
        fig.savefig(f"{result['session_dirpath']}backtest_{z + 1}of{n_parts}.png")
    fig = plot_fills(df, fdf, liq_thr=0.1)
    fig.savefig(f"{result['session_dirpath']}whole_backtest.png")

    print('plotting pos sizes...')
    plt.clf()
    fdf.long_pos_size.plot()
    fdf.shrt_pos_size.plot()
    plt.savefig(f"{result['session_dirpath']}pos_sizes_plot.png")

    print('plotting average daily gain...')
    adg_ = fdf.average_daily_gain
    adg_.index = np.linspace(0.0, 1.0, len(fdf))
    plt.clf()
    adg_c = adg_.iloc[int(len(fdf) * 0.1):]  # skipping first 10%
    print('min max', adg_c.min(), adg_c.max())
    adg_c.plot()
    plt.savefig(f"{result['session_dirpath']}average_daily_gain_plot.png")


def plot_fills(df, fdf, side_: int = 0, liq_thr=0.1):
    plt.clf()

    df.loc[fdf.index[0]:fdf.index[-1]].price.plot(style='y-')

    if side_ >= 0:
        longs = fdf[fdf.pos_side == 'long']
        le = longs[longs.type.str.endswith('entry')]
        lc = longs[longs.type == 'close']
        le.price.plot(style='b.')
        lc.price.plot(style='r.')
        longs.long_pos_price.fillna(method='ffill').plot(style='b--')
    if side_ <= 0:
        shrts = fdf[fdf.pos_side == 'shrt']
        se = shrts[shrts.type.str.endswith('entry')]
        sc = shrts[shrts.type == 'close']
        se.price.plot(style='r.')
        sc.price.plot(style='b.')
        shrts.shrt_pos_price.fillna(method='ffill').plot(style='r--')

    stops = fdf[fdf.type.str.startswith('stop_loss')]
    stops.price.plot(style='gx')
    if 'liq_price' in fdf.columns:
        liq_diff = ((fdf.liq_price - fdf.price).abs() / fdf.price)
        fdf.liq_price.where(liq_diff < liq_thr, np.nan).plot(style='k--')
    return plt


def cleanup_candidate(config: dict) -> dict:
    cleaned = config.copy()
    # for k in cleaned:
    #     if k in config['ranges']:
    #         cleaned[k] = round_(cleaned[k], config['ranges'][k][2])
    if cleaned['ema_span'] != cleaned['ema_span']:
        cleaned['ema_span'] = 1.0
    if cleaned['ema_spread'] != cleaned['ema_spread']:
        cleaned['ema_spread'] = 0.0
    return cleaned


def backtest(config: dict, ticks: np.ndarray, return_fills=False, do_print=False) -> (list, bool):
    long_pos_size, long_pos_price = 0.0, np.nan
    shrt_pos_size, shrt_pos_price = 0.0, np.nan
    liq_price = 0.0
    balance = config['starting_balance']

    pnl_plus_fees_cumsum, loss_cumsum, profit_cumsum, fee_paid_cumsum = 0.0, 0.0, 0.0, 0.0

    if config['inverse']:
        long_pnl_f = calc_long_pnl_inverse
        shrt_pnl_f = calc_shrt_pnl_inverse
        cost_f = calc_cost_inverse
        iter_entries = lambda balance, long_psize, long_pprice, shrt_psize, shrt_pprice, \
                              highest_bid, lowest_ask, last_price, do_long, do_shrt: \
            iter_entries_inverse(config['price_step'], config['qty_step'], config['min_qty'],
                                 config['min_cost'], config['ddown_factor'], config['qty_pct'],
                                 config['leverage'], config['grid_spacing'],
                                 config['grid_coefficient'], balance, long_psize, long_pprice,
                                 shrt_psize, shrt_pprice, highest_bid, lowest_ask, last_price,
                                 do_long, do_shrt)
        iter_long_closes = lambda balance, pos_size, pos_price, lowest_ask: \
            iter_long_closes_inverse(config['price_step'], config['qty_step'], config['min_qty'],
                                     config['min_cost'], config['qty_pct'], config['leverage'],
                                     config['min_markup'], config['markup_range'],
                                     config['n_close_orders'], balance, pos_size, pos_price,
                                     lowest_ask)
        iter_shrt_closes = lambda balance, pos_size, pos_price, highest_bid: \
            iter_shrt_closes_inverse(config['price_step'], config['qty_step'], config['min_qty'],
                                     config['min_cost'], config['qty_pct'], config['leverage'],
                                     config['min_markup'], config['markup_range'],
                                     config['n_close_orders'], balance, pos_size, pos_price,
                                     highest_bid)
        if config['exchange'] == 'binance':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_binance_inverse(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                           config['leverage'],
                                                           contract_size=config['contract_size'])
        elif config['exchange'] == 'bybit':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_bybit_inverse(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                         config['leverage'])
    else:
        long_pnl_f = calc_long_pnl_linear
        shrt_pnl_f = calc_shrt_pnl_linear
        cost_f = calc_cost_linear
        iter_entries = lambda balance, long_psize, long_pprice, shrt_psize, shrt_pprice, \
                              highest_bid, lowest_ask, last_price, do_long, do_shrt: \
            iter_entries_linear(config['price_step'], config['qty_step'], config['min_qty'],
                                config['min_cost'], config['ddown_factor'], config['qty_pct'],
                                config['leverage'], config['grid_spacing'],
                                config['grid_coefficient'], balance, long_psize, long_pprice,
                                shrt_psize, shrt_pprice, highest_bid, lowest_ask, last_price,
                                do_long, do_shrt)

        iter_long_closes = lambda balance, pos_size, pos_price, lowest_ask: \
            iter_long_closes_linear(config['price_step'], config['qty_step'], config['min_qty'],
                                    config['min_cost'], config['qty_pct'], config['leverage'],
                                    config['min_markup'], config['markup_range'],
                                    config['n_close_orders'], balance, pos_size, pos_price,
                                    lowest_ask)
        iter_shrt_closes = lambda balance, pos_size, pos_price, highest_bid: \
            iter_shrt_closes_linear(config['price_step'], config['qty_step'], config['min_qty'],
                                    config['min_cost'], config['qty_pct'], config['leverage'],
                                    config['min_markup'], config['markup_range'],
                                    config['n_close_orders'], balance, pos_size, pos_price,
                                    highest_bid)

        if config['exchange'] == 'binance':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_binance_linear(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                          config['leverage'])
        elif config['exchange'] == 'bybit':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_bybit_linear(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                        config['leverage'])

    prev_long_close_ts, prev_long_entry_ts, prev_long_close_price = 0, 0, 0.0
    prev_shrt_close_ts, prev_shrt_entry_ts, prev_shrt_close_price = 0, 0, 0.0
    min_trade_delay_millis = config['latency_simulation_ms'] \
        if 'latency_simulation_ms' in config else 1000

    all_fills = []
    ob = [min(ticks[0][0], ticks[1][0]),
          max(ticks[0][0], ticks[1][0])]
    ema = ticks[0][0]
    ema_alpha = 2 / (config['ema_span'] + 1)
    ema_alpha_ = 1 - ema_alpha
    # tick tuple: (price, buyer_maker, timestamp)
    for k, tick in enumerate(ticks):
        fills = []
        if tick[1]:
            # maker buy, taker sel
            if config['do_long']:
                if tick[0] <= liq_price and long_pos_size > -shrt_pos_size:
                    if (liq_diff := calc_diff(liq_price, tick[0])) < 0.05:
                        fills.append({
                            'price': tick[0], 'side': 'sel', 'pos_side': 'long',
                            'type': 'liquidation', 'qty': -long_pos_size,
                            'pnl': long_pnl_f(long_pos_price, tick[0], -long_pos_size),
                            'fee_paid': -cost_f(long_pos_size, tick[0]) * config['taker_fee'],
                            'long_pos_size': 0.0, 'long_pos_price': np.nan,
                            'shrt_pos_size': 0.0, 'shrt_pos_price': np.nan, 'liq_diff': liq_diff
                        })
                if long_pos_size == 0.0:
                    if config['ema_span'] > 1.0:
                        highest_bid = calc_initial_long_entry_price(config['price_step'],
                                                                    config['ema_spread'],
                                                                    ema, ob[0])
                    else:
                        highest_bid = ob[0]
                elif tick[0] < long_pos_price:
                    highest_bid = ob[0]
                else:
                    highest_bid = 0.0
                if highest_bid > 0.0 and tick[0] and \
                        tick[2] - prev_long_close_ts > min_trade_delay_millis:
                    # create or add to long pos
                    for tpl in iter_entries(balance, long_pos_size, long_pos_price, shrt_pos_size,
                                            shrt_pos_price, highest_bid, 0.0, tick[0], True, False):
                        if tick[0] < tpl[1]:
                            long_pos_size, long_pos_price = tpl[2], tpl[3]
                            prev_long_entry_ts = tick[2]
                            fills.append({
                                'price': tpl[1], 'side': 'buy', 'pos_side': 'long',
                                'type': 'reentry' if tpl[4] else 'entry', 'qty': tpl[0], 'pnl': 0.0,
                                'fee_paid': -cost_f(tpl[0], tpl[1]) * config['maker_fee'],
                                'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                                'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                                'liq_diff': calc_diff(liq_price, tick[0])
                            })
                        else:
                            break

            if shrt_pos_size < 0.0 and tick[0] < shrt_pos_price and \
                    (tick[2] - prev_shrt_entry_ts > min_trade_delay_millis):
                # close shrt pos
                for qty, price, new_pos_size in iter_shrt_closes(balance, shrt_pos_size,
                                                                 shrt_pos_price, ob[0]):
                    if tick[0] < price:
                        if tick[2] - prev_shrt_close_ts < min_trade_delay_millis and \
                                price >= prev_shrt_close_price:
                            break
                        pnl = shrt_pnl_f(shrt_pos_price, price, qty)
                        shrt_pos_size, prev_shrt_close_ts = new_pos_size, tick[2]
                        if shrt_pos_size == 0.0:
                            shrt_pos_price = np.nan
                        fills.append({
                            'price': price, 'side': 'buy', 'pos_side': 'shrt',
                            'type': 'close', 'qty': qty,
                            'pnl': pnl, 'fee_paid': -cost_f(qty, price) * config['maker_fee'],
                            'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                            'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                            'liq_diff': calc_diff(liq_price, tick[0])
                        })
                        prev_shrt_close_price = price
                    # else:
                    #    break
                    # it would be proper to break here,
                    # unfortunately if done, numba causes memory leak ¯\_(ツ)_/¯
            ob[0] = tick[0]
        else:
            # maker sel, taker buy
            if config['do_shrt']:
                if tick[0] >= liq_price and -shrt_pos_size > long_pos_size:
                    if (liq_diff := calc_diff(liq_price, tick[0])) < 0.1:
                        fills.append({
                            'price': tick[0], 'side': 'buy', 'pos_side': 'shrt',
                            'type': 'liquidation', 'qty': -shrt_pos_size,
                            'pnl': shrt_pnl_f(shrt_pos_price, tick[0], -shrt_pos_size),
                            'fee_paid': -cost_f(shrt_pos_size, tick[0]) * config['taker_fee']
                        })
                if shrt_pos_size == 0.0:
                    if config['ema_span'] > 1.0:
                        lowest_ask = calc_initial_shrt_entry_price(
                            config['price_step'], config['ema_spread'], ema, ob[1]
                        )
                    else:
                        lowest_ask = ob[1]
                elif tick[0] > shrt_pos_price:
                    lowest_ask = ob[1]
                else:
                    lowest_ask = 0.0
                if lowest_ask > 0.0 and tick[2] - prev_shrt_close_ts > min_trade_delay_millis:
                    # create or add to shrt pos
                    for tpl in iter_entries(balance, long_pos_size, long_pos_price, shrt_pos_size,
                                            shrt_pos_price, 0.0, lowest_ask, tick[0], False, True):
                        if tick[0] > tpl[1]:
                            shrt_pos_size, shrt_pos_price, prev_shrt_entry_ts = tpl[2], tpl[3], tick[2]
                            fills.append({
                                'price': tpl[1], 'side': 'sel', 'pos_side': 'shrt',
                                'type': 'reentry' if tpl[4] else 'entry', 'qty': tpl[0], 'pnl': 0.0,
                                'fee_paid': -cost_f(tpl[0], tpl[1]) * config['maker_fee'],
                                'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                                'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                                'liq_diff': calc_diff(liq_price, tick[0])
                            })
                        else:
                            break
            if long_pos_size > 0.0 and tick[0] > long_pos_price and \
                    (tick[2] - prev_long_entry_ts > min_trade_delay_millis):
                # close long pos
                for qty, price, new_pos_size in iter_long_closes(balance, long_pos_size,
                                                                 long_pos_price, ob[1]):
                    if tick[0] > price:
                        if tick[2] - prev_long_close_ts < min_trade_delay_millis and \
                                price <= prev_long_close_price:
                            break
                        pnl = long_pnl_f(long_pos_price, price, qty)
                        long_pos_size, prev_long_close_ts = new_pos_size, tick[2]
                        if long_pos_size == 0.0:
                            long_pos_price = np.nan
                        fills.append({
                            'price': price, 'side': 'sel', 'pos_side': 'long',
                            'type': 'close', 'qty': qty,
                            'pnl': pnl, 'fee_paid': -cost_f(qty, price) * config['maker_fee'],
                            'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                            'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                            'liq_diff': calc_diff(liq_price, tick[0])
                        })
                        prev_long_close_price = price
            ob[1] = tick[0]
        ema = calc_ema(ema_alpha, ema_alpha_, ema, tick[0])
        if len(fills) > 0:
            for fill in fills:
                balance += fill['pnl'] + fill['fee_paid']
                liq_price = liq_price_f(balance, long_pos_size, long_pos_price,
                                        shrt_pos_size, shrt_pos_price)
                ms_since_long_pos_change = tick[2] - prev_long_fill_ts \
                    if (prev_long_fill_ts := max(prev_long_close_ts, prev_long_entry_ts)) > 0 else 0
                ms_since_shrt_pos_change = tick[2] - prev_shrt_fill_ts \
                    if (prev_shrt_fill_ts := max(prev_shrt_close_ts, prev_shrt_entry_ts)) > 0 else 0

                if fill['type'].startswith('stop_loss'):
                    loss_cumsum += fill['pnl']
                else:
                    profit_cumsum += fill['pnl']
                fee_paid_cumsum += fill['fee_paid']
                pnl_plus_fees_cumsum += fill['pnl'] + fill['fee_paid']
                upnl_l = x if (x := long_pnl_f(long_pos_price, tick[0], long_pos_size)) == x else 0.0
                upnl_s = y if (y := shrt_pnl_f(shrt_pos_price, tick[0], shrt_pos_size)) == y else 0.0
                fill['equity'] = balance + upnl_l + upnl_s
                fill['pnl_plus_fees_cumsum'] = pnl_plus_fees_cumsum
                fill['loss_cumsum'] = loss_cumsum
                fill['profit_cumsum'] = profit_cumsum
                fill['fee_paid_cumsum'] = fee_paid_cumsum

                fill['balance'] = balance
                fill['liq_price'] = liq_price
                fill['timestamp'] = tick[2]
                fill['trade_id'] = k
                fill['progress'] = k / len(ticks)
                fill['drawdown'] = calc_diff(fill['balance'], fill['equity'])
                fill['gain'] = fill['equity'] / config['starting_balance']
                fill['n_days'] = (tick[2] - ticks[0][2]) / (1000 * 60 * 60 * 24)
                try:
                    fill['average_daily_gain'] = fill['gain'] ** (1 / fill['n_days']) \
                        if (fill['n_days'] > 0.5 and fill['gain'] > 0.0) else 0.0
                except:
                    fill['average_daily_gain'] = 0.0
                fill['hours_since_long_pos_change'] = (lc := ms_since_long_pos_change / (1000 * 60 * 60))
                fill['hours_since_shrt_pos_change'] = (sc := ms_since_shrt_pos_change / (1000 * 60 * 60))
                fill['hours_since_pos_change_max'] = max(lc, sc)
                all_fills.append(fill)
                if balance <= 0.0 or fill['type'] == 'liquidation':
                    if return_fills:
                        return all_fills, False
                    else:
                        result = prepare_result(all_fills, ticks, config['do_long'], config['do_shrt'])
                        objective = objective_function(result, config['minimum_liquidation_distance'],
                                                       config['maximum_daily_entries'])
                        tune.report(objective=objective)
                        del all_fills
                        gc.collect()
                        return objective
            if do_print:
                line = f"\r{all_fills[-1]['progress']:.3f} "
                line += f"adg {all_fills[-1]['average_daily_gain']:.4f} "
                print(line, end=' ')
            # print(f"\r{k / len(ticks):.2f} ", end=' ')
    if return_fills:
        return all_fills, True
    else:
        result = prepare_result(all_fills, ticks, config['do_long'], config['do_shrt'])
        objective = objective_function(result, config['minimum_liquidation_distance'], config['maximum_daily_entries'])
        tune.report(objective=objective)
        del all_fills
        gc.collect()
        return objective


def candidate_to_live_settings(exchange: str, candidate: dict) -> dict:
    live_settings = load_live_settings(exchange, do_print=False)
    for k in candidate:
        if k in live_settings:
            live_settings[k] = candidate[k]
    live_settings['config_name'] = candidate['session_name']
    live_settings['symbol'] = candidate['symbol']
    if live_settings['ema_span'] != live_settings['ema_span']:
        live_settings['ema_span'] = 1.0
    if live_settings['ema_spread'] != live_settings['ema_spread']:
        live_settings['ema_spread'] = 0.0
    for k in ['do_long', 'do_shrt']:
        live_settings[k] = bool(candidate[k])
    return live_settings


def calc_candidate_hash_key(candidate: dict, keys: [str]) -> str:
    return sha256(json.dumps({k: candidate[k] for k in sorted(keys)
                              if k in candidate}).encode()).hexdigest()


def backtest_wrap(ticks: [dict], backtest_config: dict, do_print=False) -> (dict, pd.DataFrame):
    start_ts = time()
    fills, did_finish = backtest(backtest_config, ticks, return_fills=True, do_print=do_print)
    elapsed = time() - start_ts
    if len(fills) == 0:
        return {'average_daily_gain': 0.0, 'closest_liq': 0.0, 'max_n_hours_stuck': 1000.0}, pd.DataFrame()
    fdf = pd.DataFrame(fills).set_index('trade_id')
    result = prepare_result(fills, ticks, bool(backtest_config['do_long']), bool(backtest_config['do_shrt']))
    result['seconds_elapsed'] = elapsed
    if 'key' not in result:
        result['key'] = calc_candidate_hash_key(backtest_config, backtest_config['ranges'])
    return result, fdf


def prepare_result(fills: list, ticks: np.ndarray, do_long: bool, do_shrt: bool) -> dict:
    fdf = pd.DataFrame(fills)
    if fdf.empty:
        result = {
            'net_pnl_plus_fees': 0,
            'profit_sum': 0,
            'loss_sum': 0,
            'fee_sum': 0,
            'final_equity': 0,
            'gain': 0,
            'max_drawdown': 0,
            'n_days': 0,
            'average_daily_gain': 0,
            'closest_liq': 0,
            'n_fills': 0,
            'n_entries': 0,
            'n_closes': 0,
            'n_reentries': 0,
            'n_stop_losses': 0,
            'biggest_pos_size': 0,
            'max_n_hours_stuck': 0,
            'do_long': do_long,
            'do_shrt': do_shrt
        }
    else:
        result = {
            'net_pnl_plus_fees': fills[-1]['pnl_plus_fees_cumsum'],
            'profit_sum': fills[-1]['profit_cumsum'],
            'loss_sum': fills[-1]['loss_cumsum'],
            'fee_sum': fills[-1]['fee_paid_cumsum'],
            'final_equity': fills[-1]['equity'],
            'gain': (gain := fills[-1]['gain']),
            'max_drawdown': fdf.drawdown.max(),
            'n_days': (n_days := (ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24)),
            'average_daily_gain': gain ** (1 / n_days) if gain > 0.0 else 0.0,
            'closest_liq': fdf.liq_diff.min(),
            'n_fills': len(fills),
            'n_entries': len(fdf[fdf.type.str.contains('entry')]),
            'n_closes': len(fdf[fdf.type == 'close']),
            'n_reentries': len(fdf[fdf.type == 'reentry']),
            'n_stop_losses': len(fdf[fdf.type.str.startswith('stop_loss')]),
            'biggest_pos_size': fdf[['long_pos_size', 'shrt_pos_size']].abs().max(axis=1).max(),
            'max_n_hours_stuck': max(fdf['hours_since_pos_change_max'].max(),
                                     (ticks[-1][2] - fills[-1]['timestamp']) / (1000 * 60 * 60)),
            'do_long': do_long,
            'do_shrt': do_shrt
        }
    return result


def objective_function(result: dict, liq_cap: float, n_daily_entries_cap: int) -> float:
    try:
        return (result['average_daily_gain'] *
                min(1.0, (result['n_entries'] / result['n_days']) / n_daily_entries_cap) *
                min(1.0, result['closest_liq'] / liq_cap))
    except Exception as e:
        print('error with objective function', e, result)
        return 0.0


def create_config(backtest_config: dict) -> dict:
    config = {k: backtest_config[k] for k in backtest_config
              if k not in {'session_name', 'user', 'symbol', 'start_date', 'end_date', 'ranges'}}
    for k in backtest_config['ranges']:
        if backtest_config['ranges'][k][0] == backtest_config['ranges'][k][1]:
            config[k] = backtest_config['ranges'][k][0]
        elif k in ['n_close_orders', 'leverage']:
            config[k] = tune.randint(backtest_config['ranges'][k][0], backtest_config['ranges'][k][1] + 1)
        else:
            config[k] = tune.uniform(backtest_config['ranges'][k][0], backtest_config['ranges'][k][1])
    return config


def clean_start_config(start_config: dict, backtest_config: dict) -> dict:
    clean_start = {}
    for k, v in start_config.items():
        if k in backtest_config and k not in ['do_long', 'do_shrt']:
            clean_start[k] = v
    return clean_start


def clean_result_config(config: dict) -> dict:
    for k, v in config.items():
        if type(v) == np.float64:
            config[k] = float(v)
        if type(v) == np.int64 or type(v) == np.int32 or type(v) == np.int16 or type(v) == np.int8:
            config[k] = int(v)
    return config


def backtest_tune(ticks: np.ndarray, backtest_config: dict, current_best: Union[dict, list] = None):
    config = create_config(backtest_config)
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    session_dirpath = make_get_filepath(os.path.join('reports', backtest_config['exchange'], backtest_config['symbol'],
                                                     f"{n_days}_days_{ts_to_date(time())[:19].replace(':', '')}", ''))
    iters = 10
    if 'iters' in backtest_config:
        iters = backtest_config['iters']
    else:
        print('Parameter iters should be defined in the configuration. Defaulting to 10.')
    num_cpus = 2
    if 'num_cpus' in backtest_config:
        num_cpus = backtest_config['num_cpus']
    else:
        print('Parameter num_cpus should be defined in the configuration. Defaulting to 2.')
    n_particles = 10
    if 'n_particles' in backtest_config:
        n_particles = backtest_config['n_particles']
    phi1 = 1.4962
    phi2 = 1.4962
    omega = 0.7298
    if 'options' in backtest_config:
        phi1 = backtest_config['options']['c1']
        phi2 = backtest_config['options']['c2']
        omega = backtest_config['options']['w']
    current_best_params = []
    if current_best:
        if type(current_best) == list:
            for c in current_best:
                c = clean_start_config(c, config)
                current_best_params.append(c)
        else:
            current_best = clean_start_config(current_best, config)
            current_best_params.append(current_best)

    ray.init(num_cpus=num_cpus)
    pso = ng.optimizers.ConfiguredPSO(transform='identity', popsize=n_particles, omega=omega, phip=phi1, phig=phi2)
    algo = NevergradSearch(optimizer=pso, points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=num_cpus)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(tune.with_parameters(backtest, ticks=ticks), metric='objective', mode='max', name='search',
                        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
                        reuse_actors=True, local_dir=session_dirpath)

    ray.shutdown()
    df = analysis.results_df
    df.reset_index(inplace=True)
    df.drop(columns=['trial_id', 'time_this_iter_s', 'done', 'timesteps_total', 'episodes_total', 'training_iteration',
                     'experiment_id', 'date', 'timestamp', 'time_total_s', 'pid', 'hostname', 'node_ip',
                     'time_since_restore', 'timesteps_since_restore', 'iterations_since_restore', 'experiment_tag'],
            inplace=True)
    df.to_csv(os.path.join(backtest_config['session_dirpath'], 'results.csv'), index=False)
    print('Best candidate found were: ', analysis.best_config)
    plot_wrap(backtest_config, ticks, clean_result_config(analysis.best_config))
    return analysis


def plot_wrap(bc, ticks, candidate):
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    bc['session_dirpath'] = make_get_filepath(os.path.join(
        'plots', bc['exchange'], bc['symbol'],
        f"{n_days}_days_{ts_to_date(time())[:19].replace(':', '')}", ''))
    print('backtesting...')
    result, fdf = backtest_wrap(ticks, {**bc, **{'break_on': {}}, **candidate}, do_print=True)
    if fdf is None or len(fdf) == 0:
        print('no trades')
        return
    fdf.to_csv(bc['session_dirpath'] + f"backtest_trades_{result['key']}.csv")
    print('\nmaking ticks dataframe...')
    df = pd.DataFrame({'price': ticks[:, 0], 'buyer_maker': ticks[:, 1], 'timestamp': ticks[:, 2]})
    dump_plots({**bc, **candidate, **result}, fdf, df)


async def main(args: list):
    config_name = args[1]
    backtest_config = await prep_backtest_config(config_name)
    downloader = Downloader(backtest_config)
    ticks = await downloader.get_ticks(True)
    backtest_config['n_days'] = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    if (p := '--plot') in args:
        try:
            candidate = json.load(open(args[args.index(p) + 1]))
            print('plotting given candidate')
        except Exception as e:
            print(os.listdir(backtest_config['session_dirpath']))
            try:
                candidate = json.load(open(backtest_config['session_dirpath'] + 'live_config.json'))
                print('plotting best candidate')
            except:
                return
        print(json.dumps(candidate, indent=4))
        plot_wrap(backtest_config, ticks, candidate)
        return
    start_candidate = None
    if (s := '--start') in args:
        try:
            if os.path.isdir(args[args.index(s) + 1]):
                start_candidate = [json.load(open(f)) for f in
                                   glob.glob(os.path.join(args[args.index(s) + 1], '*.json'))]
                print('Starting with all configurations in directory.')
            else:
                start_candidate = json.load(open(args[args.index(s) + 1]))
                print('Starting with specified configuration.')
        except:
            print('Could not find specified configuration.')
    if start_candidate:
        backtest_tune(ticks, backtest_config, start_candidate)
    else:
        backtest_tune(ticks, backtest_config)


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
