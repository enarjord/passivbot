import sys
import json
import numpy as np
import pandas as pd
import asyncio
import os
from time import time
from passivbot import init_ccxt, make_get_filepath, ts_to_date, print_, load_settings, \
    sort_dict_keys, round_up, round_dn, round_, calc_long_closes, calc_shrt_closes, \
    calc_long_reentry_price, calc_shrt_reentry_price, calc_diff, calc_default_qty, \
    calc_entry_qty
from binance import fetch_trades as binance_fetch_trades
from bybit import fetch_trades as bybit_fetch_trades
from bybit import calc_cross_long_liq_price as bybit_calc_cross_long_liq_price
from bybit import calc_cross_shrt_liq_price as bybit_calc_cross_shrt_liq_price
from bybit import calc_isolated_long_liq_price as bybit_calc_isolated_long_liq_price
from bybit import calc_isolated_shrt_liq_price as bybit_calc_isolated_shrt_liq_price
from binance import calc_cross_long_liq_price as binance_calc_cross_long_liq_price
from binance import calc_cross_shrt_liq_price as binance_calc_cross_shrt_liq_price
from binance import calc_isolated_long_liq_price as binance_calc_isolated_long_liq_price
from binance import calc_isolated_shrt_liq_price as binance_calc_isolated_shrt_liq_price

from typing import Iterator


def prep_trades_list(df: pd.DataFrame):
    dfc = df[df.price != df.price.shift(1)] # drop consecutive same price trades
    dfc.index = np.arange(len(dfc))
    if 'side' in dfc.columns:
        # bybit
        buyer_maker = dfc.side == 'Sell'
        buyer_maker.name = 'buyer_maker'
    elif 'is_buyer_maker' in dfc.columns:
        # binance
        buyer_maker = dfc.is_buyer_maker
        buyer_maker.name = 'buyer_maker'
    else:
        raise Exception('trades of unknown format')
    dfcc = pd.concat([dfc.price, buyer_maker], axis=1)
    return list(dfcc.to_dict(orient='index').values())


def backtest(trades_list: [dict], settings: dict):
    # trades format is [{price: float, buyer_maker: bool}]

    # no static mode
    grid_spacing = settings['grid_spacing']
    grid_coefficient = settings['grid_coefficient']
    price_step = settings['price_step']
    qty_step = settings['qty_step']
    inverse = settings['inverse']
    liq_diff_threshold = settings['liq_diff_threshold']
    stop_loss_pos_reduction = settings['stop_loss_pos_reduction']
    min_qty = settings['min_qty']
    ddown_factor = settings['ddown_factor']
    leverage = settings['leverage']
    max_leverage = settings['max_leverage']
    maker_fee = settings['maker_fee']
    taker_fee = settings['taker_fee']
    min_markup = settings['min_markup']
    max_markup = settings['max_markup']
    n_close_orders = settings['n_close_orders']

    do_long = settings['do_long']
    do_shrt = settings['do_shrt']

    min_notional = settings['min_notional'] if 'min_notional' in settings else 0.0
    cross_mode = settings['cross_mode'] if 'cross_mode' in settings else True

    if inverse:
        calc_cost = lambda qty_, price_: qty_ / price_
        if settings['cross_mode']:
            calc_liq_price = lambda balance_, pos_size_, pos_price_: \
                bybit_calc_cross_shrt_liq_price(balance_,
                                                pos_size_,
                                                pos_price_,
                                                leverage=max_leverage) \
                    if pos_size_ < 0.0 else \
                    bybit_calc_cross_long_liq_price(balance_,
                                                    pos_size_,
                                                    pos_price_,
                                                    leverage=max_leverage)
        else:
            calc_liq_price = lambda balance_, pos_size_, pos_price_: \
                bybit_calc_isolated_shrt_liq_price(balance_,
                                                   pos_size_,
                                                   pos_price_,
                                                   leverage=leverage) \
                    if pos_size_ < 0.0 else \
                    bybit_calc_isolated_long_liq_price(balance_,
                                                       pos_size_,
                                                       pos_price_,
                                                       leverage=leverage)
        if settings['default_qty'] <= 0.0:
            calc_default_qty_ = lambda balance_, last_price: \
                calc_default_qty(min_qty, qty_step, balance_ * last_price, settings['default_qty'])
        else:
            calc_default_qty_ = lambda balance_, last_price: settings['default_qty']
        calc_max_pos_size = lambda balance_, price_: balance_ * price_ * leverage
    else:
        calc_cost = lambda qty_, price_: qty_ * price_
        if settings['cross_mode']:
            calc_liq_price = lambda balance_, pos_size_, pos_price_: \
                binance_calc_cross_shrt_liq_price(balance_,
                                                  pos_size_,
                                                  pos_price_,
                                                  leverage=leverage) \
                    if pos_size_ < 0.0 else \
                    binance_calc_cross_long_liq_price(balance_,
                                                      pos_size_,
                                                      pos_price_,
                                                      leverage=leverage)
        else:
            calc_liq_price = lambda balance_, pos_size_, pos_price_: \
                binance_calc_isolated_shrt_liq_price(balance_,
                                                     pos_size_,
                                                     pos_price_,
                                                     leverage=leverage) \
                    if pos_size_ < 0.0 else \
                    binance_calc_isolated_long_liq_price(balance_,
                                                         pos_size_,
                                                         pos_price_,
                                                         leverage=leverage)
        if settings['default_qty'] <= 0.0:
            calc_default_qty_ = lambda balance_, last_price: \
                calc_default_qty(max(min_qty, round_up(min_notional / last_price, qty_step)),
                                 qty_step, balance_ / last_price, settings['default_qty'])
        else:
            calc_default_qty_ = lambda balance_, last_price: \
                max(settings['default_qty'], round_up(min_notional / last_price, qty_step))
        calc_max_pos_size = lambda balance_, price_: balance_ / price_ * leverage

    calc_long_reentry_price_ = lambda balance_, pos_margin_, pos_price_, highest_bid_: \
        min(highest_bid_, calc_long_reentry_price(price_step, grid_spacing, grid_coefficient,
                                           balance_, pos_margin_, pos_price_))
    calc_shrt_reentry_price_ = lambda balance_, pos_margin_, pos_price_, lowest_ask_: \
        max(lowest_ask_, calc_shrt_reentry_price(price_step, grid_spacing, grid_coefficient,
                                                 balance_, pos_margin_, pos_price_))
    balance = settings['starting_balance']
    trades = []
    ob = [min(trades_list[0]['price'], trades_list[1]['price']),
          max(trades_list[0]['price'], trades_list[1]['price'])]

    pos_size = 0.0
    pos_price = 0.0

    bid_price = ob[0]
    ask_price = ob[1]

    liq_price = 0.0

    pnl_sum = 0.0
    loss_sum = 0.0
    profit_sum = 0.0

    ema_alpha = 2 / (settings['ema_span'] + 1)
    ema_alpha_ = 1 - ema_alpha
    ema = trades_list[0]['price']

    k = 0
    prev_len_trades = 0
    break_on = {e[0]: eval(e[1]) for e in settings['break_on'] if e[0].startswith('ON:')}
    for t in trades_list:
        if t['buyer_maker']:
            # buy
            if pos_size == 0.0:
                # no pos
                if do_long:
                    bid_price = min(ob[0], round_dn(ema, price_step))
                    bid_qty = calc_default_qty_(balance, ob[0])
                else:
                    bid_price = 0.0
                    bid_qty = 0.0
            elif pos_size > 0.0:
                if calc_diff(liq_price, ob[1]) < liq_diff_threshold and t['price'] <= liq_price:
                    # long liq
                    print(f'break on long liquidation, liq price: {liq_price}')
                    return []
                # long reentry
                bid_qty = calc_entry_qty(qty_step, ddown_factor,
                                         calc_default_qty_(balance, ob[0]),
                                         calc_max_pos_size(balance, ob[0]),
                                         pos_size)
                if bid_qty >= min_qty:
                    pos_margin = calc_cost(pos_size, pos_price) / leverage
                    bid_price = calc_long_reentry_price_(balance, pos_margin, pos_price, ob[0])
                else:
                    bid_price = 0.0
            else:
                # short pos
                if calc_diff(liq_price, ob[0]) < liq_diff_threshold:
                    # short soft stop
                    bid_price = ob[0]
                    bid_qty = round_up(-pos_size * stop_loss_pos_reduction, qty_step)
                else:
                    if t['price'] <= pos_price:
                        # short close
                        qtys, prices = calc_shrt_closes(price_step, qty_step, min_qty, min_markup,
                                                        max_markup, pos_size, pos_price, ob[0],
                                                        n_close_orders)
                        bid_qty = qtys[0]
                        bid_price = prices[0]
                    else:
                        bid_price = 0.0
            ob[0] = t['price']
            if t['price'] < bid_price and bid_qty >= min_qty:
                # filled trade
                cost = calc_cost(bid_qty, bid_price)
                pnl = -cost * maker_fee
                if pos_size >= 0.0:
                    # create or increase long pos
                    trade_side = 'long'
                    trade_type = 'entry'
                    new_pos_size = pos_size + bid_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        bid_price * (bid_qty / new_pos_size)
                    pos_size = new_pos_size
                    roi = 0.0
                else:
                    # close short pos
                    trade_side = 'shrt'
                    gain = pos_price / bid_price - 1
                    pnl += cost * gain
                    if gain > 0.0:
                        trade_type = 'close'
                        profit_sum += pnl
                    else:
                        trade_type = 'stop_loss'
                        loss_sum += pnl
                    pos_size = pos_size + bid_qty
                    roi = gain * leverage
                balance += pnl
                pnl_sum += pnl
                liq_price = calc_liq_price(balance, pos_size, pos_price)
                trades.append({'trade_id': k, 'side': trade_side, 'type': trade_type,
                               'price': bid_price, 'qty': bid_qty, 'pnl': pnl, 'roi': roi,
                               'pos_size': pos_size, 'pos_price': pos_price, 'balance': balance,
                               'max_pos_size': calc_max_pos_size(balance, t['price']),
                               'pnl_sum': pnl_sum, 'loss_sum': loss_sum, 'profit_sum': profit_sum,
                               'progress': k / len(trades_list),
                               'liq_price': liq_price, 'liq_diff': calc_diff(liq_price, t['price'])})
        else:
            # sell
            if pos_size == 0.0:
                # no pos
                if do_shrt:
                    ask_price = max(ob[1], round_up(ema, price_step))
                    ask_qty = -calc_default_qty_(balance, ob[1])
                else:
                    ask_price = 9e9
                    ask_qty = 0.0
            elif pos_size > 0.0:
                # long pos
                if calc_diff(liq_price, ob[1]) < liq_diff_threshold:
                    # long soft stop
                    ask_price = ob[1]
                    ask_qty = -round_up(pos_size * stop_loss_pos_reduction, qty_step)
                else:
                    if t['price'] >= pos_price:
                        # long close
                        qtys, prices = calc_long_closes(price_step, qty_step, min_qty, min_markup,
                                                        max_markup, pos_size, pos_price, ob[1],
                                                        n_close_orders)
                        ask_qty = qtys[0]
                        ask_price = prices[0]
                    else:
                        ask_price = 9e9
            else:
                if calc_diff(liq_price, ob[1]) < liq_diff_threshold and t['price'] >= liq_price:
                    # shrt liq
                    print(f'break on shrt liquidation, liq price: {liq_price}')
                    return []
                # shrt reentry
                ask_qty = -calc_entry_qty(qty_step, ddown_factor,
                                          calc_default_qty_(balance, ob[1]),
                                          calc_max_pos_size(balance, ob[1]),
                                          pos_size)
                if -ask_qty >= min_qty:
                    pos_margin = calc_cost(-pos_size, pos_price) / leverage
                    ask_price = calc_shrt_reentry_price_(balance, pos_margin, pos_price, ob[0])
                else:
                    ask_price = 9e9
            ob[1] = t['price']
            if t['price'] > ask_price and abs(ask_qty) >= min_qty:
                # filled trade
                cost = calc_cost(-ask_qty, ask_price)
                pnl = -cost * maker_fee
                if pos_size <= 0.0:
                    # create or increase shrt pos
                    trade_side = 'shrt'
                    trade_type = 'entry'
                    new_pos_size = pos_size + ask_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        ask_price * (ask_qty / new_pos_size)
                    pos_size = new_pos_size
                    roi = 0.0
                else:
                    # close long pos
                    trade_side = 'long'
                    gain = ask_price / pos_price - 1
                    pnl += cost * gain
                    if gain > 0.0:
                        trade_type = 'close'
                        profit_sum += pnl
                    else:
                        trade_type = 'stop_loss'
                        loss_sum += pnl
                    pos_size = pos_size + ask_qty
                    roi = gain * leverage
                balance += pnl
                pnl_sum += pnl
                liq_price = calc_liq_price(balance, pos_size, pos_price)
                trades.append({'trade_id': k, 'side': trade_side, 'type': trade_type,
                               'price': ask_price, 'qty': ask_qty, 'pnl': pnl, 'roi': roi,
                               'pos_size': pos_size, 'pos_price': pos_price, 'balance': balance,
                               'max_pos_size': calc_max_pos_size(balance, t['price']),
                               'pnl_sum': pnl_sum, 'loss_sum': loss_sum, 'profit_sum': profit_sum,
                               'progress': k / len(trades_list),
                               'liq_price': liq_price, 'liq_diff': calc_diff(liq_price, t['price'])})
        ema = ema * ema_alpha_ + t['price'] * ema_alpha
        k += 1
        if k % 10000 == 0 or len(trades) != prev_len_trades:
            for key, condition in break_on.items():
                if condition(trades[-1]):
                    print('break on', key)
                    return []
            balance = max(balance, settings['starting_balance'])
            prev_len_trades = len(trades)
            progress = k / len(trades_list)
            line = f"\r{progress:.3f} pnl sum {pnl_sum:.8f} "
            line += f"loss sum {loss_sum:.5f} balance {balance:.5f} "
            plr = trades[-1]['profit_sum'] / ls_ if \
                (ls_ := abs(trades[-1]['loss_sum'])) > 0.0 else 9.0
            line += f"profit to loss ratio {plr:.3f} "
            line += f"qty {calc_default_qty_(balance, ob[0]):.4f} "
            line += f"max pos pct {abs(pos_size) / calc_max_pos_size(balance, t['price']):.3f} "
            line += f"liq diff {min(1.0, calc_diff(trades[-1]['liq_price'], ob[0])):.3f} "
            line += f"pos size {pos_size:.4f} "
            print(line, end=' ')
    return trades


def format_dict(d: dict):
    r = ''
    for key in sorted(d):
        r += f'&{key}={round(d[key], 10) if type(d[key]) in [float, int] else str(d[key])}'
    return r[1:]

def unformat_dict(d: str):
    kv = d.split('&')
    kvs = [kv.split('=') for kv in d.split('&')]
    result = {}
    for kv in kvs:
        try:
            result[kv[0]] = eval(kv[1])
        except:
            result[kv[0]] = kv[1]
    return result


def jackrabbit(trades_list: [dict],
               backtesting_settings: dict,
               ranges: dict,
               base_filepath: str):
    best_filepath = base_filepath + 'best.json'
    if backtesting_settings['random_starting_candidate']:
        best = {key: calc_new_val((abs(ranges[key][1]) - abs(ranges[key][0])) / 2, ranges[key], 1.0)
                for key in sorted(ranges)}
        print('\nrandom starting candidate:', best)
    elif os.path.exists(best_filepath):
        best = json.load(open(best_filepath))
        print('\nloaded best candidate', best)
        print()
    else:
        best = sort_dict_keys({k_: backtesting_settings[k_] for k_ in ranges})

    n_days = backtesting_settings['n_days']
    results = {}
    best_gain = -9e9
    candidate = best

    ks = backtesting_settings['n_jackrabbit_iterations']
    k = backtesting_settings['starting_k']
    ms = np.array([1 / (i / 2 + 16) for i in range(ks)])
    ms = ((ms - ms.min()) / (ms.max() - ms.min()))
    trades_filepath = make_get_filepath(os.path.join(base_filepath, 'backtest_trades', ''))
    json.dump(backtesting_settings, open(base_filepath + 'backtesting_settings.json', 'w'),
              indent=4, sort_keys=True)
    results_filename = base_filepath + 'results.txt'    

    print('\n', backtesting_settings, '\n\n')

    while k < ks:
        mutation_coefficient = ms[k]
        if candidate['min_markup'] >= candidate['max_markup']:
            candidate['min_markup'] = candidate['max_markup']

        settings_ = {**backtesting_settings, **candidate}
        key = format_dict(candidate)
        if key in results:
            print('\nskipping', key)
            if os.path.exists(best_filepath):
                best = json.load(open(best_filepath))
            candidate = get_new_candidate(ranges, best)
            continue
        print(f'\nk={k}, m={mutation_coefficient:.4f} candidate:\n', candidate)
        start_time = time()
        trades = backtest(trades_list, settings_)
        print('\ntime elapsed', round(time() - start_time, 1), 'seconds')
        k += 1
        if not trades:
            print('\nno trades')
            best = json.load(open(best_filepath))
            candidate = get_new_candidate(ranges, best)
            continue
        tdf = pd.DataFrame(trades).set_index('trade_id')
        tdf.to_csv(trades_filepath + key + '.csv')
        closest_liq = ((tdf.price - tdf.liq_price).abs() / tdf.price).min()
        biggest_pos_size = tdf.pos_size.abs().max()
        n_closes = len(tdf[tdf.type == 'close'])
        pnl_sum = tdf.pnl.sum()
        loss_sum = tdf[tdf.type == 'stop_loss'].pnl.sum()
        abs_pos_sizes = tdf.pos_size.abs()
        gain = (pnl_sum + settings_['starting_balance']) / settings_['starting_balance']
        average_daily_gain = gain ** (1 / n_days)
        n_trades = len(tdf)
        result = {'n_closes': n_closes, 'pnl_sum': pnl_sum, 'loss_sum': loss_sum,
                  'average_daily_gain': average_daily_gain,
                  'gain': gain, 'n_trades': n_trades, 'closest_liq': closest_liq,
                  'biggest_pos_size': biggest_pos_size, 'n_days': n_days}
        print('\n\n', result)
        results[key] = {**result, **candidate}

        if os.path.exists(best_filepath):
            best = json.load(open(best_filepath))
            best_gain = best['gain']

        if gain > best_gain:
            best = candidate
            best_gain = gain
            print('\n\n\n###############\nnew best', best, '\naverage daily gain:',
                  round(average_daily_gain, 5), '\n\n')
            print(settings_, '\n')
            print(results[key], '\n\n')
            default_live_settings = load_settings(settings_['exchange'], print_=False)
            live_settings = {k: settings_[k] if k in settings_ else default_live_settings[k]
                             for k in default_live_settings}
            live_settings['indicator_settings'] = {'tick_ema': {'span': best['ema_span']}}
            json.dump(live_settings,
                      open(base_filepath + 'best_result_live_settings.json', 'w'),
                      indent=4, sort_keys=True)
            print('\n\n', json.dumps(live_settings, indent=4, sort_keys=True), '\n\n')
            json.dump(results[key], open(base_filepath + 'best_result.json', 'w'),
                      indent=4, sort_keys=True)
            json.dump({**{'gain': result['gain']}, **best}, open(best_filepath, 'w'),
                      indent=4, sort_keys=True)
        candidate = get_new_candidate(ranges, best, m=mutation_coefficient)
        with open(results_filename, 'a') as f:
            f.write(json.dumps(results[key]) + '\n')


def calc_new_val(val, range_, m):
    choice_span = (range_[1] - range_[0]) * m / 2
    biased_mid_point = max(range_[0] + choice_span, min(val, range_[1] - choice_span))
    choice_range = (biased_mid_point - choice_span, biased_mid_point + choice_span)
    new_val = np.random.choice(np.linspace(choice_range[0], choice_range[1], 200))
    return round_(new_val, range_[2])


def get_new_candidate(ranges: dict, best: dict, m=0.2):
    new_candidate = {}
    for key in best:
        if key not in ranges:
            continue
        if type(best[key]) == tuple:
            new_candidate[key] = tuple(sorted([calc_new_val(e, ranges[key], m) for e in best[key]]))
        else:
            new_candidate[key] = calc_new_val(best[key], ranges[key], m)
    return {k_: new_candidate[k_] for k_ in sorted(new_candidate)}


def get_downloaded_trades(filepath: str, age_limit_millis: float) -> (pd.DataFrame, dict):
    if os.path.isdir(filepath):
        filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')],
                           key=lambda x: int(x[:x.find('_')].replace('.cs', '').replace('v', '')))
        chunks = []
        chunk_lengths = {}
        for f in filenames[::-1]:
            chunk = pd.read_csv(filepath + f).set_index('trade_id')
            chunk_lengths[f] = len(chunk)
            print('\rloaded chunk of trades', f, ts_to_date(chunk.timestamp.iloc[0] / 1000),
                  end='     ')
            chunks.append(chunk)
            if chunk.timestamp.iloc[0] < age_limit_millis:
                break
        if chunks:
            df = pd.concat(chunks, axis=0).sort_index()
            return df[~df.index.duplicated()], chunk_lengths
        else:
            return None, {}
    else:
        return None, {}


async def load_trades(exchange: str, user: str, symbol: str, n_days: float) -> pd.DataFrame:

    def skip_ids(id_, ids_):
        if id_ in ids_:
            print('skipping from', id_)
            while id_ in ids_:
                id_ -= 1
            print('           to', id_)
        return id_

    def load_cache():
        cache_filenames = [f for f in os.listdir(cache_filepath) if '.csv' in f]
        if cache_filenames:
            print('loading cached trades')
            cache_df = pd.concat([pd.read_csv(cache_filepath + f) for f in cache_filenames], axis=0)
            cache_df = cache_df.set_index('trade_id')
            return cache_df
        return None

    if exchange == 'binance':
        fetch_trades_func = binance_fetch_trades
    elif exchange == 'bybit':
        fetch_trades_func = bybit_fetch_trades
    else:
        print(exchange, 'not found')
        return
    cc = init_ccxt(exchange, user)
    filepath = make_get_filepath(os.path.join('historical_data', exchange, 'agg_trades_futures',
                                              symbol, ''))
    cache_filepath = make_get_filepath(filepath.replace(symbol, symbol + '_cache'))
    age_limit = time() - 60 * 60 * 24 * n_days
    age_limit_millis = age_limit * 1000
    print('age_limit', ts_to_date(age_limit))
    cache_df = load_cache()
    trades_df, chunk_lengths = get_downloaded_trades(filepath, age_limit_millis)
    ids = set()
    if trades_df is not None:
        ids.update(trades_df.index)
    if cache_df is not None:
        ids.update(cache_df.index)
    gaps = []
    if trades_df is not None and len(trades_df) > 0:
        # 
        sids = sorted(ids)
        for i in range(1, len(sids)):
            if sids[i-1] + 1 != sids[i]:
                gaps.append((sids[i-1], sids[i]))
        if gaps:
            print('gaps', gaps)
        # 
    prev_fetch_ts = time()
    new_trades = await fetch_trades_func(cc, symbol)
    k = 0
    while True:
        k += 1
        if (break_ := new_trades[0]['timestamp'] <= age_limit_millis) or k % 20 == 0:
            print('caching trades...')
            new_tdf = pd.DataFrame(new_trades).set_index('trade_id')
            cache_filename = f'{cache_filepath}{new_tdf.index[0]}_{new_tdf.index[-1]}.csv'
            new_tdf.to_csv(cache_filename)
            new_trades = [new_trades[0]]
            if break_:
                break
        from_id = skip_ids(new_trades[0]['trade_id'] - 1, ids) - 999
        # wait at least 0.75 sec between each fetch
        sleep_for = max(0.0, 0.75 - (time() - prev_fetch_ts))
        await asyncio.sleep(sleep_for)
        prev_fetch_ts = time()
        new_trades = await fetch_trades_func(cc, symbol, from_id=from_id) + new_trades
        ids.update([e['trade_id'] for e in new_trades])
    tdf = pd.concat([load_cache(), trades_df], axis=0).sort_index()
    tdf = tdf[~tdf.index.duplicated()]
    dump_chunks(filepath, tdf, chunk_lengths)
    cache_filenames = [f for f in os.listdir(cache_filepath) if '.csv' in f]
    print('removing cache...\n')
    for filename in cache_filenames:
        print(f'\rremoving {filename}', end='   ')
        os.remove(cache_filepath + filename)
    await cc.close()
    return tdf[tdf.timestamp >= age_limit_millis]


def dump_chunks(filepath: str, tdf: pd.DataFrame, chunk_lengths: dict, chunk_size=100000):
    chunk_ids = tdf.index // chunk_size * chunk_size
    for g in tdf.groupby(chunk_ids):
        filename = f'{g[1].index[0]}_{g[1].index[-1]}.csv'
        if filename not in chunk_lengths or chunk_lengths[filename] != chunk_size:
            print('dumping chunk', filename)
            g[1].to_csv(f'{filepath}{filename}')


async def main():
    exchange = sys.argv[1]
    user = sys.argv[2]

    settings_filepath = os.path.join('backtesting_settings', exchange, '')
    backtesting_settings = \
        json.load(open(os.path.join(settings_filepath, 'backtesting_settings.json')))

    try:
        session_name = sys.argv[3]
        print('\n\nusing given session name.', session_name, '\n\n')
    except IndexError:
        session_name = backtesting_settings['session_name']
        print('\n\nusing session name from backtesting_settings.json.', session_name, '\n\n')

    symbol = backtesting_settings['symbol']
    n_days = backtesting_settings['n_days']
    ranges = json.load(open(os.path.join(settings_filepath, 'ranges.json')))
    print(settings_filepath)
    results_filepath = make_get_filepath(
        os.path.join('backtesting_results', exchange, symbol, session_name, '')
    )
    print(results_filepath)
    trades_list_filepath = os.path.join(results_filepath, f"{n_days}_days_trades_list_cache.npy")
    if os.path.exists(trades_list_filepath):
        print('loading cached trade list', trades_list_filepath)
        trades_list = np.load(trades_list_filepath, allow_pickle=True)
    else:
        agg_trades = await load_trades(exchange, user, symbol, n_days)
        print('preparing trades...')
        trades_list = prep_trades_list(agg_trades)
        np.save(trades_list_filepath, trades_list)
    jackrabbit(trades_list, backtesting_settings, ranges, results_filepath)


if __name__ == '__main__':
    asyncio.run(main())

